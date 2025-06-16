import glob
import json
import math
import os

import lightning as L
import torch
import torch.nn as nn
from PIL import Image
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from src.callbacks.ema import EMACallback

# Constants for model
CNN_INPUT_WIDTH = 64
CNN_OUTPUT_WIDTH = 32
CNN_OUTPUT_CHANNELS = 32  # Based on Fig 3, final upsampled feature map
TRANSFORMER_D_MODEL = CNN_OUTPUT_WIDTH * CNN_OUTPUT_CHANNELS  # 32 * 32 = 1024 based on Fig 3
# If d_model needs to be 2048 as mentioned in text (Fig 2 example),
# then CNN_OUTPUT_CHANNELS should be 64. Let's use 1024 for now based on Fig 3.
TRANSFORMER_N_HEADS = 8  # From Fig 2
TRANSFORMER_NUM_ENCODER_LAYERS = 6  # From Fig 2
TRANSFORMER_NUM_DECODER_LAYERS = 6  # Common for encoder-decoder, not explicitly stated for 3rd design

MAX_IMAGE_HEIGHT = 1024  # Fixed padding height
MAX_LABEL_LENGTH = 256  # Max sequence length for tokenizer


# --- 1. Modules for CNN Frontend ---
class RBlock(nn.Module):
    """
    Residual Block based on Fig. 3 and general Residual Network principles.
    Consists of Conv2d, BatchNorm, ReLU, and a skip connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If stride or channels change, adjust shortcut to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class UpConv(nn.Module):
    """
    Upsampling block as shown in Fig. 3.
    Uses ConvTranspose2d (or nn.Upsample + Conv2d).
    Let's use ConvTranspose2d for learned upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))


class CNNFrontend(nn.Module):
    """
    CNN Frontend based on Fig. 3 (U-Net like architecture).
    Input: (B, 1, H, 64) -> Output: (B, 32, H', 32)
    Where H' = H / 2 based on diagram (1024 -> 512).
    """

    def __init__(self, in_channels=1, output_channels=CNN_OUTPUT_CHANNELS):
        super().__init__()

        # Encoder Path
        # Input: 1xHx64
        self.enc1 = RBlock(in_channels, 32, stride=1, padding=1)  # -> 32xHx64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32xH/2x32

        self.enc2 = RBlock(32, 64, stride=1, padding=1)  # -> 64xH/2x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64xH/4x16

        self.enc3 = RBlock(64, 128, stride=1, padding=1)  # -> 128xH/4x16
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128xH/8x8

        self.enc4 = RBlock(128, 256, stride=1, padding=1)  # -> 256xH/8x8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256xH/16x4 (Bottleneck)

        # Bottleneck (additional RBlock before upsampling starts in Fig. 3's bottleneck part)
        self.bottleneck = RBlock(256, 256, stride=1, padding=1)  # -> 256xH/16x4

        # Decoder Path (UpConv + skip connection fusion + RBlock)
        # Note: Fig 3 shows a specific data flow for UpConvs and RBlocks
        # without explicit skip connection fusion as typically seen in U-Net diagrams.
        # Following the data flow shown in Fig 3.
        # From 256xH/16x4 -> UpConv -> 128xH/8x8
        self.upconv1 = UpConv(256, 128, kernel_size=4, stride=2, padding=1)  # Output H/8, W=8
        self.dec1 = RBlock(128, 128, stride=1, padding=1)  # -> 128xH/8x8

        # From 128xH/8x8 -> UpConv -> 64xH/4x16
        self.upconv2 = UpConv(128, 64, kernel_size=4, stride=2, padding=1)  # Output H/4, W=16
        self.dec2 = RBlock(64, 64, stride=1, padding=1)  # -> 64xH/4x16

        # From 64xH/4x16 -> UpConv -> 32xH/2x32
        # Final output of CNN Frontend as per Fig. 3 is 32xL/2x32
        self.upconv3 = UpConv(64, output_channels, kernel_size=4, stride=2, padding=1)  # Output H/2, W=32
        self.dec3 = RBlock(output_channels, output_channels, stride=1, padding=1)  # -> 32xH/2x32

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)  # 32xH/2x32

        x2 = self.enc2(x)
        x = self.pool2(x2)  # 64xH/4x16

        x3 = self.enc3(x)
        x = self.pool3(x3)  # 128xH/8x8

        x4 = self.enc4(x)
        x = self.pool4(x4)  # 256xH/16x4

        # Bottleneck
        x = self.bottleneck(x)  # 256xH/16x4

        # Decoder
        x = self.upconv1(x)  # 128xH/8x8
        x = self.dec1(x)  # 128xH/8x8

        x = self.upconv2(x)  # 64xH/4x16
        x = self.dec2(x)  # 64xH/4x16

        x = self.upconv3(x)  # 32xH/2x32
        x = self.dec3(x)  # 32xH/2x32

        return x  # (B, 32, H/2, 32)


# --- 2. Transformer Model ---
class KuzushijiTransformer(nn.Module):
    """
    Transformer Encoder-Decoder model for Kuzushiji recognition.
    Based on Fig. 2 and Fig. 4(c) of the paper.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int = 2048,  # Common default for Transformer
        dropout: float = 0.1,
        vocab_size: int = 0,  # To be set by tokenizer
        max_target_seq_len: int = MAX_LABEL_LENGTH + 2,
    ):  # +2 for BOS/EOS tokens
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.max_target_seq_len = max_target_seq_len

        # Positional encoding for image features (encoder input)
        # Using a simple learned positional encoding
        # This will be added to the reshaped CNN features (B, H_out, d_model)
        self.image_pos_encoding = nn.Parameter(torch.randn(1, MAX_IMAGE_HEIGHT // 2, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Query embedding for decoder (fixed queries as per DETR inspiration)
        # The paper says "Query bounding boxes are input to the Transformer decoder" and
        # "Query-box contents are zero with positional encodings".
        # This implies fixed, learned query embeddings whose "positional encodings" are their inherent values.
        self.query_embed = nn.Embedding(max_target_seq_len, d_model)  # Learned query embeddings

        # Output linear layers from Encoder output (as per Fig. 2 "OUT" fan-out)
        # These layers project each encoder output element to its respective prediction.
        self.encoder_char_head = nn.Linear(d_model, vocab_size)  # L_map-code (character classification per row)
        self.encoder_top_head = nn.Linear(d_model, 1)  # L_top (binary: 0 or 1, top of char)
        self.encoder_box_head = nn.Linear(d_model, CNN_OUTPUT_WIDTH)  # L_box (occupancy mask of width 32)

        # Output linear layer from Decoder output (as per Fig. 4(c) "Seq of Code")
        self.decoder_char_head = nn.Linear(d_model, vocab_size)  # L_seq-code (sequence generation)

    def forward(self, src_features, tgt_label_ids=None):
        """
        src_features: (B, C_out, H_out, W_out) from CNNFrontend -> (B, H_out, C_out * W_out) for Transformer
        tgt_label_ids: (B, L_tgt) target sequence for decoder, for training (used for teacher forcing, if applicable)
        """
        B, C_out, H_out, W_out = src_features.shape

        # Reshape CNN features for Encoder input: (B, H_out, C_out * W_out)
        # Permute to (B, H_out, W_out, C_out) then reshape to (B, H_out, W_out * C_out)
        src = src_features.permute(0, 2, 3, 1).reshape(B, H_out, C_out * W_out)

        # Add positional encoding to encoder input
        # Ensure image_pos_encoding matches the dynamic H_out
        src_with_pos = src + self.image_pos_encoding[:, :H_out, :]

        # Transformer Encoder forward
        encoder_output = self.transformer_encoder(src_with_pos)  # (B, H_out, d_model)

        # Encoder output heads (L_map-code, L_top, L_box)
        encoder_char_logits = self.encoder_char_head(encoder_output)  # (B, H_out, vocab_size)
        encoder_top_logits = self.encoder_top_head(encoder_output).squeeze(-1)  # (B, H_out)
        encoder_box_logits = self.encoder_box_head(encoder_output)  # (B, H_out, CNN_OUTPUT_WIDTH)

        # Transformer Decoder forward
        # Create query embeddings (positional embeddings only, content is zero as per DETR inspiration)
        tgt_queries = self.query_embed(torch.arange(self.max_target_seq_len, device=src.device)).unsqueeze(0).repeat(B, 1, 1)

        # In training, the decoder generates a sequence of predictions using teacher forcing.
        # The `tgt_label_ids` is the ground truth target sequence for loss calculation.
        # The attention mask (`tgt_mask`) typically prevents decoder from seeing future tokens.
        # For simplicity here, we assume direct generation and `tgt_label_ids` is just for loss.
        # No explicit `tgt_mask` is used in this forward pass, assuming the queries implicitly handle sequence generation.
        decoder_output = self.transformer_decoder(tgt_queries, encoder_output)  # (B, max_target_seq_len, d_model)
        decoder_char_logits = self.decoder_char_head(decoder_output)  # (B, max_target_seq_len, vocab_size)

        return {
            "encoder_char_logits": encoder_char_logits,
            "encoder_top_logits": encoder_top_logits,
            "encoder_box_logits": encoder_box_logits,
            "decoder_char_logits": decoder_char_logits,
        }


# --- 3. Dataset ---
class KuzushijiDataset(Dataset):
    def __init__(self, data_root: str, split: str, tokenizer: PreTrainedTokenizerFast):
        self.image_dir = os.path.join(data_root, split, "images")
        self.label_dir = os.path.join(data_root, split, "labels")
        self.bbox_dir = os.path.join(data_root, split, "bounding_boxes")
        self.tokenizer = tokenizer

        self.image_files = []
        self.label_files = []
        self.bbox_files = []

        # List all image files and find corresponding label and bbox files
        # Traverse BookID subdirectories
        for book_id_dir in os.listdir(self.image_dir):
            full_image_book_path = os.path.join(self.image_dir, book_id_dir)
            if os.path.isdir(full_image_book_path):
                for img_file in glob.glob(os.path.join(full_image_book_path, "*.jpg")):
                    # Assuming consistent naming convention: .jpg, .txt, .json
                    base_name = os.path.basename(img_file).replace(".jpg", "")
                    label_path = os.path.join(self.label_dir, book_id_dir, base_name + ".txt")
                    bbox_path = os.path.join(self.bbox_dir, book_id_dir, base_name + ".json")

                    if os.path.exists(label_path) and os.path.exists(bbox_path):
                        self.image_files.append(img_file)
                        self.label_files.append(label_path)
                        self.bbox_files.append(bbox_path)
                    else:
                        pass
                        # print(f"Warning: Missing label or bbox file for {base_name}. Skipping.")

        # Ensure all lists have the same length
        assert len(self.image_files) == len(self.label_files) == len(self.bbox_files), (
            "Mismatch in number of image, label, or bbox files."
        )
        print(f"Found {len(self.image_files)} samples in {split} split.")

        # Scaling factors from original image dimensions to CNN output feature map dimensions
        self.cnn_output_height_scale = (MAX_IMAGE_HEIGHT / 2) / MAX_IMAGE_HEIGHT  # 512 / 1024 = 0.5
        self.cnn_output_width_scale = CNN_OUTPUT_WIDTH / CNN_INPUT_WIDTH  # 32 / 64 = 0.5

        self.padding_token_id = tokenizer.pad_token_id
        # Define a special token for background/no character, if padding_token_id is not suitable.
        # For now, using pad_token_id as background.

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        bbox_path = self.bbox_files[idx]

        # Load image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        # The prompt states images are already padded to 1024x64.
        # Convert to tensor and normalize
        image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)

        # Load label
        with open(label_path, encoding="utf-8") as f:
            label_text = f.read().strip()

        # Load bounding boxes
        with open(bbox_path, encoding="utf-8") as f:
            bboxes = json.load(f)  # List of [x_min, y_min, x_max, y_max]

        # Tokenize label_text for L_seq-code target
        tokenized_label = self.tokenizer(
            label_text,
            max_length=MAX_LABEL_LENGTH + 2,  # +2 for BOS/EOS tokens
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,  # Add BOS/EOS
        )
        label_ids_target = tokenized_label["input_ids"].squeeze(0)  # (MAX_LABEL_LENGTH + 2,)

        # Prepare targets for Encoder output: L_map-code, L_top, L_box
        # CNN output height is MAX_IMAGE_HEIGHT // 2 = 512
        cnn_out_height = MAX_IMAGE_HEIGHT // 2

        # Initialize targets with padding/background values
        # L_map-code: character ID for each row. Use tokenizer.pad_token_id as background/no char
        map_code_target = torch.full((cnn_out_height,), self.padding_token_id, dtype=torch.long)
        # L_top: binary (0 or 1) for top of char. 0 for no top, 1 for top.
        top_target = torch.zeros(cnn_out_height, dtype=torch.float32)
        # L_box: float mask for x-occupancy. (cnn_out_height, CNN_OUTPUT_WIDTH)
        box_occupancy_target = torch.zeros((cnn_out_height, CNN_OUTPUT_WIDTH), dtype=torch.float32)

        # Iterate through bounding boxes to populate targets
        # Assuming bboxes are ordered by reading direction (top-to-bottom)
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox

            # Scale bounding box coordinates to CNN feature map dimensions
            # floor for min, ceil for max to ensure coverage
            scaled_x_min = math.floor(x_min * self.cnn_output_width_scale)
            scaled_y_min = math.floor(y_min * self.cnn_output_height_scale)
            scaled_x_max = math.ceil(x_max * self.cnn_output_width_scale)
            scaled_y_max = math.ceil(y_max * self.cnn_output_height_scale)

            # Clamp scaled coordinates to valid feature map dimensions
            scaled_x_min = max(0, min(scaled_x_min, CNN_OUTPUT_WIDTH))
            scaled_y_min = max(0, min(scaled_y_min, cnn_out_height))
            scaled_x_max = max(0, min(scaled_x_max, CNN_OUTPUT_WIDTH))  # exclusive upper bound
            scaled_y_max = max(0, min(scaled_y_max, cnn_out_height))  # exclusive upper bound

            # Get character ID for this bounding box
            # Assuming bboxes are aligned with characters in label_text.
            # If label_text is shorter than bboxes, it's an edge case, treat as padding.
            char_id = self.padding_token_id  # Default to padding
            if i < len(label_text):
                # Tokenize single character to get its ID.
                # Handle cases where character might not be in vocab (UNK)
                char_tokens = self.tokenizer.encode(label_text[i], add_special_tokens=False)
                if char_tokens:
                    char_id = char_tokens[0]

            # Populate map_code_target and box_occupancy_target for the rows covered by this bbox
            for y_out in range(scaled_y_min, scaled_y_max):
                if 0 <= y_out < cnn_out_height:  # Ensure within bounds
                    # L_map-code target: assign character ID to rows covered by the character
                    map_code_target[y_out] = char_id

                    # L_box target: create a mask for x-occupancy
                    # The range needs to be clamped to valid indices for slicing
                    effective_x_min = max(0, scaled_x_min)
                    effective_x_max = min(CNN_OUTPUT_WIDTH, scaled_x_max)
                    if effective_x_min < effective_x_max:
                        box_occupancy_target[y_out, effective_x_min:effective_x_max] = 1.0

            # L_top target: Mark the very first row of the character's scaled bounding box
            if 0 <= scaled_y_min < cnn_out_height:
                top_target[scaled_y_min] = 1.0  # Mark the beginning of the character

        return {
            "image": image_tensor,
            "label_ids": label_ids_target,  # For L_seq-code
            "map_code_target": map_code_target,  # For L_map-code
            "top_target": top_target,  # For L_top
            "box_occupancy_target": box_occupancy_target,  # For L_box
            "original_label_text": label_text,  # For debugging/logging
        }


# --- 4. Lightning Module ---
class OCRModel(L.LightningModule):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, learning_rate: float = 1e-5, batch_size: int = 16):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cnn_frontend = CNNFrontend()
        self.transformer = KuzushijiTransformer(
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_N_HEADS,
            num_encoder_layers=TRANSFORMER_NUM_ENCODER_LAYERS,
            num_decoder_layers=TRANSFORMER_NUM_DECODER_LAYERS,
            vocab_size=self.tokenizer.vocab_size,
            max_target_seq_len=MAX_LABEL_LENGTH + 2,  # +2 for BOS/EOS tokens
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Using BCEWithLogitsLoss for L_top and L_box as they are binary/mask predictions
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, image, label_ids=None):
        cnn_features = self.cnn_frontend(image)  # (B, C_out, H_out, W_out)

        # Transformer forward requires reshaping cnn_features
        transformer_outputs = self.transformer(cnn_features, label_ids)

        return transformer_outputs

    def _calculate_loss(self, outputs, targets):
        # L_map-code (from encoder, character classification per slice)
        # Permute (B, H_out, vocab_size) to (B, vocab_size, H_out) for CrossEntropyLoss
        loss_map_code = self.ce_loss(outputs["encoder_char_logits"].permute(0, 2, 1), targets["map_code_target"])

        # L_top (from encoder, top position detection per slice)
        loss_top = self.bce_loss(outputs["encoder_top_logits"], targets["top_target"])

        # L_box (from encoder, bounding box x-occupancy per slice)
        loss_box = self.bce_loss(outputs["encoder_box_logits"], targets["box_occupancy_target"])

        # L_seq-code (from decoder, sequence generation)
        # Permute (B, max_target_seq_len, vocab_size) to (B, vocab_size, max_target_seq_len) for CrossEntropyLoss
        loss_seq_code = self.ce_loss(outputs["decoder_char_logits"].permute(0, 2, 1), targets["label_ids"])

        # Total loss as per Equation (2): Loss_sequence = L_map-code + L_top + L_box + L_seq-code
        total_loss = loss_map_code + loss_top + loss_box + loss_seq_code
        return total_loss, loss_map_code, loss_top, loss_box, loss_seq_code

    def _calculate_accuracy(self, outputs, targets):
        """各タスクの精度を計算"""
        # 1. L_map-code精度 (文字分類精度)
        map_code_predictions = torch.argmax(outputs["encoder_char_logits"], dim=-1)  # (B, H_out)
        map_code_targets = targets["map_code_target"]  # (B, H_out)

        # padding tokenを除外して精度計算
        valid_mask = map_code_targets != self.tokenizer.pad_token_id
        if valid_mask.sum() > 0:
            map_code_accuracy = (map_code_predictions[valid_mask] == map_code_targets[valid_mask]).float().mean()
        else:
            map_code_accuracy = torch.tensor(0.0, device=map_code_predictions.device)

        # 2. L_top精度 (二値分類精度)
        top_predictions = (torch.sigmoid(outputs["encoder_top_logits"]) > 0.5).float()  # (B, H_out)
        top_targets = targets["top_target"]  # (B, H_out)
        top_accuracy = (top_predictions == top_targets).float().mean()

        # 3. L_box精度 (二値分類精度)
        box_predictions = (torch.sigmoid(outputs["encoder_box_logits"]) > 0.5).float()  # (B, H_out, W_out)
        box_targets = targets["box_occupancy_target"]  # (B, H_out, W_out)
        box_accuracy = (box_predictions == box_targets).float().mean()

        # 4. L_seq-code精度 (シーケンス生成精度)
        seq_predictions = torch.argmax(outputs["decoder_char_logits"], dim=-1)  # (B, max_seq_len)
        seq_targets = targets["label_ids"]  # (B, max_seq_len)

        # padding tokenを除外して精度計算
        seq_valid_mask = seq_targets != self.tokenizer.pad_token_id
        if seq_valid_mask.sum() > 0:
            seq_accuracy = (seq_predictions[seq_valid_mask] == seq_targets[seq_valid_mask]).float().mean()
        else:
            seq_accuracy = torch.tensor(0.0, device=seq_predictions.device)

        return map_code_accuracy, top_accuracy, box_accuracy, seq_accuracy

    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"], batch["label_ids"])
        total_loss, loss_map_code, loss_top, loss_box, loss_seq_code = self._calculate_loss(outputs, batch)
        self.log(
            "train/total_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/loss_map_code",
            loss_map_code,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/loss_top", loss_top, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "train/loss_box", loss_box, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "train/loss_seq_code",
            loss_seq_code,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"], batch["label_ids"])
        total_loss, loss_map_code, loss_top, loss_box, loss_seq_code = self._calculate_loss(outputs, batch)

        # 精度計算
        acc_map_code, acc_top, acc_box, acc_seq_code = self._calculate_accuracy(outputs, batch)

        # 損失ログ
        self.log(
            "val/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/loss_map_code",
            loss_map_code,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/loss_top", loss_top, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "val/loss_box", loss_box, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "val/loss_seq_code",
            loss_seq_code,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # 精度ログ
        self.log(
            "val/acc_map_code",
            acc_map_code,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/acc_top", acc_top, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "val/acc_box", acc_box, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "val/acc_seq_code",
            acc_seq_code,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["image"], batch["label_ids"])
        total_loss, loss_map_code, loss_top, loss_box, loss_seq_code = self._calculate_loss(outputs, batch)

        # 精度計算
        acc_map_code, acc_top, acc_box, acc_seq_code = self._calculate_accuracy(outputs, batch)

        # 損失ログ
        self.log("test/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/loss_map_code", loss_map_code, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/loss_top", loss_top, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/loss_box", loss_box, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/loss_seq_code", loss_seq_code, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # 精度ログ
        self.log("test/acc_map_code", acc_map_code, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/acc_top", acc_top, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/acc_box", acc_box, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/acc_seq_code", acc_seq_code, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = RAdamScheduleFree(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "train"):
            optimizer.train()

    def on_validation_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_test_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_predict_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()


# --- 5. DataModule ---
class KuzushijiDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, tokenizer_path: str, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = None

    def prepare_data(self):
        # Load tokenizer. This is called once.
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Ensure special tokens exist for robustness.
        # Note: If adding tokens, ensure the model's embedding layer vocabulary size is updated
        # or handle it in the model's __init__ after tokenizer is loaded.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"bos_token": "[BOS]"})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})

        # After adding tokens, the vocab_size might change, so update it.
        # This tokenizer will be passed to the model.

    def setup(self, stage=None):
        if self.tokenizer is None:
            # This handles cases where `trainer.fit` might call setup before `prepare_data`
            self.prepare_data()

        if stage == "fit" or stage is None:
            self.train_dataset = KuzushijiDataset(self.data_dir, "train", self.tokenizer)
            self.val_dataset = KuzushijiDataset(self.data_dir, "val", self.tokenizer)
        if stage == "test" or stage is None:
            self.test_dataset = KuzushijiDataset(self.data_dir, "test", self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

# Main execution block
def main():
    # Example usage:
    # Set data root and tokenizer path
    # Replace with actual paths if running with real data
    DATA_ROOT = "data/column_dataset_padded"
    TOKENIZER_PATH = "experiments/kuzushiji_tokenizer_one_char"

    # Initialize data module
    dm = KuzushijiDataModule(data_dir=DATA_ROOT, tokenizer_path=TOKENIZER_PATH, batch_size=16, num_workers=8)
    dm.prepare_data()  # Load tokenizer
    dm.setup()  # Set up datasets

    # Initialize model
    model = OCRModel(tokenizer=dm.tokenizer, learning_rate=1e-5)

    # logger
    logger = WandbLogger(project="kuzushiji-ocr-f", name="test")

    # Add callbacks
    lr_finder = FineTuneLearningRateFinder(milestones=[5, 10])
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor="val/acc_seq_code", mode="max", save_top_k=3, save_last=True)
    ema_callback = EMACallback(decay=0.9999)
    early_stopping_callback = EarlyStopping(monitor="val/acc_seq_code", mode="max", patience=10)

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="auto",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
        log_every_n_steps=1,
        logger=logger,
        callbacks=[
            lr_finder,
            lr_monitor_callback,
            checkpoint_callback,
            ema_callback,
            early_stopping_callback
        ]
    )

    # To run actual training/validation/testing, uncomment the lines below.
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
