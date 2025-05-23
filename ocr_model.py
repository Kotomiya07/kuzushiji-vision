import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
import pytorch_lightning as pl
import editdistance # For CER calculation

# Assume char_to_int will have 0 as blank, matching CTCLoss default
# Assume int_to_char will be passed for decoding

class OCRModel(pl.LightningModule):
    def __init__(self, num_chars, int_to_char, max_label_length,
                 input_height=64, input_width=512, input_channels=3, # Added input_channels
                 rnn_hidden_size=256, rnn_layers=2,
                 encoder_name="resnet34", pretrained_encoder=True,
                 learning_rate=1e-3, lambda_bbox=1.0):
        super().__init__()
        # int_to_char is not a simple hparam, ignore for save_hyperparameters
        self.save_hyperparameters(ignore=['int_to_char']) 

        self.num_chars = num_chars
        self.int_to_char = int_to_char
        # self.max_label_length is now in hparams
        self.input_height = input_height # hparams.input_height
        self.input_width = input_width   # hparams.input_width
        self.input_channels = input_channels # hparams.input_channels
        self.rnn_hidden_size = rnn_hidden_size # hparams.rnn_hidden_size
        self.rnn_layers = rnn_layers         # hparams.rnn_layers
        self.encoder_name = encoder_name     # hparams.encoder_name
        self.pretrained_encoder = pretrained_encoder # hparams.pretrained_encoder
        # self.learning_rate is in hparams
        self.lambda_bbox = lambda_bbox       # hparams.lambda_bbox


        # --- Encoder (ResNet-like) ---
        if self.hparams.encoder_name == "resnet34":
            backbone = models.resnet34(pretrained=self.hparams.pretrained_encoder)
            self.encoder_output_channels = 512 # ResNet34 last block channels
        elif self.hparams.encoder_name == "resnet50":
            backbone = models.resnet50(pretrained=self.hparams.pretrained_encoder)
            self.encoder_output_channels = 2048 # ResNet50 last block channels
        else:
            raise ValueError(f"Unsupported encoder: {self.hparams.encoder_name}")

        # Modify ResNet's first conv layer if input_channels is not 3
        if self.hparams.input_channels != 3:
            original_conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                self.hparams.input_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
        
        # Remove avgpool and fc
        self.encoder = nn.Sequential(*(list(backbone.children())[:-2]))
        
        # --- Adaptive Pooling ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) 

        # --- Decoder (Bidirectional LSTM) ---
        self.decoder = nn.LSTM(
            input_size=self.encoder_output_channels,
            hidden_size=self.hparams.rnn_hidden_size,
            num_layers=self.hparams.rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        # --- Output Layers ---
        # Character prediction head
        self.char_fc = nn.Linear(self.hparams.rnn_hidden_size * 2, self.num_chars)

        # Bounding box prediction head (per time step)
        self.bbox_fc = nn.Linear(self.hparams.rnn_hidden_size * 2, 4)
        # If fixed max_label_length output is strictly needed for bbox:
        # self.bbox_fc_alt = nn.Linear(self.hparams.rnn_hidden_size * 2, self.hparams.max_label_length * 4)

        # --- Loss Functions ---
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='none') # Using SmoothL1Loss as GIoU can be unstable
        # self.bbox_loss_fn = ops.generalized_box_iou_loss # Alternative

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        features = self.encoder(x)  # (batch_size, encoder_output_channels, H', W')
        pooled_features = self.adaptive_pool(features) # (batch_size, channels, 1, W_seq)
        
        b, c, _, w_seq = pooled_features.size()
        rnn_input = pooled_features.squeeze(2).permute(0, 2, 1) # (batch_size, W_seq, channels)
        
        rnn_output, _ = self.decoder(rnn_input) # (batch_size, W_seq, rnn_hidden_size * 2)
        
        char_logits = self.char_fc(rnn_output) # (batch_size, W_seq, num_chars)
        bbox_preds_seq = self.bbox_fc(rnn_output) # (batch_size, W_seq, 4)

        # If using fixed max_label_length output for bbox:
        # aggregated_rnn_output = rnn_output.mean(dim=1) # Or other aggregation
        # bbox_preds_flat = self.bbox_fc_alt(aggregated_rnn_output)
        # bbox_preds = bbox_preds_flat.view(b, self.hparams.max_label_length, 4)
        # For now, returning sequence, alignment handled in loss.
        
        return {"char_logits": char_logits, "bbox_preds": bbox_preds_seq}

    def _common_step(self, batch, batch_idx, stage):
        images, labels, bboxes_gt, label_lengths = batch["image"], batch["label"], batch["bounding_boxes"], batch["label_lengths"]
        # label_lengths is crucial and must be provided by the DataLoader.

        outputs = self(images)
        char_logits = outputs["char_logits"] # (N, T_model, C)
        bbox_preds = outputs["bbox_preds"]   # (N, T_model, 4)

        batch_size = images.size(0)
        
        # --- CTC Loss ---
        log_probs = F.log_softmax(char_logits, dim=2).permute(1, 0, 2) # (T_model, N, C)
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long, device=self.device)
        ctc_loss = self.ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

        # --- Bounding Box Loss ---
        # Align bbox_preds (len T_model) with bboxes_gt (len S_max = self.hparams.max_label_length)
        # Use label_lengths to compute loss only for actual characters.
        bbox_loss_sum = torch.tensor(0.0, device=self.device)
        total_valid_boxes = 0
        
        # Number of box predictions to consider from model output, cannot exceed T_model
        # Number of GT boxes to consider, cannot exceed S_max
        # We compare up to the true length of the label.
        
        # T_model (model's output sequence length for bboxes)
        t_model = bbox_preds.size(1)

        for i in range(batch_size):
            current_label_length = label_lengths[i].item()
            if current_label_length == 0:
                continue

            # Number of boxes to compare for this sample:
            # min of actual label length, model's output seq length, and GT's max length
            len_to_compare = min(current_label_length, t_model, self.hparams.max_label_length)

            if len_to_compare == 0:
                continue
                
            pred_boxes_sample = bbox_preds[i, :len_to_compare]    # (len_to_compare, 4)
            gt_boxes_sample = bboxes_gt[i, :len_to_compare]        # (len_to_compare, 4)
            
            # Mask for valid ground truth boxes (e.g. not all [0,0,0,0] padding)
            # This check might be useful if GT padding isn't just zeros.
            # valid_gt_mask = gt_boxes_sample.abs().sum(dim=1) > 0 
            # For now, assume length-based selection is sufficient.

            # L1 Smooth Loss expects (N,C) or (N,4)
            loss_sample_bbox = self.bbox_loss_fn(pred_boxes_sample, gt_boxes_sample) # (len_to_compare, 4)
            bbox_loss_sum += loss_sample_bbox.sum() # Sum over coordinates and then boxes for this sample
            total_valid_boxes += len_to_compare
        
        avg_bbox_loss = bbox_loss_sum / total_valid_boxes if total_valid_boxes > 0 else torch.tensor(0.0, device=self.device)
        
        total_loss = ctc_loss + self.hparams.lambda_bbox * avg_bbox_loss

        self.log(f'{stage}_loss', total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f'{stage}_ctc_loss', ctc_loss, on_step=(stage=='train'), on_epoch=True, logger=True, batch_size=batch_size)
        self.log(f'{stage}_bbox_loss', avg_bbox_loss, on_step=(stage=='train'), on_epoch=True, logger=True, batch_size=batch_size)
        
        # --- Metrics (for val/test) ---
        if stage != 'train':
            # CER
            decoded_preds_int = self._greedy_decode(char_logits)
            preds_str_list = self._convert_to_strings(decoded_preds_int)
            # For labels, ensure using self.int_to_char and label_lengths for correct conversion
            targets_str_list = self._convert_to_strings_gt(labels.cpu().numpy(), label_lengths.cpu().numpy())
            
            cer = self.calculate_cer(preds_str_list, targets_str_list)
            self.log(f'{stage}_cer', cer, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

            # Mean IoU
            # num_preds_to_consider_for_iou is effectively t_model here
            mean_iou = self.calculate_mean_iou(bbox_preds, bboxes_gt, label_lengths, t_model, self.hparams.max_label_length)
            self.log(f'{stage}_mean_iou', mean_iou, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _greedy_decode(self, char_logits):
        # char_logits: (N, T_model, C)
        pred_indices = torch.argmax(char_logits, dim=2) # (N, T_model)
        decoded_sequences = []
        for i in range(pred_indices.size(0)):
            seq = pred_indices[i].cpu().numpy()
            decoded_seq = []
            last_char = -1 # Stores the last non-blank character index
            for char_idx in seq:
                if char_idx == 0: # CTC Blank token
                    last_char = -1 # Reset last_char on blank
                    continue
                if char_idx == last_char: # Repeated token
                    continue
                decoded_seq.append(char_idx)
                last_char = char_idx
            decoded_sequences.append(decoded_seq)
        return decoded_sequences

    def _convert_to_strings(self, sequences_of_int):
        # For predicted sequences (list of lists of ints)
        strings = []
        for seq_int in sequences_of_int:
            strings.append("".join([self.int_to_char.get(token, "") for token in seq_int]))
        return strings
        
    def _convert_to_strings_gt(self, labels_tensor_padded, label_lengths_tensor):
        # For ground truth labels (numpy array of padded labels and their true lengths)
        strings = []
        for i, label_vec in enumerate(labels_tensor_padded):
            true_len = label_lengths_tensor[i]
            strings.append("".join([self.int_to_char.get(token, "") for token in label_vec[:true_len]]))
        return strings

    @staticmethod
    def calculate_cer(preds_str_list, targets_str_list):
        if len(preds_str_list) != len(targets_str_list): return 1.0
        total_edit_distance = 0
        total_target_length = 0
        for pred, target in zip(preds_str_list, targets_str_list):
            total_edit_distance += editdistance.eval(pred, target)
            total_target_length += len(target)
        return total_edit_distance / total_target_length if total_target_length > 0 else (0.0 if total_edit_distance == 0 else 1.0)

    @staticmethod
    def calculate_mean_iou(bbox_preds_seq, bbox_targets_padded, target_lengths, t_model, s_max):
        # bbox_preds_seq: (N, T_model, 4) - model output sequence
        # bbox_targets_padded: (N, S_max, 4) - ground truth, padded to S_max
        # target_lengths: (N) - actual length of each sequence in batch
        # t_model: sequence length of predictions
        # s_max: max padded length of ground truth
        
        all_ious = []
        batch_size = bbox_preds_seq.size(0)

        for i in range(batch_size):
            actual_len = target_lengths[i].item()
            if actual_len == 0: continue

            # Number of boxes to compare for this sample
            len_to_compare = min(actual_len, t_model, s_max)
            if len_to_compare == 0: continue

            preds = bbox_preds_seq[i, :len_to_compare]      # (len_to_compare, 4)
            targets = bbox_targets_padded[i, :len_to_compare]  # (len_to_compare, 4)
            
            try:
                # torchvision.ops.box_iou expects (x1, y1, x2, y2)
                # It returns a matrix of (num_preds_boxes, num_targets_boxes)
                iou_matrix = ops.box_iou(preds, targets) # (len_to_compare, len_to_compare)
                # We want diagonal IoUs for matched pairs
                ious = torch.diag(iou_matrix) # (len_to_compare)
                
                # Filter out NaNs or Infs if any (e.g. from zero-area boxes if not careful)
                valid_ious = ious[~torch.isnan(ious) & ~torch.isinf(ious)]
                all_ious.extend(valid_ious.tolist())
            except Exception:
                # print(f"Warning: box_iou calculation error: {e}")
                pass # Skip if error, e.g. invalid boxes

        return sum(all_ious) / len(all_ious) if all_ious else 0.0

if __name__ == '__main__':
    # --- Dummy int_to_char and other params for local testing ---
    dummy_int_to_char = {0: "<PAD>", 1: "<UNK>", 2: "a", 3: "b", 4: "c"} 
    dummy_num_chars = len(dummy_int_to_char)
    DUMMY_MAX_LABEL_LENGTH = 10 # S_max for GT padding
    
    model = OCRModel(
        num_chars=dummy_num_chars,
        int_to_char=dummy_int_to_char,
        max_label_length=DUMMY_MAX_LABEL_LENGTH, # This is S_max
        input_height=64,
        input_width=256, 
        input_channels=3,
        rnn_hidden_size=128,
        encoder_name="resnet34",
        pretrained_encoder=False,
        learning_rate=1e-4,
        lambda_bbox=0.5
    )
    
    batch_size = 4
    img_height = model.hparams.input_height
    img_width = model.hparams.input_width
    channels = model.hparams.input_channels
    
    dummy_images = torch.randn(batch_size, channels, img_height, img_width)
    
    # Labels are integer encoded, padded to DUMMY_MAX_LABEL_LENGTH (S_max)
    dummy_labels_list = [
        [2,3,4,0,0,0,0,0,0,0], # abc
        [3,4,2,3,0,0,0,0,0,0], # bcab
        [4,2,0,0,0,0,0,0,0,0], # ca
        [2,3,2,3,2,0,0,0,0,0]  # ababa
    ]
    dummy_labels = torch.tensor(dummy_labels_list, dtype=torch.long)
    dummy_label_lengths = torch.tensor([3, 4, 2, 5], dtype=torch.long) # Actual lengths

    # BBoxes GT are padded to DUMMY_MAX_LABEL_LENGTH (S_max)
    dummy_bboxes_gt_list = []
    for i in range(batch_size):
        length = dummy_label_lengths[i].item()
        boxes = []
        for j in range(DUMMY_MAX_LABEL_LENGTH):
            if j < length:
                boxes.append([j*10, 10, j*10+9, 29]) # x1,y1,x2,y2 valid box
            else:
                boxes.append([0,0,0,0]) # padding box
        dummy_bboxes_gt_list.append(boxes)
    dummy_bboxes_gt = torch.tensor(dummy_bboxes_gt_list, dtype=torch.float32)

    dummy_batch = {
        "image": dummy_images,
        "label": dummy_labels, # Padded to S_max
        "bounding_boxes": dummy_bboxes_gt, # Padded to S_max
        "label_lengths": dummy_label_lengths # True lengths
    }

    print(f"Model Max Label Length (S_max for GT): {model.hparams.max_label_length}")
    print(f"Dummy Labels Shape: {dummy_labels.shape}") # (N, S_max)
    print(f"Dummy BBoxes GT Shape: {dummy_bboxes_gt.shape}") # (N, S_max, 4)
    print(f"Dummy Label Lengths: {dummy_label_lengths}")

    print("\n--- Testing forward pass ---")
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)
    char_logits = outputs["char_logits"] # (N, T_model, C)
    bbox_preds = outputs["bbox_preds"]   # (N, T_model, 4)
    # T_model is model's output sequence length, depends on input_width & encoder downsampling.
    print(f"char_logits shape: {char_logits.shape} (N, T_model, C)")
    print(f"bbox_preds shape: {bbox_preds.shape} (N, T_model, 4)")

    print("\n--- Testing training_step (via _common_step) ---")
    model.train()
    loss = model.training_step(dummy_batch, 0)
    print("Training loss:", loss.item())

    print("\n--- Testing validation_step (via _common_step) ---")
    model.eval()
    val_metrics = model.validation_step(dummy_batch, 0) # This returns loss
    print("Validation loss:", val_metrics.item())
    # Logged metrics (val_cer, val_mean_iou) can be checked via a logger or by inspecting model.trainer.logged_metrics
    
    # Manually test CER and IoU calculations with example data
    print("\n--- Testing CER calculation logic ---")
    # Example: preds from greedy decode (list of lists of ints)
    ex_preds_int = model._greedy_decode(char_logits.detach()) # Use actual model output
    preds_str = model._convert_to_strings(ex_preds_int)
    targets_str = model._convert_to_strings_gt(dummy_labels.cpu().numpy(), dummy_label_lengths.cpu().numpy())
    print("Sample Pred strings:", preds_str[:2])
    print("Sample Target strings:", targets_str[:2])
    cer = OCRModel.calculate_cer(preds_str, targets_str)
    print("Calculated CER on batch:", cer)

    print("\n--- Testing IoU calculation logic ---")
    t_model_val = bbox_preds.size(1)
    s_max_val = dummy_bboxes_gt.size(1)
    mean_iou = OCRModel.calculate_mean_iou(bbox_preds.detach(), dummy_bboxes_gt, dummy_label_lengths, t_model_val, s_max_val)
    print("Calculated Mean IoU on batch:", mean_iou)
    
    print("\nocr_model.py local test finished.")
