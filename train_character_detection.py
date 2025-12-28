# train_ocr.py
import ast  # For safely evaluating string representations of lists/dicts
import datetime
import os
import random
from collections.abc import Callable
from functools import partial

import jiwer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoConfig, AutoModelForCausalLM

import wandb


# --- ViTEncoder Code (Copied from enhancing-transformers/enhancing/modules/stage1/layers.py) ---
import math
import numpy as np
from typing import Union, Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) # Corrected: np.float to np.float32
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \\
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \\
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)

        return x

# --- End of ViTEncoder Code ---


# --- Constants and Configs ---
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# --- Vocabulary and Tokenizer ---
class Vocab:
    def __init__(self, unicode_csv_path: str, special_tokens: list[str]):
        self.special_tokens = special_tokens
        df = pd.read_csv(unicode_csv_path)

        self.char_to_id = {}
        self.id_to_char = {}

        for token in special_tokens:
            self._add_char(token)

        for _, row in df.iterrows():
            char = row["char"]
            if isinstance(char, str):
                self._add_char(char)

        self.pad_id = self.char_to_id[PAD_TOKEN]
        self.sos_id = self.char_to_id[SOS_TOKEN]
        self.eos_id = self.char_to_id[EOS_TOKEN]
        self.unk_id = self.char_to_id[UNK_TOKEN]

    def _add_char(self, char):
        if char not in self.char_to_id:
            idx = len(self.char_to_id)
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        ids = [self.char_to_id.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], remove_special_tokens=True) -> str:
        chars = []
        for token_id in ids:
            if remove_special_tokens and token_id in [self.sos_id, self.eos_id, self.pad_id]:
                if token_id == self.eos_id:
                    break
                continue
            chars.append(self.id_to_char.get(token_id, UNK_TOKEN))
        return "".join(chars)

    def __len__(self):
        return len(self.char_to_id)


# --- Dataset Helper ---
def parse_unicode_ids_to_text(unicode_ids_str: str) -> str:
    try:
        unicode_list = ast.literal_eval(unicode_ids_str)
        text = ""
        for u_id in unicode_list:
            if isinstance(u_id, str) and u_id.startswith("U+"):
                try:
                    char_code = int(u_id[2:], 16)
                    text += chr(char_code)
                except ValueError:
                    pass
        return text
    except (ValueError, SyntaxError):
        return ""


# --- Dataset ---
class KuzushijiDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Vocab,
        transform: Callable | None = None,
        max_label_len: int = 256,
        image_channels: int = 1,
        target_image_height: int = 1024,
        target_image_width: int = 64,
        csv_filename_template: str = "{split_name}_column_info.csv",
        image_col: str = "column_image",
        label_col: str = "unicode_ids",
        bbox_col: str = "char_boxes_in_column",
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_label_len = max_label_len
        self.image_channels = image_channels
        self.target_image_height = target_image_height
        self.target_image_width = target_image_width

        self.image_paths = []
        self.labels = []
        self.char_boxes_list = []

        split_name = os.path.basename(data_dir)
        csv_filename = csv_filename_template.format(split_name=split_name)
        csv_path = os.path.join(data_dir, csv_filename)
        
        project_root_for_images = os.path.dirname(self.data_dir)

        if not os.path.exists(csv_path):
            print(f"Warning: Annotation CSV file not found at {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}")
            return

        if not all(col in df.columns for col in [image_col, label_col, bbox_col]):
            print(f"Error: CSV file {csv_path} must contain '{image_col}', '{label_col}', and '{bbox_col}' columns.")
            return

        for _, row in df.iterrows():
            image_filename_relative = row[image_col]
            unicode_ids_str = str(row[label_col])
            char_boxes_val = row[bbox_col]

            if pd.isna(image_filename_relative) or pd.isna(unicode_ids_str):
                continue

            full_image_path = os.path.join(project_root_for_images, image_filename_relative)
            if not os.path.exists(full_image_path):
                continue

            label_str = parse_unicode_ids_to_text(unicode_ids_str)
            if not label_str:
                continue
            
            raw_char_boxes = []
            if not pd.isna(char_boxes_val):
                try:
                    parsed_boxes = ast.literal_eval(str(char_boxes_val))
                    if isinstance(parsed_boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in parsed_boxes):
                        raw_char_boxes = parsed_boxes
                except (ValueError, SyntaxError, TypeError):
                    pass

            num_chars_in_label = len(label_str)
            aligned_char_boxes = raw_char_boxes[:num_chars_in_label]

            self.image_paths.append(full_image_path)
            self.labels.append(label_str)
            self.char_boxes_list.append(aligned_char_boxes)


        if not self.image_paths:
            print(f"Warning: No valid image-label-bbox triplets found from {csv_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]
        char_boxes_for_item = self.char_boxes_list[idx]

        try:
            img = Image.open(img_path)
            original_width, original_height = img.size
            if self.image_channels == 1:
                image = img.convert("L")
            else:
                image = img.convert("RGB")
        except Exception as e:
            print(f"Error opening or processing image {img_path}: {e}")
            dummy_image = torch.zeros((self.image_channels, self.target_image_height, self.target_image_width))
            dummy_label_ids = torch.full((self.max_label_len,), self.tokenizer.pad_id, dtype=torch.long)
            dummy_label_ids[0] = self.tokenizer.sos_id
            dummy_label_ids[1] = self.tokenizer.eos_id
            dummy_bboxes = torch.zeros((self.max_label_len - 1, 4), dtype=torch.float)
            return dummy_image, dummy_label_ids, dummy_bboxes

        if self.transform:
            image = self.transform(image)

        label_ids_no_special = self.tokenizer.encode(label_str, add_special_tokens=False)
        label_ids_no_special = label_ids_no_special[:self.max_label_len - 2]
        
        final_label_ids = [self.tokenizer.sos_id] + label_ids_no_special + [self.tokenizer.eos_id]
        padding_len = self.max_label_len - len(final_label_ids)
        final_label_ids_padded = final_label_ids + [self.tokenizer.pad_id] * padding_len
        label_tensor = torch.tensor(final_label_ids_padded, dtype=torch.long)
        
        scaled_char_boxes = []
        scale_w = self.target_image_width / original_width if original_width > 0 else 1
        scale_h = self.target_image_height / original_height if original_height > 0 else 1

        num_actual_chars = len(label_ids_no_special)
        for i in range(num_actual_chars):
            if i < len(char_boxes_for_item):
                box = char_boxes_for_item[i]
                scaled_box = [
                    box[0] * scale_w, box[1] * scale_h,
                    box[2] * scale_w, box[3] * scale_h
                ]
                scaled_char_boxes.append(scaled_box)
            else:
                scaled_char_boxes.append([0.0, 0.0, 0.0, 0.0]) 

        target_bboxes_tensor = torch.zeros((self.max_label_len - 1, 4), dtype=torch.float)
        
        if scaled_char_boxes:
            target_bboxes_tensor[:num_actual_chars] = torch.tensor(scaled_char_boxes, dtype=torch.float)
            
        return image, label_tensor, target_bboxes_tensor


# --- Collate Function ---
def collate_fn(batch, pad_token_id):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    bboxes = [item[2] for item in batch] 

    try:
        images_stacked = torch.stack(images, dim=0)
    except RuntimeError as e:
        print(f"Error stacking images in collate_fn: {e}. Check if all images have the same size after transforms.")
        raise

    padded_labels = torch_pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    try:
        padded_bboxes = torch.stack(bboxes, dim=0)
    except RuntimeError as e:
        print(f"Error stacking bboxes: {e}. Check bbox tensor shapes.")
        raise

    return images_stacked, padded_labels, padded_bboxes


# --- OCR Model ---
class OCRModel(nn.Module):
    def __init__(
        self,
        vit_encoder_config: dict, # Changed: navit_config to vit_encoder_config
        deberta_model_name: str, # This will be 'KoichiYasuoka/roberta-small-japanese-aozora-char'
        vocab_size: int,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        max_decoder_len: int,
        image_channels: int = 1, # Added: image_channels for ViTEncoder
    ):
        super().__init__()
        self.vit_encoder_config = vit_encoder_config
        self.deberta_model_name = deberta_model_name
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_decoder_len = max_decoder_len

        # Encoder (ViTEncoder)
        # Assumes vit_encoder_config contains: image_size, patch_size, dim, depth, heads, mlp_dim
        self.encoder = ViTEncoder(
            image_size=vit_encoder_config["image_size"], 
            patch_size=vit_encoder_config["patch_size"], 
            dim=vit_encoder_config["dim"], 
            depth=vit_encoder_config["depth"], 
            heads=vit_encoder_config["heads"], 
            mlp_dim=vit_encoder_config["mlp_dim"],
            channels=image_channels, # Use image_channels passed to OCRModel
            # dim_head can be defaulted by ViTEncoder or added to config if needed
        )

        # Load config first
        decoder_config = AutoConfig.from_pretrained(deberta_model_name)
        # Modify config for use as a decoder in an encoder-decoder setup
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.output_hidden_states = True # Keep this for bbox prediction
        # deberta_hidden_size is already part of decoder_config.hidden_size
        deberta_hidden_size = decoder_config.hidden_size

        # Load the model with the modified config
        self.decoder_hf = AutoModelForCausalLM.from_pretrained(
            deberta_model_name,
            config=decoder_config, # Pass the modified config object
            # is_decoder=True, # This was causing the error, now handled in config
            # add_cross_attention=True, # Also handled in config
            ignore_mismatched_sizes=True # Keep this if resizing embeddings
        )
        self.decoder_hf.resize_token_embeddings(vocab_size)

        # Configure decoder for generation based on the loaded config
        self.decoder_hf.config.decoder_start_token_id = sos_id
        self.decoder_hf.config.bos_token_id = sos_id
        self.decoder_hf.config.eos_token_id = eos_id
        self.decoder_hf.config.pad_token_id = pad_id
        self.decoder_hf.config.max_length = self.max_decoder_len
        # vocab_size might already be updated by resize_token_embeddings,
        # but setting it explicitly in config doesn't hurt
        self.decoder_hf.config.vocab_size = vocab_size


        if self.vit_encoder_config["dim"] != deberta_hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.vit_encoder_config["dim"], deberta_hidden_size)
        else:
            self.enc_to_dec_proj = nn.Identity()

        self.bbox_predictor = nn.Linear(deberta_hidden_size, 4)


    def encode(self, images_tensor: Tensor) -> tuple[Tensor, Tensor]:
        # Pass images through ViT encoder
        encoder_features = self.encoder(images_tensor)
        
        # Create a dummy attention mask (all ones, as ViTEncoder output is not padded in the same way as NaViT's variable input)
        # The shape should be (batch_size, num_patches)
        batch_size, num_patches, _ = encoder_features.shape
        attention_mask = torch.ones((batch_size, num_patches), dtype=torch.long, device=encoder_features.device)
        
        return encoder_features, attention_mask

    def forward(self, images: Tensor, target_ids: Tensor | None = None, target_bboxes: Tensor | None = None):
        encoder_hidden_states, encoder_attention_mask = self.encode(images)
        encoder_hidden_states_proj = self.enc_to_dec_proj(encoder_hidden_states)

        if target_ids is not None:
            decoder_input_ids = target_ids[:, :-1]
            labels_for_char_loss = target_ids[:, 1:]
            decoder_attention_mask = (decoder_input_ids != self.pad_id)

            outputs = self.decoder_hf(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states_proj,
                encoder_attention_mask=encoder_attention_mask,
                # output_hidden_states is set in config, so it should produce outputs.hidden_states
            )
            char_logits = outputs.logits
            # Ensure hidden_states are being output
            if outputs.hidden_states is None:
                raise ValueError("Decoder did not return hidden_states. Check config `output_hidden_states=True`.")
            decoder_last_hidden_state = outputs.hidden_states[-1]
            predicted_bboxes = self.bbox_predictor(decoder_last_hidden_state)

            return char_logits, labels_for_char_loss, predicted_bboxes, target_bboxes

        else:
            # For generation, the decoder_hf.generate method should use the config attributes
            # (is_decoder, add_cross_attention) correctly.
            generated_ids = self.decoder_hf.generate(
                encoder_hidden_states=encoder_hidden_states_proj,
                encoder_attention_mask=encoder_attention_mask,
                # max_length etc. are taken from self.decoder_hf.config
            )
            return generated_ids, None, None, None

# --- CER Calculation ---
def calculate_cer(predictions_ids: list[list[int]], targets_ids: list[list[int]], tokenizer: Vocab) -> float:
    pred_strs = [tokenizer.decode(p, remove_special_tokens=True) for p in predictions_ids]
    target_strs = [tokenizer.decode(t, remove_special_tokens=True) for t in targets_ids]

    filtered_preds = []
    filtered_targets = []
    for p, t in zip(pred_strs, target_strs, strict=False):
        if t:
            filtered_preds.append(p)
            filtered_targets.append(t)

    if not filtered_targets:
        return 1.0 if any(filtered_preds) else 0.0

    return jiwer.cer(filtered_targets, filtered_preds)


# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, criterion_char, criterion_bbox, bbox_loss_weight, device, tokenizer, epoch_num, clip_grad_norm_val: float | None = 1.0): # WANDB: Added epoch_num
    model.train()
    total_char_loss_epoch = 0
    total_bbox_loss_epoch = 0
    total_combined_loss_epoch = 0
    
    if len(dataloader) == 0:
        print("Warning: Training dataloader is empty.")
        return 0.0, 0.0, 0.0

    for i, (images, target_ids, target_bboxes) in enumerate(dataloader):
        images = images.to(device)
        target_ids = target_ids.to(device)
        target_bboxes = target_bboxes.to(device)

        optimizer.zero_grad()
        
        char_logits, labels_for_char_loss, pred_bboxes, target_bboxes_for_loss = model(images, target_ids, target_bboxes)
        
        loss_char = criterion_char(char_logits.reshape(-1, char_logits.size(-1)), labels_for_char_loss.reshape(-1))

        bbox_loss_mask = (labels_for_char_loss != tokenizer.pad_id).unsqueeze(-1).expand_as(pred_bboxes)
        loss_bbox_unreduced = criterion_bbox(pred_bboxes, target_bboxes_for_loss)
        loss_bbox_masked = torch.where(bbox_loss_mask, loss_bbox_unreduced, torch.zeros_like(loss_bbox_unreduced))
        num_valid_bboxes = bbox_loss_mask.sum().clamp(min=1)
        loss_bbox = loss_bbox_masked.sum() / num_valid_bboxes 
        
        total_combined_loss = loss_char + bbox_loss_weight * loss_bbox
        
        total_combined_loss.backward()
        if clip_grad_norm_val:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_val)
        optimizer.step()

        total_char_loss_epoch += loss_char.item()
        total_bbox_loss_epoch += loss_bbox.item()
        total_combined_loss_epoch += total_combined_loss.item()

        # WANDB: Log batch metrics
        if (i + 1) % 50 == 0: # Log every 50 batches, or adjust as needed
            wandb.log({
                "train_batch_char_loss": loss_char.item(),
                "train_batch_bbox_loss": loss_bbox.item(),
                "train_batch_combined_loss": total_combined_loss.item(),
                "train_step": epoch_num * len(dataloader) + i # Global step
            })
            print(f"Epoch {epoch_num}, Batch {i+1}/{len(dataloader)}, Char Loss: {loss_char.item():.4f}, BBox Loss: {loss_bbox.item():.4f}, Combined: {total_combined_loss.item():.4f}")


    avg_char_loss = total_char_loss_epoch / len(dataloader)
    avg_bbox_loss = total_bbox_loss_epoch / len(dataloader)
    avg_combined_loss = total_combined_loss_epoch / len(dataloader)
    return avg_char_loss, avg_bbox_loss, avg_combined_loss


# --- Training and Validation Functions ---

def validate(model, dataloader, criterion_char, criterion_bbox, bbox_loss_weight, device, tokenizer: Vocab, epoch_num): # WANDB: Added epoch_num
    model.eval()
    total_char_loss_epoch = 0
    total_bbox_loss_epoch = 0
    all_predictions_ids = []
    all_targets_ids = []
    first_batch_printed = False

    example_images = []
    example_preds_text = []
    example_targets_text = []


    if len(dataloader) == 0:
        print("Warning: Validation dataloader is empty.")
        return 0.0, 0.0, 0.0, 1.0

    with torch.no_grad():
        for i, (images, target_ids, target_bboxes) in enumerate(dataloader):
            images = images.to(device)
            target_ids = target_ids.to(device)
            target_bboxes = target_bboxes.to(device)

            char_logits_val, labels_for_char_loss_val, pred_bboxes_val, target_bboxes_for_loss_val = model(images, target_ids, target_bboxes)
            
            if char_logits_val is not None and labels_for_char_loss_val is not None:
                loss_char_val = criterion_char(char_logits_val.reshape(-1, char_logits_val.size(-1)), labels_for_char_loss_val.reshape(-1))
                total_char_loss_epoch += loss_char_val.item()

                bbox_loss_mask_val = (labels_for_char_loss_val != tokenizer.pad_id).unsqueeze(-1).expand_as(pred_bboxes_val)
                loss_bbox_unreduced_val = criterion_bbox(pred_bboxes_val, target_bboxes_for_loss_val)
                loss_bbox_masked_val = torch.where(bbox_loss_mask_val, loss_bbox_unreduced_val, torch.zeros_like(loss_bbox_unreduced_val))
                num_valid_bboxes_val = bbox_loss_mask_val.sum().clamp(min=1)
                loss_bbox_val = loss_bbox_masked_val.sum() / num_valid_bboxes_val
                total_bbox_loss_epoch += loss_bbox_val.item()

            # Get generated IDs for CER calculation (inference path)
            # model's inference path returns: generated_output, None, None, None
            # The first element of the tuple from model forward (inference path) is the output of generate()
            generated_output, _, _, _ = model(images, target_ids=None, target_bboxes=None)

            # Extract the sequences tensor from the output object
            # The actual attribute name might vary based on the model type and transformers version.
            # Common attributes are 'sequences' or it might be the object itself if return_dict_in_generate=False (not default usually)
            if hasattr(generated_output, 'sequences'):
                generated_sequences = generated_output.sequences
            else:
                # If it's already a tensor (older versions or specific generate configs)
                generated_sequences = generated_output

            all_predictions_ids.extend(generated_sequences.cpu().tolist())
            all_targets_ids.extend(target_ids.cpu().tolist()) # Ground truth target_ids

            if not first_batch_printed and i == 0 and len(generated_sequences) > 0 and len(target_ids) > 0:
                pred_text_sample = tokenizer.decode(generated_sequences[0].cpu().tolist(), remove_special_tokens=True)
                target_text_sample = tokenizer.decode(target_ids[0].cpu().tolist(), remove_special_tokens=True)
                print("  Validation Example (Batch 1, Sample 1):")
                print(f"    Predicted: '{pred_text_sample}'")
                print(f"    Target:    '{target_text_sample}'")
                first_batch_printed = True
                
                if epoch_num % 5 == 0 or epoch_num == EPOCHS -1 :
                    num_examples_to_log = min(5, images.size(0))
                    for k in range(num_examples_to_log):
                        example_images.append(wandb.Image(images[k]))
                        example_preds_text.append(tokenizer.decode(generated_sequences[k].cpu().tolist(), remove_special_tokens=True))
                        example_targets_text.append(tokenizer.decode(target_ids[k].cpu().tolist(), remove_special_tokens=True))

    avg_char_loss = total_char_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
    avg_bbox_loss = total_bbox_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
    avg_combined_loss = avg_char_loss + bbox_loss_weight * avg_bbox_loss
    
    cer = calculate_cer(all_predictions_ids, all_targets_ids, tokenizer)

    if example_images:
        try:
            table = wandb.Table(columns=["Epoch", "Image", "Predicted Text", "Target Text"])
            for img, pred, target in zip(example_images, example_preds_text, example_targets_text, strict=False):
                table.add_data(epoch_num, img, pred, target)
            wandb.log({"validation_examples": table}, step=epoch_num)
        except Exception as e:
            print(f"WANDB: Error logging validation examples table: {e}")

    return avg_char_loss, avg_bbox_loss, avg_combined_loss, cer


# --- Main Script ---
if __name__ == "__main__":
    # --- Configuration ---
    # WANDB: Define hyperparameters in a dictionary for wandb.config
    config_defaults = {
        "DATA_BASE_DIR": "data/column_dataset_padded",
        "UNICODE_CSV_PATH": "data/unicode_translation.csv",
        "IMAGE_HEIGHT": 1024,
        "IMAGE_WIDTH": 64,
        "IMAGE_CHANNELS": 3,
        "PATCH_SIZE": 16,
        "NAVIT_DIM": 256,
        "NAVIT_DEPTH": 6,
        "NAVIT_HEADS": 8,
        "NAVIT_MLP_DIM_RATIO": 4, # NAVIT_MLP_DIM = NAVIT_DIM * NAVIT_MLP_DIM_RATIO
        "NAVIT_EMB_DROPOUT": 0.1,
        "NAVIT_DROPOUT": 0.1,
        "DEBERTA_MODEL_NAME": "KoichiYasuoka/roberta-small-japanese-aozora-char",
        "MAX_LABEL_LEN": 128,
        "BATCH_SIZE": 128,
        "EPOCHS": 1000, # Will be overwritten by global EPOCHS
        "LEARNING_RATE": 5e-5,
        "CLIP_GRAD_NORM": 1.0,
        "NUM_WORKERS": 24,
        "BBOX_LOSS_WEIGHT": 0.5,
        "SEED": 42,
        "WANDB_PROJECT": "kuzushiji-ocr",
        "WANDB_ENTITY": None, # WANDB: Set your entity (username or team), or leave as None for default
    }
    
    # Global EPOCHS, will be used by validate function for example logging condition
    EPOCHS = config_defaults["EPOCHS"]


    # WANDB: Initialize wandb
    run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=config_defaults["WANDB_PROJECT"],
        entity=config_defaults["WANDB_ENTITY"],
        config=config_defaults,
        name=run_name,
        # mode="disabled" # WANDB: Uncomment to disable wandb for debugging
    )
    # WANDB: Access hyperparameters from wandb.config for consistency
    cfg = wandb.config

    DATA_BASE_DIR = cfg.DATA_BASE_DIR
    UNICODE_CSV_PATH = cfg.UNICODE_CSV_PATH
    IMAGE_HEIGHT = cfg.IMAGE_HEIGHT
    IMAGE_WIDTH = cfg.IMAGE_WIDTH
    IMAGE_CHANNELS = cfg.IMAGE_CHANNELS
    PATCH_SIZE = cfg.PATCH_SIZE
    NAVIT_DIM = cfg.NAVIT_DIM
    NAVIT_DEPTH = cfg.NAVIT_DEPTH
    NAVIT_HEADS = cfg.NAVIT_HEADS
    NAVIT_MLP_DIM = NAVIT_DIM * cfg.NAVIT_MLP_DIM_RATIO
    NAVIT_EMB_DROPOUT = cfg.NAVIT_EMB_DROPOUT
    NAVIT_DROPOUT = cfg.NAVIT_DROPOUT
    DEBERTA_MODEL_NAME = cfg.DEBERTA_MODEL_NAME
    MAX_LABEL_LEN = cfg.MAX_LABEL_LEN
    MAX_DECODER_LEN = MAX_LABEL_LEN
    BATCH_SIZE = cfg.BATCH_SIZE
    EPOCHS = cfg.EPOCHS # Overwrite global EPOCHS with wandb config
    LEARNING_RATE = cfg.LEARNING_RATE
    CLIP_GRAD_NORM = cfg.CLIP_GRAD_NORM
    NUM_WORKERS = cfg.NUM_WORKERS
    BBOX_LOSS_WEIGHT = cfg.BBOX_LOSS_WEIGHT
    SEED = cfg.SEED


    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Vocab(UNICODE_CSV_PATH, SPECIAL_TOKENS)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    train_transform = transforms.Compose(
        [
            #transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*IMAGE_CHANNELS, std=[0.5]*IMAGE_CHANNELS),
        ]
    )
    val_transform = transforms.Compose(
        [
            #transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*IMAGE_CHANNELS, std=[0.5]*IMAGE_CHANNELS),
        ]
    )

    train_dataset_dir = os.path.join(DATA_BASE_DIR, "train")
    val_dataset_dir = os.path.join(DATA_BASE_DIR, "val")

    train_dataset = KuzushijiDataset(
        train_dataset_dir, tokenizer, train_transform, MAX_LABEL_LEN, IMAGE_CHANNELS,
        target_image_height=IMAGE_HEIGHT, target_image_width=IMAGE_WIDTH,
        image_col="column_image", label_col="unicode_ids", bbox_col="char_boxes_in_column"
    )
    val_dataset = KuzushijiDataset(
        val_dataset_dir, tokenizer, val_transform, MAX_LABEL_LEN, IMAGE_CHANNELS,
        target_image_height=IMAGE_HEIGHT, target_image_width=IMAGE_WIDTH,
        image_col="column_image", label_col="unicode_ids", bbox_col="char_boxes_in_column"
    )

    if len(train_dataset) == 0:
        print(f"Error: Training dataset is empty. Check path: {train_dataset_dir} and its contents.")
        wandb.finish(exit_code=1) # WANDB: Finish run with error code
        exit()
    if len(val_dataset) == 0:
        print(f"Warning: Validation dataset is empty. Check path: {val_dataset_dir}.")

    custom_collate_fn = partial(collate_fn, pad_token_id=tokenizer.pad_id)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True if device.type == "cuda" else False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True if device.type == "cuda" else False,
    )

    navit_config_dict = {
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "patch_size": PATCH_SIZE,
        "num_classes": NAVIT_DIM, 
        "dim": NAVIT_DIM,
        "depth": NAVIT_DEPTH,
        "heads": NAVIT_HEADS,
        "mlp_dim": NAVIT_MLP_DIM,
        "channels": IMAGE_CHANNELS,
        "emb_dropout": NAVIT_EMB_DROPOUT,
        "dropout": NAVIT_DROPOUT,
    }

    model = OCRModel(
        vit_encoder_config=navit_config_dict,
        deberta_model_name=DEBERTA_MODEL_NAME,
        vocab_size=vocab_size,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_decoder_len=MAX_DECODER_LEN,
        image_channels=IMAGE_CHANNELS,
    ).to(device)

    # WANDB: Watch the model for gradients and topology (optional)
    #wandb.watch(model, log="all", log_freq=100) # Log gradients every 100 batches

    criterion_char = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    criterion_bbox = nn.L1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_cer = float("inf")
    # Output directory for local saves, wandb will also save artifacts
    output_dir_base = "experiments/ocr" # WANDB: Base directory for local saves
    output_dir = os.path.join(output_dir_base, run_name) # WANDB: Use run ID for unique folder
    
    print(f"Local output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")

        avg_char_loss_train, avg_bbox_loss_train, avg_combined_loss_train = train_one_epoch(
            model, train_dataloader, optimizer, criterion_char, criterion_bbox, 
            BBOX_LOSS_WEIGHT, device, tokenizer, epoch, CLIP_GRAD_NORM # WANDB: Pass current epoch
        )
        print(f"Epoch {epoch + 1} Training: Char Loss: {avg_char_loss_train:.4f}, BBox Loss: {avg_bbox_loss_train:.4f}, Combined Loss: {avg_combined_loss_train:.4f}")
        # WANDB: Log epoch training metrics
        wandb.log({
            "train_epoch_char_loss": avg_char_loss_train,
            "train_epoch_bbox_loss": avg_bbox_loss_train,
            "train_epoch_combined_loss": avg_combined_loss_train,
            "epoch": epoch
        })


        if len(val_dataloader) > 0:
            avg_char_loss_val, avg_bbox_loss_val, avg_combined_loss_val, val_cer = validate(
                model, val_dataloader, criterion_char, criterion_bbox, 
                BBOX_LOSS_WEIGHT, device, tokenizer, epoch # WANDB: Pass current epoch
            )
            print(f"Epoch {epoch + 1} Validation: Char Loss: {avg_char_loss_val:.4f}, BBox Loss: {avg_bbox_loss_val:.4f}, Combined Loss: {avg_combined_loss_val:.4f}, CER: {val_cer:.4f}")
            # WANDB: Log epoch validation metrics
            wandb.log({
                "val_epoch_char_loss": avg_char_loss_val,
                "val_epoch_bbox_loss": avg_bbox_loss_val,
                "val_epoch_combined_loss": avg_combined_loss_val,
                "val_epoch_cer": val_cer,
                "epoch": epoch
            })


            if val_cer < best_val_cer:
                best_val_cer = val_cer
                wandb.run.summary["best_val_cer"] = best_val_cer # WANDB: Update summary for best CER
                
                model_save_path = os.path.join(output_dir, f"best_ocr_bbox_model_epoch_{epoch + 1}_cer_{val_cer:.4f}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model locally to {model_save_path}")
                
                # Option: Save model as artifact
                # artifact = wandb.Artifact(f'model-{wandb.run.id}-best', type='model')
                # artifact.add_file(model_save_path)
                # wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch+1}"])
                # print(f"Saved best model artifact to wandb")

        else:
            print("Skipping validation as validation dataloader is empty.")
            if (epoch + 1) % 10 == 0:
                 model_save_path = os.path.join(output_dir, f"ocr_bbox_model_epoch_{epoch + 1}_no_val.pth")
                 torch.save(model.state_dict(), model_save_path)
                 print(f"Saved model (no validation) locally to {model_save_path}")
                 # WANDB: Save checkpoint model as artifact
                #  artifact_ckpt = wandb.Artifact(f'model-{wandb.run.id}-ckpt-epoch-{epoch+1}', type='model')
                #  artifact_ckpt.add_file(model_save_path)
                #  wandb.log_artifact(artifact_ckpt)


    print("Training finished.")
    if len(val_dataloader) > 0:
        print(f"Best Validation CER: {best_val_cer:.4f}")
    else:
        print("Training finished (no validation performed).")

    # Test set evaluation (if available)
    test_dataset_dir = os.path.join(DATA_BASE_DIR, "test")
    test_csv_path = os.path.join(test_dataset_dir, "test_column_info.csv")

    if os.path.exists(test_csv_path):
        test_dataset = KuzushijiDataset(
            test_dataset_dir, tokenizer, val_transform, MAX_LABEL_LEN, IMAGE_CHANNELS,
            target_image_height=IMAGE_HEIGHT, target_image_width=IMAGE_WIDTH,
            image_col="column_image", label_col="unicode_ids", bbox_col="char_boxes_in_column"
        )
        if len(test_dataset) > 0:
            test_dataloader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, 
                num_workers=NUM_WORKERS
            )
            print("Evaluating on Test Set...")
            # Load the best model for testing
            # This assumes the best model was saved. If using wandb artifacts, you might download it.
            # For simplicity, we use the current model state or you can manually load the saved .pth.
            # Example: model.load_state_dict(torch.load(best_model_path))
            
            # WANDB: For test evaluation, pass a different epoch number or a specific identifier if needed for logging step
            test_char_loss, test_bbox_loss, test_combined_loss, test_cer = validate(
                model, test_dataloader, criterion_char, criterion_bbox, BBOX_LOSS_WEIGHT, device, tokenizer, epoch_num=EPOCHS # Log against final epoch num
            )
            print(f"Test Set Results: Char Loss: {test_char_loss:.4f}, BBox Loss: {test_bbox_loss:.4f}, Combined Loss: {test_combined_loss:.4f}, CER: {test_cer:.4f}")
            # WANDB: Log test metrics
            wandb.log({
                "test_char_loss": test_char_loss,
                "test_bbox_loss": test_bbox_loss,
                "test_combined_loss": test_combined_loss,
                "test_cer": test_cer,
            })
        else:
            print("Test dataset is empty or could not be loaded.")
    else:
        print(f"Test dataset CSV not found at {test_csv_path}. Skipping test set evaluation.")

    # WANDB: Finish the run
    wandb.finish()
