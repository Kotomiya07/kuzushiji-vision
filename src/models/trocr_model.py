from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from schedulefree import RAdamScheduleFree
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, ViTConfig


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) is not tuple else grid_size
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

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CustomPatchEmbeddings(nn.Module):
    """Custom patch embeddings that can handle non-square images"""

    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        # Calculate number of patches
        self.num_patches_h = image_size[0] // patch_size[0]
        self.num_patches_w = image_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding layer
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape

        # Check input size
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # Apply patch embedding
        embeddings = self.projection(pixel_values)  # (batch_size, hidden_size, num_patches_h, num_patches_w)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_size)

        return embeddings


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for TrOCR"""

    def __init__(
        self,
        image_size: tuple[int, int] = (1024, 64),
        patch_size: tuple[int, int] = (16, 16),
        num_channels: int = 3,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # Store image and patch dimensions
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Create configuration
        self.config = ViTConfig(
            image_size=max(self.image_size),  # Use max for config, but we'll handle actual size
            patch_size=self.patch_size[0],
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=1e-12,
        )

        # Custom patch embeddings
        self.patch_embeddings = CustomPatchEmbeddings(self.image_size, self.patch_size, num_channels, hidden_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))

        # Initialize position embeddings
        self._init_position_embeddings()

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Transformer encoder layers
        from transformers.models.vit.modeling_vit import ViTEncoder as TransformerEncoder

        self.encoder = TransformerEncoder(self.config)

        # Layer norm
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def _init_position_embeddings(self):
        """Initialize position embeddings using 2D sincos embedding"""
        num_patches_h = self.image_size[0] // self.patch_size[0]  # 1024 // 16 = 64
        num_patches_w = self.image_size[1] // self.patch_size[1]  # 64 // 16 = 4

        # Initialize position embeddings using 2D sincos embedding
        pos_embed = get_2d_sincos_pos_embed(self.config.hidden_size, (num_patches_h, num_patches_w))

        # Add CLS token embedding (zeros)
        pos_embed_with_cls = np.zeros((1 + self.num_patches, self.config.hidden_size))
        pos_embed_with_cls[1:, :] = pos_embed

        # Set the position embeddings
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed_with_cls).float().unsqueeze(0))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch_size, num_channels, height, width)

        Returns:
            encoder_outputs: (batch_size, num_patches, hidden_size)
        """
        batch_size = pixel_values.shape[0]

        # Get patch embeddings
        embeddings = self.patch_embeddings(pixel_values)  # (batch_size, num_patches, hidden_size)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Add position embeddings
        embeddings = embeddings + self.position_embeddings

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Apply transformer encoder
        encoder_outputs = self.encoder(embeddings)
        sequence_output = encoder_outputs.last_hidden_state

        # Apply layer norm
        sequence_output = self.layernorm(sequence_output)

        # Remove CLS token, keep only patch embeddings
        sequence_output = sequence_output[:, 1:, :]  # (batch_size, num_patches, hidden_size)

        return sequence_output


class TrOCRModel(pl.LightningModule):
    """TrOCR Model with ViT Encoder and RoBERTa Decoder"""

    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_path: str,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load tokenizer and decoder
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_path)

        # Load decoder configuration and modify for cross-attention
        decoder_config = AutoConfig.from_pretrained(decoder_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.output_hidden_states = True

        # Load the pre-trained decoder
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_path, config=decoder_config)

        # Initialize ViT encoder
        self.encoder = ViTEncoder(
            image_size=encoder_config.get("image_size", (1024, 64)),
            patch_size=encoder_config.get("patch_size", (16, 16)),
            num_channels=encoder_config.get("num_channels", 3),
            hidden_size=encoder_config.get("hidden_size", 768),
            num_hidden_layers=encoder_config.get("num_hidden_layers", 12),
            num_attention_heads=encoder_config.get("num_attention_heads", 12),
            intermediate_size=encoder_config.get("intermediate_size", 3072),
            hidden_dropout_prob=encoder_config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=encoder_config.get("attention_probs_dropout_prob", 0.1),
        )

        # Projection layer to match encoder and decoder dimensions
        encoder_dim = self.encoder.config.hidden_size
        decoder_dim = self.decoder.config.hidden_size

        if encoder_dim != decoder_dim:
            self.encoder_decoder_proj = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_decoder_proj = nn.Identity()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (batch_size, num_channels, height, width)
            labels: (batch_size, sequence_length) - optional for training

        Returns:
            Dictionary containing logits and optionally loss
        """
        # Encode images
        encoder_outputs = self.encoder(pixel_values)  # (batch_size, num_patches, encoder_dim)
        encoder_outputs = self.encoder_decoder_proj(encoder_outputs)  # (batch_size, num_patches, decoder_dim)

        if labels is not None:
            # Training mode
            # Prepare decoder inputs (shift labels right)
            decoder_input_ids = labels[:, :-1].contiguous()
            decoder_labels = labels[:, 1:].contiguous()

            # Create attention mask for decoder inputs
            decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).long()

            # Forward through decoder
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs,
                use_cache=False,
                return_dict=True,
            )

            logits = decoder_outputs.logits

            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))

            return {
                "logits": logits,
                "loss": loss,
            }
        else:
            # Inference mode - generate text
            batch_size = pixel_values.size(0)

            # Start with BOS token
            input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=pixel_values.device)

            # Generate sequence
            with torch.no_grad():
                generated = self.decoder.generate(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_outputs,
                    max_length=128,  # Adjust as needed
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )

            return {
                "generated_ids": generated.sequences,
                "logits": None,
            }

    def training_step(self, batch, batch_idx):
        """Training step"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels)
        loss = outputs["loss"]

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels)
        loss = outputs["loss"]

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generate predictions for evaluation
        if batch_idx < 5:  # Only evaluate a few batches to save time
            pred_outputs = self(pixel_values, labels=None)
            generated_ids = pred_outputs["generated_ids"]

            # Decode predictions and targets
            predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Calculate character error rate (CER) for first sample in batch
            if len(predictions) > 0 and len(targets) > 0:
                pred_text = predictions[0]
                target_text = targets[0]
                cer = self.calculate_cer(pred_text, target_text)
                self.log("val_cer", cer, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels)
        loss = outputs["loss"]

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generate predictions for evaluation
        pred_outputs = self(pixel_values, labels=None)
        generated_ids = pred_outputs["generated_ids"]

        # Decode predictions and targets
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate metrics for all samples in batch
        batch_cer = []
        for pred_text, target_text in zip(predictions, targets, strict=False):
            cer = self.calculate_cer(pred_text, target_text)
            batch_cer.append(cer)

        # Log average CER for this batch
        avg_cer = sum(batch_cer) / len(batch_cer) if batch_cer else 0.0
        self.log("test_cer", avg_cer, on_step=False, on_epoch=True, prog_bar=True)

        # Log some example predictions for inspection
        if batch_idx < 3:  # Log first 3 batches
            for i, (pred, target) in enumerate(zip(predictions[:2], targets[:2], strict=False)):  # First 2 samples per batch
                print(f"Batch {batch_idx}, Sample {i}:")
                print(f"  Target: {target}")
                print(f"  Prediction: {pred}")
                print(f"  CER: {batch_cer[i]:.4f}")

        return loss

    def calculate_cer(self, pred_text: str, target_text: str) -> float:
        """Calculate Character Error Rate"""
        if len(target_text) == 0:
            return 1.0 if len(pred_text) > 0 else 0.0

        # Simple character-level edit distance
        import editdistance

        distance = editdistance.eval(pred_text, target_text)
        cer = distance / len(target_text)
        return cer

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = RAdamScheduleFree(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )
        return optimizer

    def on_train_epoch_start(self) -> None:
        """Set optimizer to train mode at the start of each training epoch"""
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "train"):
            optimizer.train()

    def on_validation_epoch_start(self) -> None:
        """Set optimizer to eval mode at the start of each validation epoch"""
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_validation_epoch_end(self) -> None:
        """Set optimizer back to train mode after validation"""
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "train"):
            optimizer.train()

    def on_test_epoch_start(self) -> None:
        """Set optimizer to eval mode at the start of each test epoch"""
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_predict_epoch_start(self) -> None:
        """Set optimizer to eval mode at the start of each predict epoch"""
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def decode_predictions(self, pixel_values: torch.Tensor) -> list[str]:
        """Decode predictions from images"""
        self.eval()
        with torch.no_grad():
            outputs = self(pixel_values, labels=None)
            generated_ids = outputs["generated_ids"]
            predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions
