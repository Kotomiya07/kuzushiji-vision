#!/usr/bin/env python3
"""
Training script for TrOCR model with ViT encoder and RoBERTa decoder
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.trocr_dataset import create_trocr_dataloaders
from models.trocr_model import TrOCRModel
from src.callbacks.ema import EMACallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train TrOCR model")

    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data/processed_v2/column_info.csv", help="Path to column_info.csv")
    parser.add_argument(
        "--image_root", type=str, default="data/processed_v2/column_images", help="Root directory for column images"
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default="experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000",
        help="Path to pre-trained RoBERTa decoder",
    )

    # Model arguments
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 64], help="Image size (height width)")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[16, 16], help="Patch size (height width)")
    parser.add_argument("--encoder_hidden_size", type=int, default=768, help="ViT encoder hidden size")
    parser.add_argument("--encoder_num_layers", type=int, default=12, help="Number of ViT encoder layers")
    parser.add_argument("--encoder_num_heads", type=int, default=8, help="Number of ViT encoder attention heads")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW optimizer")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Epsilon for AdamW optimizer")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")

    # Split ratios
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test data ratio")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="experiments/trocr", help="Output directory for checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="trocr_vit_roberta", help="Experiment name")

    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed", choices=["16-mixed", "32", "bf16-mixed"], help="Training precision"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Decoder path: {args.decoder_path}")

    # Check if decoder path exists
    if not os.path.exists(args.decoder_path):
        raise FileNotFoundError(f"Decoder path not found: {args.decoder_path}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_trocr_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        tokenizer_path=args.decoder_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create encoder configuration
    encoder_config = {
        "image_size": tuple(args.image_size),
        "patch_size": tuple(args.patch_size),
        "num_channels": 3,
        "hidden_size": args.encoder_hidden_size,
        "num_hidden_layers": args.encoder_num_layers,
        "num_attention_heads": args.encoder_num_heads,
        "intermediate_size": args.encoder_hidden_size * 4,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }

    # Create model
    print("Creating model...")
    model = TrOCRModel(
        encoder_config=encoder_config,
        decoder_path=args.decoder_path,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="trocr-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_cer",
        mode="min",
        patience=10,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    ema_callback = EMACallback(decay=0.9999)

    # Create logger
    logger = WandbLogger(
        project="trocr",
        name=args.experiment_name,
        save_dir=output_dir,
        log_model=True,
    )

    # Create trainer
    if args.gpus > 0:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            devices=args.gpus,
            accelerator="gpu",
            precision=args.precision,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=logger,
            log_every_n_steps=50,
            val_check_interval=0.5,  # Check validation every half epoch
            gradient_clip_val=1.0,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="cpu",
            precision="32",  # CPU doesn't support mixed precision
            callbacks=[checkpoint_callback, early_stopping, lr_monitor, ema_callback],
            logger=logger,
            log_every_n_steps=50,
            val_check_interval=0.5,  # Check validation every half epoch
            gradient_clip_val=1.0,
        )

    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test model
    print("Testing model...")
    trainer.test(model, test_loader, ckpt_path="best")

    print(f"Training completed! Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
