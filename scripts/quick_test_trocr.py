#!/usr/bin/env python3
"""
Quick test for TrOCR training with minimal setup
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytorch_lightning as pl
import torch

from data.trocr_dataset import create_trocr_dataloaders
from models.trocr_model import TrOCRModel


def main():
    print("Quick TrOCR Training Test")
    print("=" * 40)

    # Set random seed
    pl.seed_everything(42)

    # Configuration
    csv_path = "data/processed_v2/column_info.csv"
    image_root = "data/processed_v2/column_images"
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    # Check paths
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return False

    if not os.path.exists(decoder_path):
        print(f"❌ Decoder not found: {decoder_path}")
        return False

    print("✅ All paths exist")

    # Create small dataloaders for testing
    print("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_trocr_dataloaders(
            csv_path=csv_path,
            image_root=image_root,
            tokenizer_path=decoder_path,
            batch_size=2,
            num_workers=0,  # No multiprocessing for debugging
            image_size=(1024, 64),  # Column images are tall and narrow
            max_length=64,  # Shorter sequences for faster training
            train_ratio=0.98,  # Use 98% for training
            val_ratio=0.01,  # 1% for validation
            test_ratio=0.01,  # 1% for test
        )
        print("✅ Dataloaders created")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        return False

    # Test a batch
    print("Testing batch loading...")
    try:
        batch = next(iter(train_loader))
        print("✅ Batch loaded")
        print(f"   Pixel values: {batch['pixel_values'].shape}")
        print(f"   Labels: {batch['labels'].shape}")
        print(f"   Sample text: '{batch['text'][0]}'")
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        return False

    # Create model
    print("Creating model...")
    try:
        encoder_config = {
            "image_size": (1024, 64),  # Column images are tall and narrow
            "patch_size": (16, 16),
            "num_channels": 3,
            "hidden_size": 256,  # Smaller for faster training
            "num_hidden_layers": 6,  # Fewer layers
            "num_attention_heads": 8,
            "intermediate_size": 1024,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
        }

        model = TrOCRModel(
            encoder_config=encoder_config,
            decoder_path=decoder_path,
            learning_rate=1e-3,  # Higher learning rate for quick test
            weight_decay=0.01,
            warmup_steps=10,
        )
        print("✅ Model created")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test forward pass
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = model(batch["pixel_values"], batch["labels"])
            print("✅ Forward pass successful")
            print(f"   Loss: {outputs['loss'].item():.4f}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Quick training test (1 step)
    print("Testing training step...")
    try:
        # Create minimal trainer
        trainer = pl.Trainer(
            max_epochs=1,
            max_steps=2,  # Just 2 steps
            devices=1 if torch.cuda.is_available() else 0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,  # No logging
            enable_checkpointing=False,  # No checkpointing
            enable_progress_bar=True,
        )

        # Train for 2 steps
        trainer.fit(model, train_loader, val_loader)
        print("✅ Training test successful")

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

    print("\n🎉 All tests passed! TrOCR setup is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
