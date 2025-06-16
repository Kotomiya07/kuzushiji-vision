#!/usr/bin/env python3
"""
Test script to verify TrOCR setup and data loading
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from data.trocr_dataset import TrOCRDataset
from models.trocr_model import TrOCRModel


def test_tokenizer():
    """Test tokenizer loading"""
    print("Testing tokenizer...")
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    try:
        tokenizer = AutoTokenizer.from_pretrained(decoder_path)
        print("✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

        # Test tokenization
        test_text = "四修の事行者"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"  Test text: '{test_text}'")
        print(f"  Tokens: {tokens['input_ids']}")
        print(f"  Decoded: '{tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)}'")

        return True
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset...")

    csv_path = "data/processed_v2/column_info.csv"
    image_root = "data/processed_v2/column_images"
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False

    if not os.path.exists(image_root):
        print(f"✗ Image root not found: {image_root}")
        return False

    try:
        dataset = TrOCRDataset(
            csv_path=csv_path,
            image_root=image_root,
            tokenizer_path=decoder_path,
            split="train",
            image_size=(1024, 64),  # Column images are tall and narrow
            max_length=128,
        )

        print("✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Image shape: {sample['pixel_values'].shape}")
            print(f"  Labels shape: {sample['labels'].shape}")
            print(f"  Text: '{sample['text']}'")
            print(f"  Image path: {sample['image_path']}")

        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model():
    """Test model creation"""
    print("\nTesting model...")

    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    encoder_config = {
        "image_size": (1024, 64),  # Column images are tall and narrow
        "patch_size": (16, 16),
        "num_channels": 3,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }

    try:
        model = TrOCRModel(
            encoder_config=encoder_config,
            decoder_path=decoder_path,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=1000,
        )

        print("✓ Model created successfully")
        print(f"  Encoder hidden size: {model.encoder.config.hidden_size}")
        print(f"  Decoder hidden size: {model.decoder.config.hidden_size}")
        print(f"  Tokenizer vocab size: {len(model.tokenizer)}")

        # Test forward pass
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 1024, 64)  # Updated for new image size
        labels = torch.randint(0, len(model.tokenizer), (batch_size, 32))

        with torch.no_grad():
            outputs = model(pixel_values, labels)
            print("  Forward pass successful")
            print(f"  Output keys: {list(outputs.keys())}")
            print(f"  Logits shape: {outputs['logits'].shape}")
            print(f"  Loss: {outputs['loss'].item():.4f}")

        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_inference():
    """Test model inference"""
    print("\nTesting inference...")

    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    encoder_config = {
        "image_size": (1024, 64),  # Column images are tall and narrow
        "patch_size": (16, 16),
        "num_channels": 3,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }

    try:
        model = TrOCRModel(
            encoder_config=encoder_config,
            decoder_path=decoder_path,
        )
        model.eval()

        # Test inference
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 1024, 64)  # Updated for new image size

        with torch.no_grad():
            outputs = model(pixel_values, labels=None)
            generated_ids = outputs["generated_ids"]
            predictions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            print("✓ Inference successful")
            print(f"  Generated IDs shape: {generated_ids.shape}")
            print(f"  Prediction: '{predictions[0]}'")

        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def main():
    """Run all tests"""
    print("TrOCR Setup Test")
    print("=" * 50)

    tests = [
        test_tokenizer,
        test_dataset,
        test_model,
        test_inference,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Summary:")
    test_names = ["Tokenizer", "Dataset", "Model", "Inference"]
    for name, result in zip(test_names, results, strict=False):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
