#!/usr/bin/env python3
"""
Simple test to verify all components work correctly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image


def test_image_resize():
    """Test image resizing functionality"""
    print("Testing image resize...")
    
    from data.trocr_dataset import ResizeWithPadding
    
    # Test ResizeWithPadding
    resize_transform = ResizeWithPadding(target_size=(1024, 64))
    
    # Create test image
    test_image = Image.new('RGB', (100, 800), color=(128, 128, 128))
    resized_image = resize_transform(test_image)
    
    assert resized_image.size == (64, 1024), f"Expected (64, 1024), got {resized_image.size}"
    print("✅ Image resize test passed")
    return True

def test_tokenizer():
    """Test tokenizer loading"""
    print("Testing tokenizer...")
    
    from transformers import AutoTokenizer
    
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"
    
    if not os.path.exists(decoder_path):
        print(f"❌ Decoder path not found: {decoder_path}")
        return False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(decoder_path)
        print(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    
    from models.trocr_model import TrOCRModel
    
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"
    
    encoder_config = {
        "image_size": (1024, 64),
        "patch_size": (16, 16),
        "num_channels": 3,
        "hidden_size": 256,  # Smaller for testing
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 1024,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }
    
    try:
        model = TrOCRModel(
            encoder_config=encoder_config,
            decoder_path=decoder_path,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=10,
        )
        print("✅ Model created successfully")
        return True, model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None

def test_forward_pass(model):
    """Test forward pass"""
    print("Testing forward pass...")
    
    try:
        # Create dummy input
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 1024, 64)
        labels = torch.randint(0, len(model.tokenizer), (batch_size, 32))
        
        with torch.no_grad():
            outputs = model(pixel_values, labels)
            
        print(f"✅ Forward pass successful, loss: {outputs['loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_inference(model):
    """Test inference"""
    print("Testing inference...")
    
    try:
        # Create dummy input
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 1024, 64)
        
        with torch.no_grad():
            outputs = model(pixel_values, labels=None)
            generated_ids = outputs['generated_ids']
            predictions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        print(f"✅ Inference successful, prediction: '{predictions[0][:50]}...'")
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation with small sample"""
    print("Testing dataset creation...")
    
    from data.trocr_dataset import TrOCRDataset
    
    csv_path = "data/processed_v2/column_info.csv"
    image_root = "data/processed_v2/column_images"
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return False
    
    try:
        dataset = TrOCRDataset(
            csv_path=csv_path,
            image_root=image_root,
            tokenizer_path=decoder_path,
            image_size=(1024, 64),
            split="train",
            train_ratio=0.99,  # Use most data for training
            val_ratio=0.005,
            test_ratio=0.005,
        )
        
        print(f"✅ Dataset created, size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample image shape: {sample['pixel_values'].shape}")
            print(f"   Sample text: '{sample['text']}'")
        
        return True
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Simple TrOCR Test")
    print("=" * 40)
    
    tests = [
        test_image_resize,
        test_tokenizer,
        test_dataset_creation,
    ]
    
    results = []
    model = None
    
    # Run basic tests
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Test model creation and operations
    try:
        print("Testing model operations...")
        success, model = test_model_creation()
        results.append(success)
        print()
        
        if success and model is not None:
            # Test forward pass
            success = test_forward_pass(model)
            results.append(success)
            print()
            
            # Test inference
            success = test_inference(model)
            results.append(success)
            print()
        else:
            results.extend([False, False])
    except Exception as e:
        print(f"❌ Model tests failed: {e}")
        results.extend([False, False, False])
    
    # Summary
    print("=" * 40)
    print("Test Summary:")
    test_names = [
        "Image Resize", 
        "Tokenizer", 
        "Dataset Creation", 
        "Model Creation", 
        "Forward Pass", 
        "Inference"
    ]
    
    for name, result in zip(test_names, results, strict=False):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
