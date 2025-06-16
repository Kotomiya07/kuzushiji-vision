#!/usr/bin/env python3
"""
Debug test to identify and fix issues
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image
import torchvision.transforms as transforms

def test_vit_encoder():
    """Test ViT encoder with correct image size"""
    print("Testing ViT encoder...")
    
    from models.trocr_model import ViTEncoder
    
    try:
        encoder = ViTEncoder(
            image_size=(1024, 64),
            patch_size=(16, 16),
            num_channels=3,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
        )
        
        print(f"✅ ViT encoder created")
        print(f"   Image size: {encoder.image_size}")
        print(f"   Patch size: {encoder.patch_size}")
        print(f"   Num patches: {encoder.num_patches}")
        
        # Test forward pass
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 1024, 64)
        
        with torch.no_grad():
            outputs = encoder(pixel_values)
            print(f"   Output shape: {outputs.shape}")
            print(f"   Expected: ({batch_size}, {encoder.num_patches}, {encoder.config.hidden_size})")
        
        return True
    except Exception as e:
        print(f"❌ ViT encoder test failed: {e}")
        return False

def test_image_transform():
    """Test image transform pipeline"""
    print("Testing image transform...")
    
    from data.trocr_dataset import ResizeWithPadding
    
    try:
        # Create transform
        transform = transforms.Compose([
            ResizeWithPadding(target_size=(1024, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Create test image
        test_image = Image.new('RGB', (100, 800), color=(128, 128, 128))
        tensor = transform(test_image)
        
        print(f"✅ Image transform successful")
        print(f"   Input size: {test_image.size}")
        print(f"   Output shape: {tensor.shape}")
        print(f"   Expected: (3, 1024, 64)")
        
        return tensor.shape == (3, 1024, 64)
    except Exception as e:
        print(f"❌ Image transform failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer"""
    print("Testing tokenizer...")
    
    from transformers import AutoTokenizer
    
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(decoder_path)
        
        # Test tokenization
        test_text = "四修の事行者"
        tokens = tokenizer(test_text, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
        
        print(f"✅ Tokenizer test successful")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Test text: '{test_text}'")
        print(f"   Token shape: {tokens['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_model_integration():
    """Test model integration"""
    print("Testing model integration...")
    
    from models.trocr_model import TrOCRModel
    
    decoder_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"
    
    encoder_config = {
        "image_size": (1024, 64),
        "patch_size": (16, 16),
        "num_channels": 3,
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 1024,
    }
    
    try:
        model = TrOCRModel(
            encoder_config=encoder_config,
            decoder_path=decoder_path,
        )
        
        print(f"✅ Model created")
        print(f"   Encoder hidden size: {model.encoder.config.hidden_size}")
        print(f"   Decoder hidden size: {model.decoder.config.hidden_size}")
        
        # Test forward pass
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 1024, 64)
        labels = torch.randint(0, len(model.tokenizer), (batch_size, 32))
        
        with torch.no_grad():
            outputs = model(pixel_values, labels)
            print(f"   Forward pass successful")
            print(f"   Loss: {outputs['loss'].item():.4f}")
            print(f"   Logits shape: {outputs['logits'].shape}")
        
        # Test inference
        with torch.no_grad():
            outputs = model(pixel_values, labels=None)
            print(f"   Inference successful")
            print(f"   Generated shape: {outputs['generated_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests"""
    print("Debug Test")
    print("=" * 40)
    
    tests = [
        test_image_transform,
        test_tokenizer,
        test_vit_encoder,
        test_model_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 40)
    print("Test Summary:")
    test_names = ["Image Transform", "Tokenizer", "ViT Encoder", "Model Integration"]
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
