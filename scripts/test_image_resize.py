#!/usr/bin/env python3
"""
Test image resizing functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torchvision.transforms as transforms
from PIL import Image

from data.trocr_dataset import ResizeWithPadding


def test_resize_with_padding():
    """Test the ResizeWithPadding transform"""
    print("Testing ResizeWithPadding...")
    
    # Create transform
    resize_transform = ResizeWithPadding(target_size=(1024, 64))
    
    # Test with different aspect ratios
    test_cases = [
        (100, 800),   # Tall narrow (similar to column)
        (200, 400),   # Tall narrow
        (400, 200),   # Wide short
        (64, 1024),   # Already correct size
        (32, 512),    # Half size
    ]
    
    for width, height in test_cases:
        # Create test image
        test_image = Image.new('RGB', (width, height), color=(128, 128, 128))
        print(f"  Original: {test_image.size} (W×H)")
        
        # Apply transform
        resized_image = resize_transform(test_image)
        print(f"  Resized:  {resized_image.size} (W×H)")
        
        # Check if size is correct
        assert resized_image.size == (64, 1024), f"Expected (64, 1024), got {resized_image.size}"
        print("  ✅ Size correct")
        print()
    
    print("✅ ResizeWithPadding test passed!")
    return True


def test_full_transform():
    """Test the full image transform pipeline"""
    print("Testing full transform pipeline...")
    
    # Create transform pipeline
    transform = transforms.Compose([
        ResizeWithPadding(target_size=(1024, 64), fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Create test image
    test_image = Image.new('RGB', (100, 800), color=(128, 128, 128))
    print(f"Original image size: {test_image.size}")
    
    # Apply transform
    tensor = transform(test_image)
    print(f"Tensor shape: {tensor.shape}")
    
    # Check tensor properties
    assert tensor.shape == (3, 1024, 64), f"Expected (3, 1024, 64), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    
    # Check normalization (should be in range [-1, 1])
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0, f"Values out of range: min={tensor.min()}, max={tensor.max()}"
    
    print("✅ Full transform test passed!")
    return True


def test_vit_compatibility():
    """Test compatibility with ViT input requirements"""
    print("Testing ViT compatibility...")
    
    # Create batch of images
    batch_size = 2
    transform = transforms.Compose([
        ResizeWithPadding(target_size=(1024, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Create test images
    images = []
    for i in range(batch_size):
        test_image = Image.new('RGB', (100 + i*50, 800 + i*100), color=(128, 128, 128))
        tensor = transform(test_image)
        images.append(tensor)
    
    # Stack into batch
    batch = torch.stack(images)
    print(f"Batch shape: {batch.shape}")
    
    # Check batch properties
    assert batch.shape == (batch_size, 3, 1024, 64), f"Expected ({batch_size}, 3, 1024, 64), got {batch.shape}"
    
    # Calculate number of patches for ViT
    patch_size = 16
    num_patches_h = 1024 // patch_size  # 64
    num_patches_w = 64 // patch_size    # 4
    total_patches = num_patches_h * num_patches_w  # 256
    
    print(f"Number of patches: {num_patches_h} × {num_patches_w} = {total_patches}")
    print("✅ ViT compatibility test passed!")
    return True


def main():
    """Run all tests"""
    print("Image Resize Test")
    print("=" * 40)
    
    tests = [
        test_resize_with_padding,
        test_full_transform,
        test_vit_compatibility,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
            print()
    
    print("=" * 40)
    print("Test Summary:")
    test_names = ["ResizeWithPadding", "Full Transform", "ViT Compatibility"]
    for name, result in zip(test_names, results, strict=False):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
