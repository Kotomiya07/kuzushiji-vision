import pytest
import torch
from torchvision import transforms
from PIL import Image
import os
import json
import shutil
import tempfile
from pathlib import Path

# Adjust import path if tests directory is not at the same level as data_loader.py
# For example, if project root is parent of tests/ and data_loader.py:
# import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import (
    build_char_to_int_map, 
    OCRDataset, 
    get_data_loader,
    MAX_LABEL_LENGTH as DEFAULT_MAX_LABEL_LENGTH # Use an alias for clarity
)

# --- Test Fixtures ---

@pytest.fixture(scope="module") # Use module scope for efficiency if many tests use it
def temp_data_dir():
    """Creates a temporary directory with a dummy dataset structure."""
    base_dir = tempfile.mkdtemp(prefix="ocr_test_data_")
    print(f"Created temporary data directory: {base_dir}")

    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    num_samples_per_split = {"train": 3, "val": 2, "test": 2}

    for split, num_samples in num_samples_per_split.items():
        img_dir = Path(base_dir) / split / "images"
        lbl_dir = Path(base_dir) / split / "labels"
        bbox_dir = Path(base_dir) / split / "bounding_boxes"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(bbox_dir, exist_ok=True)

        for i in range(num_samples):
            img_filename = f"sample_{split}_{i}.png"
            txt_filename = f"sample_{split}_{i}.txt"
            json_filename = f"sample_{split}_{i}.json"

            try:
                dummy_img = Image.new('RGB', (100, 30), color='black')
                dummy_img.save(img_dir / img_filename)
            except Exception as e:
                print(f"Warning: Could not create dummy image {img_filename}: {e}")
                continue

            label_len = (i % 5) + 5 # Vary label length (5 to 9 chars)
            label_text = "".join(chars[j % len(chars)] for j in range(label_len))
            with open(lbl_dir / txt_filename, "w") as f:
                f.write(label_text)

            bboxes = [[j * 10, 5, (j * 10) + 8, 25] for j in range(len(label_text))]
            with open(bbox_dir / json_filename, "w") as f:
                json.dump(bboxes, f)
    
    yield str(base_dir) # Provide the path as a string

    print(f"Cleaning up temporary data directory: {base_dir}")
    shutil.rmtree(base_dir)


@pytest.fixture(scope="module")
def char_maps(temp_data_dir): # Depends on temp_data_dir
    """Provides char_to_int and int_to_char mappings."""
    char_to_int, int_to_char = build_char_to_int_map(temp_data_dir)
    return char_to_int, int_to_char

# --- Test Cases ---

def test_build_char_to_int_map(char_maps): # Uses char_maps fixture
    char_to_int, int_to_char = char_maps
    
    assert "<PAD>" in char_to_int, "PAD token missing"
    assert "<UNK>" in char_to_int, "UNK token missing"
    assert char_to_int["<PAD>"] == 0, "PAD token should be 0"
    
    # Check consistency
    for char, code in char_to_int.items():
        assert int_to_char[code] == char, f"Mismatch for char '{char}' and code {code}"
        
    # Verify vocab size based on dummy data (chars 'a'-'z', '0'-'9')
    # The dummy data generator uses 'abcdefghijklmnopqrstuvwxyz0123456789'
    # Max label length varies 5-9. All these characters should be present.
    expected_chars = set("abcdefghijklmnopqrstuvwxyz0123456789")
    present_chars_in_map = set(char_to_int.keys()) - {"<PAD>", "<UNK>"}
    
    assert expected_chars.issubset(present_chars_in_map), \
        f"Not all expected characters found in map. Missing: {expected_chars - present_chars_in_map}"
    
    # Vocab size = unique data chars + <PAD> + <UNK>
    assert len(char_to_int) == len(expected_chars) + 2, "Vocabulary size mismatch"


def test_ocr_dataset(temp_data_dir, char_maps):
    char_to_int, _ = char_maps
    max_label_length = 15 # Test with a specific max_label_length for dataset
    
    # Basic image transform for testing
    image_transform = transforms.Compose([
        transforms.Resize((64, 256)), # Example dimensions
        transforms.ToTensor()
    ])

    dataset = OCRDataset(
        data_dir=temp_data_dir,
        split="train", # Test with train split
        image_transform=image_transform,
        char_to_int=char_to_int,
        max_label_length=max_label_length
    )

    # Test __len__
    assert len(dataset) == 3, "Dataset length mismatch for 'train' split" # 3 samples in dummy train split

    # Test __getitem__ for one sample
    if len(dataset) > 0:
        sample = dataset[0]
        
        assert "image" in sample, "Missing 'image' key"
        assert "label" in sample, "Missing 'label' key"
        assert "bounding_boxes" in sample, "Missing 'bounding_boxes' key"
        assert "label_lengths" in sample, "Missing 'label_lengths' key"

        # Image assertions
        assert isinstance(sample["image"], torch.Tensor), "Image is not a Tensor"
        assert sample["image"].shape[0] == 3, "Image should have 3 channels (RGB)" # Assuming ToTensor makes C,H,W
        assert sample["image"].shape[1] == 64, "Image height mismatch after transform"
        assert sample["image"].shape[2] == 256, "Image width mismatch after transform"
        # Value range for ToTensor is [0, 1]
        assert sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0, "Image values out of [0,1] range"

        # Label assertions
        assert isinstance(sample["label"], torch.Tensor), "Label is not a Tensor"
        assert sample["label"].dtype == torch.long, "Label dtype is not torch.long"
        assert sample["label"].shape == (max_label_length,), f"Label shape mismatch, expected ({max_label_length},)"
        assert sample["label"].min() >= 0 and sample["label"].max() < len(char_to_int), "Label values out of char_map range"

        # Bounding Boxes assertions
        assert isinstance(sample["bounding_boxes"], torch.Tensor), "Bounding_boxes is not a Tensor"
        assert sample["bounding_boxes"].dtype == torch.float32, "Bounding_boxes dtype is not torch.float32"
        assert sample["bounding_boxes"].shape == (max_label_length, 4), f"Bounding_boxes shape mismatch, expected ({max_label_length}, 4)"

        # Label Lengths assertions
        assert isinstance(sample["label_lengths"], torch.Tensor), "Label_lengths is not a Tensor"
        assert sample["label_lengths"].dtype == torch.long, "Label_lengths dtype is not torch.long"
        assert sample["label_lengths"].ndim == 0, "Label_lengths should be a scalar tensor" # PyTorch 0-dim tensor
        
        # Verify label_lengths value (specific to the first sample of dummy data if predictable)
        # First train sample (i=0) has label_len = (0 % 5) + 5 = 5
        # This length should be capped by max_label_length if shorter.
        expected_len = min(5, max_label_length)
        assert sample["label_lengths"].item() == expected_len, \
            f"Label length for first sample incorrect. Expected {expected_len}, got {sample['label_lengths'].item()}"
        
        # Check padding in label based on label_lengths
        if expected_len < max_label_length:
            assert sample["label"][expected_len:].eq(char_to_int["<PAD>"]).all(), \
                "Label not correctly padded with <PAD> token"


def test_get_data_loader(temp_data_dir, char_maps):
    char_to_int, _ = char_maps
    max_label_length = 15 # Test with a specific max_label_length
    batch_size = 2

    data_loader = get_data_loader(
        data_dir=temp_data_dir,
        split="train",
        batch_size=batch_size,
        num_workers=0, # Use 0 for test simplicity
        char_to_int=char_to_int,
        max_label_length=max_label_length,
        shuffle=False # Disable shuffle for predictable batch content if needed
    )

    assert data_loader is not None
    
    # Iterate one batch
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        pytest.fail("DataLoader is empty or could not fetch a batch.")

    assert "image" in batch
    assert "label" in batch
    assert "bounding_boxes" in batch
    assert "label_lengths" in batch

    # Check shapes (assuming C, H, W for image after transforms)
    # Target height/width are from get_data_loader defaults or passed if customized
    # For this test, get_data_loader uses DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH
    from data_loader import DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH
    
    # Number of samples in train split is 3, batch_size is 2. First batch has 2 samples.
    current_batch_size = min(batch_size, 3) 

    assert batch["image"].shape == (current_batch_size, 3, DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH), \
        f"Batch image shape mismatch. Expected ({current_batch_size}, 3, {DEFAULT_TARGET_HEIGHT}, {DEFAULT_TARGET_WIDTH})"
    assert batch["label"].shape == (current_batch_size, max_label_length), \
        f"Batch label shape mismatch. Expected ({current_batch_size}, {max_label_length})"
    assert batch["bounding_boxes"].shape == (current_batch_size, max_label_length, 4), \
        f"Batch bounding_boxes shape mismatch. Expected ({current_batch_size}, {max_label_length}, 4)"
    assert batch["label_lengths"].shape == (current_batch_size,), \
        f"Batch label_lengths shape mismatch. Expected ({current_batch_size},)"

    # Check label_lengths content for the first two samples in train split
    # Sample 0: len = (0%5)+5 = 5
    # Sample 1: len = (1%5)+5 = 6
    expected_lengths = torch.tensor([
        min(5, max_label_length), 
        min(6, max_label_length)
    ], dtype=torch.long)
    
    if current_batch_size >= 2: # Ensure we have at least 2 samples in the batch
        assert torch.equal(batch["label_lengths"][:2], expected_lengths[:current_batch_size]), \
            f"Batch label_lengths content mismatch. Expected {expected_lengths[:current_batch_size]}, got {batch['label_lengths'][:2]}"

# To run these tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the directory containing 'tests' and run: pytest
# Or, if 'tests' is in the project root: pytest tests/test_data_loader.py
# Make sure data_loader.py is in the PYTHONPATH or in a location discoverable by Python.
# If data_loader.py is in the parent directory of 'tests/':
# Add this to the top of test_data_loader.py:
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from data_loader import ...
# (Already handled by the placeholder comment in the template)
#
# Note on DEFAULT_MAX_LABEL_LENGTH:
# The data_loader.py defines MAX_LABEL_LENGTH. If test needs its own, it's fine.
# Here, the tests use a local max_label_length = 15 for dataset/loader tests for clarity.
# build_char_to_int_map doesn't depend on it.
# OCRDataset and get_data_loader take it as an argument.
# If these tests were to rely on the global default from data_loader,
# it would be DEFAULT_MAX_LABEL_LENGTH (aliased).
# The current setup correctly uses a test-specific max_label_length.
#
# Note on image normalization in test_ocr_dataset:
# The test currently checks if pixel values are in [0, 1] because only ToTensor is applied.
# If Normalize transform were added to the test's image_transform, 
# the value range check would need to be adjusted or removed.
# get_data_loader *does* apply normalization, so if testing its output image values,
# that would be different. This test only checks OCRDataset with a minimal transform.
