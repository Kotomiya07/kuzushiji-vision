import pytest
import torch
import pytorch_lightning as pl

# Adjust import path if necessary
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.ocr_model import OCRModel # Assuming ocr_model.py is in the parent directory or PYTHONPATH

# --- Test Setup & Fixtures ---

@pytest.fixture(scope="module")
def model_params():
    """Provides common parameters for model initialization."""
    char_to_int = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "c": 4} # Added 'c' for diversity
    int_to_char = {v: k for k, v in char_to_int.items()}
    return {
        "num_chars": len(char_to_int),
        "int_to_char": int_to_char,
        "max_label_length": 10,
        "input_height": 32, # Smaller for faster tests
        "input_width": 128, # Smaller for faster tests
        "input_channels": 3,
        "rnn_hidden_size": 64, # Smaller for faster tests
        "rnn_layers": 1,       # Smaller for faster tests
        "encoder_name": "resnet34", # Using resnet34 as it's common
        "pretrained_encoder": False, # Avoid downloads during tests
        "learning_rate": 1e-4,
        "lambda_bbox": 0.5
    }

@pytest.fixture(scope="module")
def ocr_model_instance(model_params):
    """Initializes an OCRModel instance for testing."""
    return OCRModel(**model_params)

# --- Test Cases ---

def test_model_initialization(ocr_model_instance):
    assert isinstance(ocr_model_instance, pl.LightningModule), "Model is not a PyTorch Lightning module."
    # Check if some key layers exist (optional, but good for sanity)
    assert hasattr(ocr_model_instance, "encoder"), "Model missing encoder"
    assert hasattr(ocr_model_instance, "decoder"), "Model missing decoder"
    assert hasattr(ocr_model_instance, "char_fc"), "Model missing char_fc"
    assert hasattr(ocr_model_instance, "bbox_fc"), "Model missing bbox_fc"

def test_model_forward_pass(ocr_model_instance, model_params):
    batch_size = 2
    dummy_images = torch.randn(
        batch_size, 
        model_params["input_channels"], 
        model_params["input_height"], 
        model_params["input_width"]
    )

    ocr_model_instance.eval() # Set to eval mode for forward pass testing
    with torch.no_grad():
        outputs = ocr_model_instance(dummy_images)

    assert "char_logits" in outputs, "Output missing 'char_logits'"
    assert "bbox_preds" in outputs, "Output missing 'bbox_preds'"

    # char_logits shape: (batch_size, sequence_length_from_model, num_chars)
    # sequence_length_from_model depends on input_width and encoder downsampling
    # For ResNet34, typical downsampling is 32x. If input_width=128, seq_len might be 128/32 = 4.
    # This can vary based on exact ResNet structure and adaptive pooling.
    # Let's check dimensions dynamically.
    assert outputs["char_logits"].shape[0] == batch_size
    assert outputs["char_logits"].shape[2] == model_params["num_chars"]
    model_sequence_length = outputs["char_logits"].shape[1] # T_model
    assert model_sequence_length > 0, "Model output sequence length is zero"

    # bbox_preds shape: (batch_size, sequence_length_from_model, 4)
    # This matches the "per time step" prediction strategy.
    assert outputs["bbox_preds"].shape == (batch_size, model_sequence_length, 4)


def test_model_step_methods(ocr_model_instance, model_params):
    batch_size = 2
    max_len = model_params["max_label_length"]
    num_c = model_params["num_chars"]

    # Create a dummy batch
    dummy_batch = {
        "image": torch.randn(
            batch_size, 
            model_params["input_channels"],
            model_params["input_height"],
            model_params["input_width"]
        ),
        # Labels: (N, S_max), ensure values are valid indices (not 0 for PAD in true part)
        "label": torch.randint(1, num_c, (batch_size, max_len), dtype=torch.long),
        # Label lengths: (N), ensure lengths are <= max_len and > 0
        "label_lengths": torch.randint(1, max_len + 1, (batch_size,), dtype=torch.long),
        # Bounding boxes: (N, S_max, 4)
        "bounding_boxes": torch.rand(batch_size, max_len, 4) * model_params["input_width"]
    }
    # Ensure label_lengths don't exceed max_len
    dummy_batch["label_lengths"] = torch.clamp(dummy_batch["label_lengths"], max=max_len)
    # Ensure labels are padded correctly according to their lengths for a more realistic test
    for i in range(batch_size):
        true_len = dummy_batch["label_lengths"][i].item()
        if true_len < max_len:
            dummy_batch["label"][i, true_len:] = model_params["int_to_char"].get("<PAD>", 0)


    # Test training_step
    ocr_model_instance.train() # Set to train mode
    train_loss = ocr_model_instance.training_step(dummy_batch, 0)
    assert isinstance(train_loss, torch.Tensor), "Training loss is not a Tensor"
    assert train_loss.ndim == 0, "Training loss is not a scalar"
    assert not torch.isnan(train_loss) and not torch.isinf(train_loss), "Training loss is NaN or Inf"

    # Test validation_step
    ocr_model_instance.eval() # Set to eval mode
    val_loss = ocr_model_instance.validation_step(dummy_batch, 0)
    assert isinstance(val_loss, torch.Tensor), "Validation loss is not a Tensor"
    assert val_loss.ndim == 0, "Validation loss is not a scalar"
    assert not torch.isnan(val_loss) and not torch.isinf(val_loss), "Validation loss is NaN or Inf"
    
    # Test test_step (similar to validation_step)
    test_loss = ocr_model_instance.test_step(dummy_batch, 0)
    assert isinstance(test_loss, torch.Tensor), "Test loss is not a Tensor"
    assert test_loss.ndim == 0, "Test loss is not a scalar"
    assert not torch.isnan(test_loss) and not torch.isinf(test_loss), "Test loss is NaN or Inf"


def test_calculate_cer(ocr_model_instance, model_params): # Pass model_params for int_to_char
    # Example data for CER
    # Need model_instance only if int_to_char is not passed directly, but it's static
    # _convert_to_strings needs int_to_char from model_params if we were to use it.
    # However, calculate_cer itself takes lists of strings.
    
    preds_str = ["ab", "cat", "test"]
    targets_str = ["ac", "cot", "test"]
    # Edit distances: "ab" vs "ac" -> 1 (b->c)
    #                 "cat" vs "cot" -> 1 (a->o)
    #                 "test" vs "test" -> 0
    # Target lengths: 2, 3, 4
    # CER = (1 + 1 + 0) / (2 + 3 + 4) = 2 / 9
    expected_cer = (1/2 + 1/3 + 0/4) / 3 # This is average of CERs per sample
    # The model's CER is (total_edit_distance / total_target_length)
    expected_cer_model_impl = (1 + 1 + 0) / (len("ac") + len("cot") + len("test")) # 2 / (2+3+4) = 2/9

    cer = OCRModel.calculate_cer(preds_str, targets_str)
    assert isinstance(cer, float), "CER should be a float"
    assert pytest.approx(cer, 0.001) == expected_cer_model_impl


def test_calculate_mean_iou(ocr_model_instance, model_params): # Needs model instance for static method call
    # Example data for Mean IoU
    # bbox_preds: (N, T_model, 4)
    # bbox_targets: (N, S_max, 4)
    # target_lengths: (N)
    # t_model (model output seq len), s_max (GT padded len)
    
    # Batch size = 1, 2 boxes for the sample
    # T_model = 2 (model predicted 2 boxes)
    # S_max = 2 (GT also has 2 boxes, padded to this)
    
    bbox_preds = torch.tensor([[[0,0,10,10], [20,20,30,30]]], dtype=torch.float32)  # (1, 2, 4)
    bbox_targets = torch.tensor([[[0,0,10,10], [25,25,35,35]]], dtype=torch.float32) # (1, 2, 4)
    target_lengths = torch.tensor([2], dtype=torch.long) # Actual length is 2 boxes
    
    t_model = bbox_preds.size(1)
    s_max = bbox_targets.size(1)

    # Box 1: pred=[0,0,10,10], target=[0,0,10,10]. IoU = 1.0
    # Box 2: pred=[20,20,30,30], target=[25,25,35,35]
    # Intersection: x1=max(20,25)=25, y1=max(20,25)=25, x2=min(30,35)=30, y2=min(30,35)=30
    # Intersection area = (30-25) * (30-25) = 25
    # Union: Area1 + Area2 - Intersection = 100 + 100 - 25 = 175. IoU = 25/175 = 1/7 ≈ 0.1429
    # Mean IoU = (1.0 + 0.1429) / 2 ≈ 0.5714
    
    expected_mean_iou = (1.0 + 1/7) / 2  # ≈ 0.5714
    
    mean_iou = OCRModel.calculate_mean_iou(bbox_preds, bbox_targets, target_lengths, t_model, s_max)
    assert isinstance(mean_iou, float), "Mean IoU should be a float"
    assert pytest.approx(mean_iou) == expected_mean_iou

    # Test with different lengths
    bbox_preds2 = torch.tensor([[[0,0,10,10], [20,20,30,30], [0,0,1,1]]], dtype=torch.float32)  # (1, 3, 4) -> T_model=3
    bbox_targets2 = torch.tensor([[[0,0,10,10], [25,25,35,35]]], dtype=torch.float32) # (1, 2, 4) -> S_max=2
    target_lengths2 = torch.tensor([1], dtype=torch.long) # Only consider the first box
    
    t_model2 = bbox_preds2.size(1)
    s_max2 = bbox_targets2.size(1)
    
    # Only first box is compared due to target_lengths2 = [1].
    # Pred_box1=[0,0,10,10], Target_box1=[0,0,10,10]. IoU = 1.0
    # Mean IoU = 1.0 / 1 = 1.0
    expected_mean_iou2 = 1.0
    mean_iou2 = OCRModel.calculate_mean_iou(bbox_preds2, bbox_targets2, target_lengths2, t_model2, s_max2)
    assert pytest.approx(mean_iou2) == expected_mean_iou2

# Run with: pytest tests/test_ocr_model.py
# Ensure ocr_model.py is accessible in PYTHONPATH.
# Add to top of file if needed:
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from ocr_model import OCRModel
# (Already handled by placeholder comment)
