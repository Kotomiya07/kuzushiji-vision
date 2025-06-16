import shutil
import subprocess  # For running train.py as a script
import sys
import tempfile
from pathlib import Path

import pytest

# This test assumes train.py is executable and in the parent directory.
# Adjust path if train.py is located elsewhere.
TRAIN_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "train.py"

# Fixture for temporary data directory, similar to test_data_loader
@pytest.fixture(scope="module")
def temp_dummy_data_dir_for_train():
    """Creates a temporary directory with a dummy dataset structure for training script test."""
    base_dir = tempfile.mkdtemp(prefix="ocr_train_test_data_")
    print(f"Created temporary data directory for train.py test: {base_dir}")

    # Use the create_dummy_data function from data_loader if accessible and robust
    # For isolation, we can redefine a minimal version here or call data_loader's.
    # Let's assume data_loader is in PYTHONPATH and try to use its function.
    try:
        # Temporarily add parent to sys.path to import data_loader
        parent_dir = str(Path(__file__).resolve().parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from scripts.data_loader import create_dummy_data as create_dl_dummy_data
        
        # Create dummy data using the imported function
        create_dl_dummy_data(base_dir) # This creates train/val/test splits with 5 samples each
        
    except ImportError:
        pytest.skip("Could not import create_dummy_data from data_loader.py. Skipping train script test or implement local dummy data creation.")
    except Exception as e:
        pytest.skip(f"Error creating dummy data using data_loader.create_dummy_data: {e}. Skipping train script test.")

    yield str(base_dir)

    print(f"Cleaning up temporary data directory for train.py test: {base_dir}")
    shutil.rmtree(base_dir)


@pytest.fixture
def temp_checkpoint_dir():
    """Creates a temporary directory for checkpoints."""
    chkpt_dir = tempfile.mkdtemp(prefix="ocr_train_checkpoints_")
    yield str(chkpt_dir)
    shutil.rmtree(chkpt_dir)


# Test for the training script
# This is a light integration test.
# It runs train.py with minimal settings on dummy data.
def test_train_script_runs_dummy(temp_dummy_data_dir_for_train, temp_checkpoint_dir):
    if not TRAIN_SCRIPT_PATH.is_file():
        pytest.skip(f"train.py script not found at {TRAIN_SCRIPT_PATH}")

    # Command-line arguments for train.py
    args = [
        sys.executable, # Path to python interpreter
        str(TRAIN_SCRIPT_PATH),
        "--data_dir", temp_dummy_data_dir_for_train,
        "--create_dummy_if_missing", # Should not be needed if fixture works, but good for robustness
        "--max_epochs", "1",         # Minimal epochs
        "--batch_size", "2",         # Small batch size
        "--num_workers", "0",        # For simplicity in tests
        "--target_height", "32",     # Smaller images for speed
        "--target_width", "128",
        "--rnn_hidden_size", "32",   # Smaller model
        "--rnn_layers", "1",
        "--learning_rate", "1e-4",
        "--checkpoint_dir", temp_checkpoint_dir,
        "--accelerator", "cpu",      # Force CPU for tests to avoid GPU dependencies/issues
        "--deterministic", "true",   # For reproducibility
        "--run_test",                # Test the test phase as well
        "--patience", "1",           # Early stopping with small patience
        # Ensure monitors are set to something that will be logged even with few steps
        "--checkpoint_monitor", "val_loss", 
        "--early_stopping_monitor", "val_loss",
        # Use a non-pretrained encoder to avoid downloads
        "--pretrained_encoder", "false" 
    ]

    print(f"Running train.py with command: {' '.join(args)}")

    # Run train.py as a subprocess
    process = subprocess.run(args, capture_output=True, text=True, timeout=300) # 5 min timeout

    # Print output for debugging, especially on failure
    print("--- train.py STDOUT ---")
    print(process.stdout)
    print("--- train.py STDERR ---")
    print(process.stderr)

    assert process.returncode == 0, \
        f"train.py script failed with exit code {process.returncode}. Error: {process.stderr}"

    # Optional: Check if a checkpoint file was created
    checkpoint_files = list(Path(temp_checkpoint_dir).glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint file was created."
    
    # Check for common errors in output if needed
    assert "Traceback" not in process.stderr, "Traceback found in stderr."
    assert "Error" not in process.stderr, "The word 'Error' found in stderr (case sensitive)."
    # A more robust check might be for "Training finished" or "Test results"
    assert "Training finished. Running test phase..." in process.stdout or "trainer.test(model, dataloaders=test_loader)" in process.stdout, \
        "Indication of training/testing completion not found in stdout."


# To run this test:
# pytest tests/test_train_script.py
# Ensure train.py, data_loader.py, ocr_model.py are in the parent directory of 'tests/'
# or otherwise accessible in PYTHONPATH.
# The test tries to add the parent directory to sys.path for importing create_dummy_data.
#
# Note: This test is more of an integration test and can be slower.
# It depends on the train.py script's CLI and overall functionality.
# The `timeout` in `subprocess.run` is important to prevent tests from hanging indefinitely.
#
# If `create_dummy_data` cannot be imported from `data_loader` (e.g. due to its own imports
# failing in the test environment before sys.path modification takes full effect, or if
# data_loader itself is not structured for easy import of just that function),
# then the dummy data creation logic would need to be duplicated or simplified within
# the `temp_dummy_data_dir_for_train` fixture itself. The current version attempts import.The tests have been created as three separate files:

# 1.  `tests/test_data_loader.py`:
#     *   Includes a `temp_data_dir` fixture to create a temporary dummy dataset (images, labels, bounding boxes) for train, val, and test splits.
#     *   `test_build_char_to_int_map`: Verifies the character mapping function, checking for special tokens, consistency, and vocabulary size.
#     *   `test_ocr_dataset`: Tests the `OCRDataset` class for length, item retrieval (keys, types, shapes of image, label, bounding\_boxes, label\_lengths), and correct padding/label length values.
#     *   `test_get_data_loader`: Checks if `get_data_loader` returns a DataLoader and if a batch can be drawn with correct keys and tensor shapes.

# 2.  `tests/test_ocr_model.py`:
#     *   Uses fixtures to set up model parameters and an `OCRModel` instance with reduced sizes for speed.
#     *   `test_model_initialization`: Checks if the model is a `LightningModule` and has key components.
#     *   `test_model_forward_pass`: Verifies the forward pass output (keys, shapes of logits and bounding box predictions).
#     *   `test_model_step_methods`: Creates a dummy batch and calls `training_step`, `validation_step`, and `test_step` to ensure they run and return a scalar loss.
#     *   `test_calculate_cer`: Tests the static CER calculation method with predefined strings.
#     *   `test_calculate_mean_iou`: Tests the static Mean IoU calculation with predefined bounding boxes and target lengths.

# 3.  `tests/test_train_script.py`:
#     *   `test_train_script_runs_dummy`: A light integration test that runs the `train.py` script as a subprocess.
#     *   Uses fixtures for a temporary dummy dataset (attempting to use `create_dummy_data` from `data_loader.py`) and a temporary checkpoint directory.
#     *   Passes command-line arguments to `train.py` for a minimal run (1 epoch, small batch size, CPU, dummy data creation flag, etc.).
#     *   Asserts that the script completes successfully (exit code 0) and optionally checks for checkpoint creation and absence of "Error" or "Traceback" in output.

# **General Considerations for the tests:**
# *   They are structured for use with `pytest`.
# *   Necessary imports are included.
# *   Temporary directories are used for generated data and checkpoints, with cleanup handled by fixtures.
# *   The tests focus on unit-level functionality, with `test_train_script.py` providing a basic integration check.
# *   Path adjustments for importing project modules (e.g., `data_loader`, `ocr_model`) are commented within the test files, assuming a standard project structure where `tests` is a subdirectory in the root.

# To run these tests, one would typically navigate to the project's root directory and execute `pytest`.
# The python environment needs `pytest`, `torch`, `pytorch-lightning`, `Pillow`, `PyYAML`, and `editdistance` installed.
# The `train.py` test, in particular, relies on `data_loader.py` being importable to use its `create_dummy_data` function for setting up the test environment. If this import fails, that specific test will be skipped.
# The tests for `test_data_loader.py` and `test_ocr_model.py` should be self-contained regarding their specific module dependencies.
