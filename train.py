import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Assuming data_loader.py and ocr_model.py are in the same directory or accessible in PYTHONPATH
from scripts.data_loader import (
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_TARGET_WIDTH,
    MAX_LABEL_LENGTH,
    build_char_to_int_map,
    get_data_loader,
)
from scripts.ocr_model import OCRModel


def load_config(config_path):
    """Loads a YAML configuration file."""
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.is_file():
        print(f"Warning: Config file not found at {config_path}. Using defaults/CLI args.")
        return {}
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
            return config if config else {}
        except yaml.YAMLError as e:
            print(f"Error loading YAML config: {e}")
            return {}

def run_training(args):
    """Initializes and runs the OCR model training process."""

    # --- Configuration Priority: Defaults -> YAML Config -> CLI Args ---
    # Start with defaults (from argparse)
    config = vars(args) # Convert argparse Namespace to dict

    # Load YAML config and update defaults
    yaml_config = load_config(args.config_path)
    config.update(yaml_config) # YAML overrides defaults

    # Re-apply CLI arguments to override YAML values if they were explicitly set
    # (argparse defaults are overridden by YAML, but explicit CLI args should have highest priority)
    for key, value in vars(args).items():
        # Check if the CLI argument was set (not its default value)
        # This is a bit tricky as argparse doesn't directly tell us this.
        # A common way is to compare with a fresh parse of defaults.
        # For simplicity here, we'll assume CLI args passed override YAML if different from argparse default.
        # Or, more simply, just re-update from args, effectively making CLI override YAML.
        if getattr(args, key) != parser.get_default(key) or key not in yaml_config:
             config[key] = value
    
    # Use a new Namespace or dict for merged config
    # For simplicity, we'll just use the 'config' dict and access items via config['key']
    # Or convert it back to a Namespace if preferred by the rest of the code
    args = argparse.Namespace(**config)

    # --- Data Setup ---
    data_dir = Path(args.data_dir)

    # Check for dummy data generation
    # This condition checks if the specific 'train/images' subdirectory is missing,
    # which is a strong indicator that the dataset structure isn't there.
    train_images_dir = data_dir / "train" / "images"
    if not data_dir.exists() or not train_images_dir.exists():
        print(f"Data directory {data_dir} or its subdirectories not found or empty. Please check the --data_dir path.")
        exit(1)

    print(f"Using data directory: {data_dir}")
    
    char_to_int, int_to_char = build_char_to_int_map(str(data_dir))
    num_chars = len(char_to_int)
    print(f"Vocabulary size: {num_chars} characters.")
    if num_chars <= 2: # Only <PAD> and <UNK>
        print("Warning: Vocabulary is very small. Check data and label files.")
        if args.create_dummy_if_missing: # If dummy data was used, it might be empty or failed
             print("This might be due to dummy data being used and not containing diverse characters.")
             print("If you intended to use real data, please check the --data_dir path.")
        # Potentially exit or use a fallback vocab if this is critical
        # For now, we'll proceed, but training might not be meaningful.

    # DataLoaders
    # Using args for parameters that can be configured
    train_loader = get_data_loader(
        data_dir=str(data_dir),
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        char_to_int=char_to_int,
        max_label_length=args.max_label_length, # Use arg for this
        target_height=args.target_height,
        target_width=args.target_width,
        shuffle=True
    )
    val_loader = get_data_loader(
        data_dir=str(data_dir),
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        char_to_int=char_to_int,
        max_label_length=args.max_label_length, # Use arg for this
        target_height=args.target_height,
        target_width=args.target_width,
        shuffle=False
    )

    # --- Model Initialization ---
    # Ensure all required hparams for OCRModel are taken from 'args' Namespace
    model = OCRModel(
        num_chars=num_chars,
        int_to_char=int_to_char, # For decoding/CER
        max_label_length=args.max_label_length, # Passed to model's hparams
        input_height=args.target_height, # Renamed for clarity (target_height is loader, input_height is model)
        input_width=args.target_width,   # Renamed for clarity
        input_channels=3, # Standard for RGB images
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers, # Added from args
        encoder_name=args.encoder_name, # Added from args
        pretrained_encoder=args.pretrained_encoder, # Added from args
        learning_rate=args.learning_rate,
        lambda_bbox=args.lambda_bbox # Added from args
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor=args.checkpoint_monitor,
        dirpath=args.checkpoint_dir,
        filename="ocr-best-{epoch}-{val_loss:.2f}-{val_cer:.2f}",
        save_top_k=1,
        mode="min" if args.checkpoint_monitor == "val_loss" or args.checkpoint_monitor == "val_cer" else "max",
        verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor=args.early_stopping_monitor,
        patience=args.patience,
        mode="min" if args.early_stopping_monitor == "val_loss" or args.early_stopping_monitor == "val_cer" else "max",
        verbose=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor_callback]

    # --- Trainer Initialization ---
    # Determine accelerator and devices
    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator
    
    devices = args.devices if args.devices else 1 if accelerator == "gpu" else None
    if accelerator == "cpu" and args.devices and args.devices > 1:
        print(f"Warning: Accelerator is CPU, but devices is set to {args.devices}. Using CPU with a single process.")
        devices = None # Or 1, but None is typical for pl CPU single process

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=True, # Uses TensorBoardLogger by default
        precision=args.precision if accelerator == "gpu" else 32, # Mixed precision only on GPU
        deterministic=args.deterministic,
        # limit_train_batches=0.1, # For quick testing, uncomment
        # limit_val_batches=0.1,   # For quick testing, uncomment
        log_every_n_steps=args.log_every_n_steps
    )

    # --- Training ---
    print(f"Starting training with: Epochs={args.max_epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}")
    print(f"Using {accelerator} with {devices if devices else 1} device(s).")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --- Testing (Optional) ---
    if args.run_test:
        print("Training finished. Running test phase...")
        test_loader = get_data_loader(
            data_dir=str(data_dir),
            split="test", # Assuming a 'test' split exists
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            char_to_int=char_to_int,
            max_label_length=args.max_label_length,
            target_height=args.target_height,
            target_width=args.target_width,
            shuffle=False
        )
        # Load the best checkpoint for testing
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and Path(best_model_path).exists():
            print(f"Loading best model from: {best_model_path}")
            trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
        else:
            print("No best model checkpoint found, or path is invalid. Testing with last model state.")
            trainer.test(model, dataloaders=test_loader) # Test with current model state
    
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR Model using PyTorch Lightning.")

    # Paths and Config
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--data_dir", type=str, default="data/column_dataset_padded", help="Directory containing the dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/ocr_model", help="Directory to save model checkpoints.")
    
    # Data Parameters
    parser.add_argument("--target_height", type=int, default=DEFAULT_TARGET_HEIGHT, help="Target height for images.")
    parser.add_argument("--target_width", type=int, default=DEFAULT_TARGET_WIDTH, help="Target width for images.")
    parser.add_argument("--max_label_length", type=int, default=MAX_LABEL_LENGTH, help="Maximum length of a label.")
    parser.add_argument("--create_dummy_if_missing", action='store_true', help="Create dummy data if data_dir is not found/empty.")

    # Model Hyperparameters
    parser.add_argument("--encoder_name", type=str, default="resnet50", choices=["resnet34", "resnet50"], help="Encoder backbone.")
    parser.add_argument("--pretrained_encoder", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use pretrained encoder weights.")
    parser.add_argument("--rnn_hidden_size", type=int, default=256, help="Hidden size for RNN decoder.")
    parser.add_argument("--rnn_layers", type=int, default=2, help="Number of layers for RNN decoder.")
    parser.add_argument("--lambda_bbox", type=float, default=1.0, help="Weight for bounding box loss.")

    # Training Parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader. Set to 0 for main process debugging.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "mps"], help="Accelerator to use ('auto', 'cpu', 'gpu', 'mps').")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices to use (e.g., GPUs). None for auto.")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32], help="Training precision (16 for mixed, 32 for full).")
    parser.add_argument("--deterministic", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable deterministic mode for reproducibility.")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log training information every N steps.")

    # Callback Monitors
    parser.add_argument("--checkpoint_monitor", type=str, default="val_loss", help="Metric to monitor for ModelCheckpoint.")
    parser.add_argument("--early_stopping_monitor", type=str, default="val_loss", help="Metric to monitor for EarlyStopping.")

    # Testing
    parser.add_argument("--run_test", action='store_true', help="Run testing phase after training using the best checkpoint.")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run_training(args)
