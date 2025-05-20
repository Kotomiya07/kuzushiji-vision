import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
import os
import pandas as pd # For loading chars from CSV

# Project specific imports
from src.models.ocr_lightning import LitOCRModel
from src.data.oneline_dataset import OneLineOCRDataset
from src.utils.tokenizer import Vocab # Import Vocab class, specific IDs will come from instance


def setup_parser():
    parser = argparse.ArgumentParser(description="Train One-Line OCR Model")

    # Data arguments
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory containing train, val, test split folders (each with BookID subfolders).")
    parser.add_argument("--train_dir_name", type=str, default="train", help="Name of the training data folder under data_root_dir.")
    parser.add_argument("--val_dir_name", type=str, default="val", help="Name of the validation data folder under data_root_dir.")
    parser.add_argument("--test_dir_name", type=str, default=None, help="Optional: Name of the test data folder under data_root_dir.")
    parser.add_argument("--unicode_csv_path", type=str, required=True, help="Path to unicode_translation.csv for Vocab.")

    # Image arguments
    parser.add_argument("--image_height", type=int, default=64, help="Target image height.")
    parser.add_argument("--image_width", type=int, default=1024, help="Target image width.")
    parser.add_argument("--image_channels", type=int, default=1, choices=[1, 3], help="Number of image channels (1 for grayscale, 3 for RGB).")

    # Label and Tokenizer arguments
    parser.add_argument("--max_label_len", type=int, default=256, help="Maximum length of a label sequence.")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "auto"], help="Accelerator to use.")
    parser.add_argument("--devices", type=str, default="1", help="Number of devices to use (e.g., 1 for single GPU, '0,1' for two GPUs, or number for multi-GPU).")
    
    # Model specific arguments
    parser.add_argument("--bbox_loss_weight", type=float, default=0.05, help="Weight for the bounding box loss component.")
    
    # Encoder specific arguments (from UNetTransformerEncoder)
    parser.add_argument("--encoder_in_channels", type=int, default=1, help="Input channels for encoder (should match image_channels).")
    parser.add_argument("--encoder_initial_filters", type=int, default=64, help="Initial filters in the U-Net encoder.")
    parser.add_argument("--encoder_num_unet_layers", type=int, default=4, help="Number of U-Net downsampling layers in encoder.")
    parser.add_argument("--encoder_num_transformer_layers", type=int, default=4, help="Number of Transformer layers in encoder.")
    parser.add_argument("--encoder_transformer_heads", type=int, default=8, help="Number of attention heads in encoder's Transformer.")
    parser.add_argument("--encoder_transformer_mlp_dim", type=int, default=2048, help="Dimension of the MLP in encoder's Transformer.")

    # Decoder specific arguments (HuggingFace based)
    parser.add_argument("--decoder_model_name", type=str, default="KoichiYasuoka/roberta-small-japanese-aozora-char", help="Path or name of the HuggingFace decoder model.")
    parser.add_argument("--max_gen_len", type=int, default=50, help="Max length for sequence generation in decoder.")


    # Logging and Callbacks
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name. If None, TensorBoardLogger is used.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name.")
    parser.add_argument("--patience_early_stopping", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--checkpoint_monitor_metric", type=str, default="val_cer", help="Metric to monitor for ModelCheckpoint (e.g., val_loss, val_cer, val_iou).")
    parser.add_argument("--checkpoint_mode", type=str, default="min", choices=["min", "max"], help="Mode for ModelCheckpoint ('min' for loss/error, 'max' for accuracy/iou).")


    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser


def main(args):
    pl.seed_everything(args.seed)

    # Initialize Tokenizer directly from CSV path
    try:
        tokenizer = Vocab(unicode_csv_path=args.unicode_csv_path)
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        return # Exit if tokenizer fails

    # Datasets
    train_data_path = os.path.join(args.data_root_dir, args.train_dir_name)
    val_data_path = os.path.join(args.data_root_dir, args.val_dir_name)

    train_dataset = OneLineOCRDataset(
        data_root_dir=train_data_path,
        tokenizer=tokenizer,
        max_label_len=args.max_label_len,
        image_height=args.image_height,
        image_width=args.image_width,
        image_channels=args.image_channels, # Pass image_channels
        # transform can be None to use default, or pass custom transforms
    )
    val_dataset = OneLineOCRDataset(
        data_root_dir=val_data_path,
        tokenizer=tokenizer,
        max_label_len=args.max_label_len,
        image_height=args.image_height,
        image_width=args.image_width,
        image_channels=args.image_channels, # Pass image_channels
    )
    
    test_dataset = None
    if args.test_dir_name:
        test_data_path = os.path.join(args.data_root_dir, args.test_dir_name)
        if os.path.exists(test_data_path):
            test_dataset = OneLineOCRDataset(
                data_root_dir=test_data_path,
                tokenizer=tokenizer,
                max_label_len=args.max_label_len,
                image_height=args.image_height,
                image_width=args.image_width,
                image_channels=args.image_channels,
            )
        else:
            print(f"Warning: Test directory {test_data_path} not found. Skipping test set.")


    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True, # Added for potential speedup if using GPU
        persistent_workers=True if args.num_workers > 0 else False # Added for efficiency
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_dataloader = None
    if test_dataset:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )

    # Model Configuration
    # LitOCRModel now expects the tokenizer instance directly.
    # Model_config will not contain 'chars', 'vocab_size', or specific token IDs like 'pad_id',
    # as these are now handled internally by LitOCRModel via the tokenizer instance.
    model_config = {
        "decoder_path": args.decoder_model_name,
        "max_gen_len": args.max_gen_len,
        "bbox_loss_weight": args.bbox_loss_weight,
        # Encoder parameters
        "encoder_in_channels": args.image_channels, 
        "encoder_initial_filters": args.encoder_initial_filters,
        "encoder_num_unet_layers": args.encoder_num_unet_layers,
        "encoder_num_transformer_layers": args.encoder_num_transformer_layers,
        "encoder_transformer_heads": args.encoder_transformer_heads,
        "encoder_transformer_mlp_dim": args.encoder_transformer_mlp_dim,
    }
    optimizer_config = {
        "lr": args.learning_rate,
        # "weight_decay": args.weight_decay (if added to parser)
    }

    # Pass the initialized tokenizer to LitOCRModel
    lit_model = LitOCRModel(model_config, optimizer_config, tokenizer=tokenizer)

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor=args.checkpoint_monitor_metric,
        mode=args.checkpoint_mode,
        save_top_k=1,
        save_last=True, # Good for resuming
        filename='oneline-ocr-{epoch:02d}-{' + args.checkpoint_monitor_metric + ':.4f}'
    )
    callbacks.append(checkpoint_callback)

    if args.patience_early_stopping > 0: # Allow disabling early stopping if patience is 0 or less
        early_stopping_callback = EarlyStopping(
            monitor=args.checkpoint_monitor_metric, # Typically same as checkpoint metric
            patience=args.patience_early_stopping,
            mode=args.checkpoint_mode
        )
        callbacks.append(early_stopping_callback)
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor_callback)

    # Logger
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity, # Optional: WandB entity (team)
            name=f"oneline-ocr-run-{args.seed}", # Optional: Run name
            config=vars(args) # Log all hyperparameters
        )
        logger.watch(lit_model, log='all', log_freq=100) # Watch model gradients
    else:
        logger = TensorBoardLogger("tb_logs", name="oneline-ocr-model")

    # Trainer
    # Parse devices argument
    if args.accelerator == 'gpu' or args.accelerator == 'auto' and torch.cuda.is_available():
        try:
            # Attempt to convert to list of ints if comma-separated, else int
            if ',' in args.devices:
                devices_list = [int(d.strip()) for d in args.devices.split(',')]
                actual_devices = devices_list
            else:
                actual_devices = int(args.devices)
        except ValueError:
            print(f"Warning: Could not parse devices '{args.devices}'. Assuming 1 device if available.")
            actual_devices = 1
    else: # CPU or other accelerators
        actual_devices = args.devices # Keep as string like "1" or "auto" for PTL to handle

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=actual_devices,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        # precision=16 if args.mixed_precision else 32, # Example for mixed precision
        # deterministic=True, # For full reproducibility, might impact performance
        # gradient_clip_val=args.gradient_clip_val, # If using gradient clipping
    )

    # Training
    print("Starting training...")
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("Training finished.")

    # Testing
    if test_dataloader:
        print("Starting testing...")
        # Load best model for testing
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            trainer.test(model=lit_model, dataloaders=test_dataloader, ckpt_path=best_model_path)
        else:
            print("No best model checkpoint found, or path is invalid. Testing with last model state.")
            trainer.test(model=lit_model, dataloaders=test_dataloader) # Test with the model state at end of training
        print("Testing finished.")
    else:
        print("No test dataset specified or found. Skipping testing.")

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    
    # Ensure encoder_in_channels matches image_channels from args
    # This is a common point of error if not synced.
    # The model_config now directly uses args.image_channels for encoder_in_channels.
    # So this check is implicitly handled by using args.image_channels in model_config.
    # If args.encoder_in_channels was used in model_config, then a check like:
    # if args.encoder_in_channels != args.image_channels:
    #    raise ValueError("encoder_in_channels must match image_channels")
        
    main(args)
