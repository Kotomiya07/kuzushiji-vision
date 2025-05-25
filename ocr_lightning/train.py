import torch
import pytorch_lightning as pl
import argparse
from pathlib import Path
import os

from ocr_lightning.dataset import OcrDataset, ocr_collate_fn
from ocr_lightning.model import OCRModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms # For potential future use

# Define Character Set (Placeholder)
# Extended slightly for more coverage, including a few Japanese characters from the prompt
VOCAB = '<blank>' + 'abcdefghijklmnopqrstuvwxyz0123456789' + '帝都書肆尚書堂梓 .,:;!?\'"`-()'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}

# Ensure <blank> is at index 0, as typically expected by CTCLoss.
# The model's __init__ also checks this, but good to be explicit.
if CHAR_TO_IDX.get('<blank>') != 0:
    # Attempt to re-create CHAR_TO_IDX if <blank> is not first, assuming VOCAB was defined with <blank> first.
    if VOCAB.startswith('<blank>'):
        print("Warning: CHAR_TO_IDX was not correctly created with <blank> at index 0. Re-creating.")
        CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
        IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}
        if CHAR_TO_IDX['<blank>'] != 0:
            raise ValueError("Critical: Failed to set '<blank>' to index 0. Check VOCAB definition.")
    else:
        raise ValueError("'<blank>' character must be at index 0 in VOCAB for CTCLoss and should be the first element in VOCAB string.")

BLANK_CHAR_IDX = CHAR_TO_IDX['<blank>']


def main(args):
    pl.seed_everything(args.seed, workers=True) # Added workers=True for better reproducibility

    # image_transforms is None, OcrDataset will apply ToTensor() by default.
    image_transforms = None 
    # Example for future custom transforms:
    # image_transforms = transforms.Compose([
    #     transforms.Resize((desired_h, desired_w)), # Example resize
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    # ])

    # Create DataLoaders
    train_dataset = OcrDataset(
        data_split_dir=Path(args.train_data_dir), 
        char_to_idx=CHAR_TO_IDX, 
        image_transforms=image_transforms
    )
    val_dataset = OcrDataset(
        data_split_dir=Path(args.val_data_dir), 
        char_to_idx=CHAR_TO_IDX, 
        image_transforms=image_transforms
    )

    if len(train_dataset) == 0:
        print(f"Error: Training dataset at {args.train_data_dir} is empty. Please check the path and data structure.")
        return
    if len(val_dataset) == 0:
        print(f"Warning: Validation dataset at {args.val_data_dir} is empty. Training will proceed without validation.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=ocr_collate_fn, 
        num_workers=args.num_workers, 
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False # For speedup with num_workers > 0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=ocr_collate_fn, 
        num_workers=args.num_workers, 
        shuffle=False,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    ) if len(val_dataset) > 0 else None


    model = OCRModel(
        char_to_idx=CHAR_TO_IDX, 
        idx_to_char=IDX_TO_CHAR, 
        learning_rate=args.learning_rate, 
        max_boxes=args.max_boxes
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir), 
        filename='ocr_model-{epoch:02d}-{val/total_loss:.2f}', 
        save_top_k=1, 
        monitor='val/total_loss', 
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val/total_loss', 
        patience=args.patience, 
        verbose=True, 
        mode='min'
    )
    
    callbacks_to_use = []
    if val_loader: # Only add callbacks if there is a validation set
        callbacks_to_use.extend([checkpoint_callback, early_stop_callback])


    tensorboard_logger = TensorBoardLogger(
        save_dir=Path(args.log_dir), 
        name="ocr_logs"
    )
    
    # Handle devices argument for Trainer
    if args.accelerator in ['gpu', 'tpu', 'mps'] and args.devices and args.devices != 'auto':
        if args.devices.isdigit():
            devices_param = int(args.devices)
        elif ',' in args.devices:
            try:
                devices_param = [int(d.strip()) for d in args.devices.split(',')]
            except ValueError:
                print(f"Warning: Could not parse devices '{args.devices}' as list of ints. Using 'auto'.")
                devices_param = 'auto'
        else: # Specific device string like 'cuda:0' or 'auto'
            devices_param = args.devices
    else: # CPU or auto without specific devices
        devices_param = 'auto' if args.accelerator != 'cpu' else None


    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        accelerator=args.accelerator, 
        devices=devices_param,
        callbacks=callbacks_to_use if callbacks_to_use else None, 
        logger=tensorboard_logger, 
        deterministic=True if args.seed is not None else False,
        # log_every_n_steps=max(1, len(train_loader) // 10) # Example: Log 10 times per epoch
    )

    print(f"Starting training with args: {args}")
    print(f"Vocabulary size: {len(VOCAB)}, Blank char index: {BLANK_CHAR_IDX}")
    print(f"Train dataset size: {len(train_dataset)}")
    if val_loader:
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("No validation loader configured.")
    
    trainer.fit(model, train_loader, val_loader if val_loader else None)

    print(f"Training finished.")
    if val_loader and checkpoint_callback.best_model_path:
        print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")
    elif val_loader:
        print("No best model checkpoint was saved (perhaps training was too short, val_total_loss did not improve, or an error occurred).")
    else:
        print("No validation loader, so no checkpoint monitoring based on validation loss.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train OCR Model")
    
    parser.add_argument("--train_data_dir", type=str, required=True, help="Path to training data directory (e.g., data/train).")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Path to validation data directory (e.g., data/val).")
    parser.add_argument("--checkpoint_dir", type=str, default="./ocr_checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--log_dir", type=str, default="./ocr_logs", help="Directory for TensorBoard logs.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Set to None for no explicit seeding.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of workers for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--max_boxes", type=int, default=50, help="Max number of bounding boxes per image (model config).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    
    parser.add_argument("--accelerator", type=str, default='auto', choices=['cpu', 'gpu', 'tpu', 'mps', 'auto'], help="Accelerator ('cpu', 'gpu', 'tpu', 'mps', 'auto').")
    parser.add_argument("--devices", type=str, default='auto', help="Devices: 'auto', an int (num gpus), or comma-separated list like '0,1'.")

    args = parser.parse_args()

    if args.seed is None: # Allow disabling seed for non-deterministic runs
        args.seed = torch.Generator().seed() # Get a random seed if None
        print(f"No seed provided, using randomly generated seed: {args.seed}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    main(args)
