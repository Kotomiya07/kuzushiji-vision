import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb  # Import wandb
import yaml
from timm.scheduler import CosineLRScheduler  # Assuming timm is installed
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project specific imports
from models.character_detection import CharacterDetectionModel
from utils.dataset import CharacterDetectionDataset

# from utils.augmentation import get_character_detection_transforms # No longer needed, handled in dataset
from utils.metrics import compute_character_accuracy, compute_map
from utils.util import EasyDict, recursive_to_dict


# --- Configuration ---
def load_config(config_path: str) -> EasyDict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    # Convert nested dictionaries to EasyDict
    return EasyDict(config_dict)


def setup_experiment(config: EasyDict) -> Path:
    """Set up experiment directory and random seeds."""
    # Set random seeds
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create experiment directory
    timestamp = datetime.datetime.now().strftime(config.experiment.save_dir.split("/")[-1])  # Use format from config
    save_dir_base = os.path.dirname(config.experiment.save_dir)
    exp_dir = Path(save_dir_base) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / config.experiment.log_dir).mkdir(exist_ok=True)
    (exp_dir / config.experiment.checkpoint_dir).mkdir(exist_ok=True)
    (exp_dir / config.experiment.eval_dir).mkdir(exist_ok=True)

    # Save config to experiment directory
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(recursive_to_dict(config), f, default_flow_style=False)  # Save config as standard dict

    return exp_dir


# --- Custom Collate Function ---
def collate_fn(batch: list[dict | None]) -> dict | None:
    """
    Custom collate function to handle None items and pad images.
    Args:
        batch (List[Union[Dict, None]]): A list of samples from the dataset.
                                         Each sample is a dict or None if an error occurred.
    Returns:
        Dict or None: A dictionary containing batched data (images, boxes, labels, etc.)
                      or None if the batch is empty after filtering.
    """
    # Filter out None items (samples that failed to load/process)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty after filtering

    images = [item["image"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    image_paths = [item["image_path"] for item in batch]  # Collect image paths

    # Find the maximum height in the batch
    max_h = max(img.shape[1] for img in images)

    # Pad images to the maximum height
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        # Pad height (bottom) and potentially width (right) if needed, though width should be fixed
        # Using F.pad: (padding_left, padding_right, padding_top, padding_bottom)
        padded_img = F.pad(img, (0, 0, 0, pad_h), mode="constant", value=0)
        padded_images.append(padded_img)

    # Stack images into a batch tensor
    images_tensor = torch.stack(padded_images, dim=0)

    return EasyDict(
        {
            "images": images_tensor,
            "boxes": boxes,  # Keep as list of tensors
            "labels": labels,  # Keep as list of tensors
            "image_ids": image_ids,
            "image_paths": image_paths,  # Include image paths in the batch dict
        }
    )


# --- Training and Evaluation Functions ---
def train_one_epoch(
    model: CharacterDetectionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: EasyDict,
    wandb_run,
) -> float:
    """Train the model for one epoch."""
    model.train()
    model.set_epoch(epoch)  # Set epoch for potential dynamic adjustments (like IoU threshold)
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:  # Skip empty batches from collate_fn
            print(f"Warning: Skipping empty batch {batch_idx} in training.")
            continue

        images = batch.images.to(device)
        # Prepare targets in the expected format (list of dicts or dict of lists)
        # Model expects targets: dict[str, list[torch.Tensor]]
        targets = {"boxes": [box.to(device) for box in batch.boxes], "labels": [label.to(device) for label in batch.labels]}

        optimizer.zero_grad()
        output = model(images, targets)  # Pass targets for loss calculation
        loss = output["loss"]
        det_loss = output["detection_loss"]
        cls_loss = output["classification_loss"]

        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping batch.")
            print(f"  Image IDs: {batch.image_ids}")
            print(f"  Image Paths: {batch.image_paths}")  # Log paths for problematic images
            # Optionally save problematic batch data for debugging
            # torch.save(batch, f"nan_batch_{batch_idx}.pt")
            continue  # Skip backpropagation for this batch

        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", det=f"{det_loss:.4f}", cls=f"{cls_loss:.4f}")

        # Log batch loss to WandB
        if wandb_run:
            wandb_run.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_detection_loss": det_loss.item(),
                    "train/batch_classification_loss": cls_loss.item(),
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                }
            )

    return avg_loss


@torch.no_grad()
def evaluate(
    model: CharacterDetectionModel, dataloader: DataLoader, device: torch.device, config: EasyDict, epoch: int, wandb_run
) -> dict:
    """Evaluate the model on the validation set."""
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
    for batch in pbar:
        if batch is None:
            print("Warning: Skipping empty batch in validation.")
            continue

        images = batch.images.to(device)
        # Ground truth needed for metrics calculation
        gt_boxes = [b.to(device) for b in batch.boxes]
        gt_labels = [l.to(device) for l in batch.labels]

        output = model(images)  # No targets needed for inference

        # Collect predictions and ground truths for the entire validation set
        all_pred_boxes.extend([b.cpu() for b in output["boxes"]])
        all_pred_scores.extend([s.cpu() for s in output["scores"]])
        all_pred_labels.extend([l.cpu() for l in output["labels"]])
        all_gt_boxes.extend([b.cpu() for b in gt_boxes])
        all_gt_labels.extend([l.cpu() for l in gt_labels])

    # Compute metrics
    mAP, class_aps = compute_map(
        pred_boxes=all_pred_boxes,
        pred_scores=all_pred_scores,
        pred_labels=all_pred_labels,
        gt_boxes=all_gt_boxes,
        gt_labels=all_gt_labels,
        iou_threshold=config.evaluation.iou_threshold,
    )

    accuracy, class_accuracies = compute_character_accuracy(
        pred_labels=all_pred_labels,
        gt_labels=all_gt_labels,
        pred_boxes=all_pred_boxes,
        gt_boxes=all_gt_boxes,
        iou_threshold=config.evaluation.iou_threshold,
    )

    metrics = {
        "val/mAP": mAP,
        "val/accuracy": accuracy,
        # "val/class_aps": class_aps, # Can be large, log separately if needed
        # "val/class_accuracies": class_accuracies # Can be large
    }

    # Log metrics to WandB
    if wandb_run:
        wandb_log = metrics.copy()
        wandb_log["epoch"] = epoch + 1
        # Log class-wise APs and Accuracies separately if needed
        # for class_id, ap in class_aps.items():
        #     wandb_log[f"val/AP_class_{class_id}"] = ap
        # for class_id, acc in class_accuracies.items():
        #     wandb_log[f"val/Acc_class_{class_id}"] = acc
        wandb_run.log(wandb_log)

    return metrics


# --- Main Function ---
def main(config_path: str):
    """Main training loop."""
    config = load_config(config_path)
    exp_dir = setup_experiment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment directory: {exp_dir}")

    # --- WandB Setup ---
    wandb_run = None
    if config.wandb.project:
        try:
            wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,  # Optional: Wandb entity (username or team name)
                config=recursive_to_dict(config),  # Log config as standard dict
                name=exp_dir.name,  # Use timestamp name for the run
                dir=str(exp_dir / config.experiment.log_dir),  # Save wandb logs within experiment dir
            )
            print("WandB initialized.")
        except Exception as e:
            print(f"WandB initialization failed: {e}. Proceeding without WandB.")
            wandb_run = None
    else:
        print("WandB project not specified in config. Proceeding without WandB.")

    # --- Data Preparation ---
    print("Preparing data...")
    # Note: Transforms from config are applied within the dataset __getitem__
    # We might need separate transforms for train/val if config structure changes
    # Transforms are now handled inside the Dataset class based on config and is_train flag
    # train_transform = get_character_detection_train_transforms(config) # No longer needed here
    # val_transform = get_character_detection_val_transforms(config) # No longer needed here

    # Use image directories directly from config and pass config/is_train to Dataset
    train_dataset = CharacterDetectionDataset(
        annotation_file=config.data.train_annotation,
        image_base_dir=config.data.train_image_dir,
        unicode_map_file=config.data.unicode_dict,
        config=config,  # Pass the whole config
        is_train=True,  # Specify this is for training
    )
    val_dataset = CharacterDetectionDataset(
        annotation_file=config.data.val_annotation,
        image_base_dir=config.data.val_image_dir,
        unicode_map_file=config.data.unicode_dict,
        config=config,  # Pass the whole config
        is_train=False,  # Specify this is for validation
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,  # Use custom collate function
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,  # Use custom collate function
        pin_memory=True,
    )

    # --- Model, Optimizer, Scheduler ---
    print("Initializing model...")
    model = CharacterDetectionModel(config).to(device)
    # Log model architecture to WandB if enabled
    if wandb_run:
        # wandb.watch(model, log="all", log_freq=100) # Log gradients and parameters
        pass  # Watching can be slow, enable if needed

    # Optimizer
    if config.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
        )
    else:
        # Add other optimizers if needed
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    # Scheduler (using timm's CosineLRScheduler based on config)
    scheduler = CosineLRScheduler(
        optimizer,
        # Ensure parameters are correct type for scheduler
        t_initial=int(config.training.scheduler.t_initial),  # Total epochs
        lr_min=float(config.training.scheduler.lr_min),
        warmup_t=int(config.training.scheduler.warmup_t),
        warmup_lr_init=float(config.training.scheduler.warmup_lr_init),
        t_in_epochs=True,  # Make sure t_initial and warmup_t are in epochs
    )

    # --- Training Loop ---
    print("Starting training...")
    best_map = 0.0
    epochs_no_improve = 0
    start_epoch = 0  # Add support for resuming training later if needed

    # Determine total epochs from scheduler config
    total_epochs = config.training.scheduler.t_initial

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config, wandb_run)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, config, epoch, wandb_run)
        val_map = val_metrics["val/mAP"]
        val_acc = val_metrics["val/accuracy"]

        # Update scheduler (pass epoch number)
        scheduler.step(epoch + 1)  # timm scheduler steps based on epoch number

        # Logging
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{total_epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val mAP: {val_map:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Log epoch metrics to WandB
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/mAP": val_map,
                    "val/accuracy": val_acc,
                    "learning_rate": current_lr,
                    "epoch_time_seconds": epoch_time,
                }
            )

        # Checkpoint saving
        checkpoint_path = exp_dir / config.experiment.checkpoint_dir / f"epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_map": val_map,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model based on mAP
        if val_map > best_map:
            best_map = val_map
            epochs_no_improve = 0
            best_checkpoint_path = exp_dir / config.experiment.checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_map": val_map,
                },
                best_checkpoint_path,
            )
            print(f"*** New best model saved with mAP: {best_map:.4f} at epoch {epoch + 1} ***")
            if wandb_run:
                wandb.save(str(best_checkpoint_path))  # Save best model to wandb artifacts
        else:
            epochs_no_improve += 1

        # Early stopping
        if config.training.early_stopping_patience > 0 and epochs_no_improve >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    print("Training finished.")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Character Detection Model")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    main(args.config)
