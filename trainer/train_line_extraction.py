import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from models.column_extraction.model import ColumnDetectionModel
from utils.dataset import ColumnDetectionDataset
from utils.augmentation import get_column_detection_transforms
from utils.metrics import compute_map


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> float:
    """1エポックの訓練を実行

    Args:
        model (nn.Module): 訓練するモデル
        train_loader (DataLoader): 訓練データローダー
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学習率スケジューラ
        device (torch.device): デバイス

    Returns:
        float: 平均損失値
    """
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        # データの準備
        images = batch["image"].to(device)
        targets = {
            "boxes": [boxes.to(device) for boxes in batch["boxes"]],
            "labels": [torch.zeros_like(boxes[:, 0], dtype=torch.long).to(device) for boxes in batch["boxes"]],
        }

        # 順伝播と損失計算
        loss_dict = model(images, targets)
        loss = loss_dict["loss"]

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 学習率の更新
    scheduler.step()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, iou_threshold: float = 0.5) -> float:
    """検証を実行

    Args:
        model (nn.Module): 評価するモデル
        val_loader (DataLoader): 検証データローダー
        device (torch.device): デバイス
        iou_threshold (float, optional): IoUのしきい値. Defaults to 0.5.

    Returns:
        float: mAP
    """
    model.eval()

    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    for batch in tqdm(val_loader, desc="Validating"):
        # データの準備
        images = batch["image"].to(device)
        gt_boxes = batch["boxes"]

        # 推論
        predictions = model(images)
        pred_boxes = predictions["boxes"]
        pred_scores = predictions["scores"]

        # 結果の収集
        pred_boxes_list.extend(pred_boxes)
        pred_scores_list.extend(pred_scores)
        pred_labels_list.extend([torch.zeros_like(score, dtype=torch.long) for score in pred_scores])
        gt_boxes_list.extend(gt_boxes)
        gt_labels_list.extend([torch.zeros(len(boxes), dtype=torch.long) for boxes in gt_boxes])

    # mAPの計算
    mAP, _ = compute_map(
        pred_boxes_list, pred_scores_list, pred_labels_list, gt_boxes_list, gt_labels_list, iou_threshold=iou_threshold
    )

    return mAP


def main():
    # 設定の読み込み
    with open("config/model/column_extraction.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 実験ディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/column_detection/{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 設定の保存
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットとデータローダーの準備
    train_transform = get_column_detection_transforms(config)

    train_dataset = ColumnDetectionDataset(
        image_dir="data/processed/train/images",
        annotation_file="data/processed/train/column_annotations.csv",
        target_size=config["model"]["input_size"][0],
        transform=train_transform,
    )

    val_dataset = ColumnDetectionDataset(
        image_dir="data/processed/val/images",
        annotation_file="data/processed/val/column_annotations.csv",
        target_size=config["model"]["input_size"][0],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x,  # バッチ内の画像サイズが異なるため
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4, collate_fn=lambda x: x
    )

    # モデルの準備
    model = ColumnDetectionModel(config)
    model.to(device)

    # オプティマイザとスケジューラの設定
    optimizer = AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["scheduler"]["total_epochs"], eta_min=0)

    # 訓練ループ
    best_map = 0.0
    for epoch in range(config["training"]["scheduler"]["total_epochs"]):
        print(f"\nEpoch {epoch + 1}")

        # 訓練
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        # 検証
        val_map = validate(model, val_loader, device)
        print(f"Validation mAP: {val_map:.4f}")

        # チェックポイントの保存
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_map": val_map,
        }

        torch.save(checkpoint, exp_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        # ベストモデルの保存
        if val_map > best_map:
            best_map = val_map
            torch.save(checkpoint, exp_dir / "best.pt")
            print(f"New best model saved! mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
