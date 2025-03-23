import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from sklearn.model_selection import train_test_split

from models.character_detection.model import CharacterDetectionModel
from utils.dataset import CharacterDetectionDataset
from utils.augmentation import get_character_detection_transforms
from utils.metrics import compute_map, compute_character_accuracy
from typing import Tuple, Dict, List

logger = get_logger(__name__)

class EarlyStopping:
    """早期停止の実装"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
) -> float:
    """1エポックの訓練を実行

    Args:
        model (nn.Module): 訓練するモデル
        train_loader (DataLoader): 訓練データローダー
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学習率スケジューラ
        accelerator (Accelerator): Acceleratorインスタンス

    Returns:
        float: 平均損失値
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(
        train_loader,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        # データの準備（Acceleratorが自動的にデバイスに転送）
        images = batch["image"]
        targets = {
            "boxes": batch["boxes"],
            "labels": batch["labels"],
        }

        # 順伝播と損失計算
        with accelerator.accumulate(model):
            loss_dict = model(images, targets)
            loss = loss_dict["loss"]

            # 逆伝播と最適化
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 損失値の記録
            loss_value = accelerator.gather(loss).mean().item()
            total_loss += loss_value

            # プログレスバーの更新
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            # wandbにステップごとの損失を記録
            if accelerator.is_local_main_process:
                wandb.log({"train/step_loss": loss_value})

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def visualize_predictions(
    image: torch.Tensor,
    pred_boxes: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    save_path: Path,
    score_threshold: float = 0.5,
) -> None:
    """推論結果を可視化して保存

    Args:
        image (torch.Tensor): 入力画像 [C, H, W]
        pred_boxes (List[torch.Tensor]): 予測バウンディングボックス
        pred_labels (List[torch.Tensor]): 予測ラベル
        pred_scores (List[torch.Tensor]): 予測スコア
        gt_boxes (List[torch.Tensor]): 正解バウンディングボックス
        gt_labels (List[torch.Tensor]): 正解ラベル
        save_path (Path): 保存先のパス
        score_threshold (float, optional): スコアのしきい値. Defaults to 0.5.
    """
    # 画像をnumpy配列に変換
    image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # プロットの準備
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    # 予測結果の描画
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue
        
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # バウンディングボックスを描画（赤色）
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # ラベルとスコアを表示
        ax.text(
            x1, y1 - 5,
            f'Pred: {label.item()}\n{score:.2f}',
            color='red',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8)
        )

    # 正解の描画
    for box, label in zip(gt_boxes, gt_labels):
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # バウンディングボックスを描画（緑色）
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='g',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # ラベルを表示
        ax.text(
            x1, y2 + 5,
            f'GT: {label.item()}',
            color='green',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8)
        )

    # 軸を消して保存
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    visualization_dir: Path = None,
    iou_threshold: float = 0.5,
    num_visualizations: int = 5,
) -> Tuple[float, float]:
    """検証を実行

    Args:
        model (nn.Module): 評価するモデル
        val_loader (DataLoader): 検証データローダー
        accelerator (Accelerator): Acceleratorインスタンス
        visualization_dir (Path, optional): 可視化結果の保存ディレクトリ. Defaults to None.
        iou_threshold (float, optional): IoUのしきい値. Defaults to 0.5.
        num_visualizations (int, optional): 可視化する画像の数. Defaults to 5.

    Returns:
        Tuple[float, float]: mAPと文字認識精度
    """
    model.eval()

    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []
    visualized_count = 0

    progress_bar = tqdm(
        val_loader,
        desc="Validating",
        disable=not accelerator.is_local_main_process,
    )

    for batch_idx, batch in enumerate(progress_bar):
        # データの準備
        images = batch["image"]
        gt_boxes = batch["boxes"]
        gt_labels = batch["labels"]

        # 推論
        predictions = model(images)
        pred_boxes = predictions["boxes"]
        pred_scores = predictions["scores"]
        pred_labels = predictions["labels"]

        # 結果の収集（全プロセスから）
        pred_boxes = accelerator.gather(pred_boxes)
        pred_scores = accelerator.gather(pred_scores)
        pred_labels = accelerator.gather(pred_labels)
        gt_boxes = accelerator.gather(gt_boxes)
        gt_labels = accelerator.gather(gt_labels)

        pred_boxes_list.extend(pred_boxes)
        pred_scores_list.extend(pred_scores)
        pred_labels_list.extend(pred_labels)
        gt_boxes_list.extend(gt_boxes)
        gt_labels_list.extend(gt_labels)

        # 可視化（メインプロセスのみ）
        if accelerator.is_local_main_process and visualization_dir is not None:
            for i in range(len(images)):
                if visualized_count >= num_visualizations:
                    break
                
                # 可視化して保存
                save_path = visualization_dir / f"batch_{batch_idx}_sample_{i}.png"
                visualize_predictions(
                    images[i],
                    [pred_boxes[i]],
                    [pred_labels[i]],
                    [pred_scores[i]],
                    [gt_boxes[i]],
                    [gt_labels[i]],
                    save_path
                )
                visualized_count += 1

                # wandbにアップロード
                if accelerator.is_local_main_process:
                    wandb.log({
                        f"val_predictions/image_{visualized_count}": wandb.Image(str(save_path))
                    })

    # メインプロセスでのみ評価を実行
    if accelerator.is_local_main_process:
        # mAPの計算
        mAP, _ = compute_map(
            pred_boxes_list, pred_scores_list, pred_labels_list,
            gt_boxes_list, gt_labels_list, iou_threshold=iou_threshold
        )

        # 文字認識精度の計算
        accuracy, _ = compute_character_accuracy(
            pred_labels_list, gt_labels_list,
            pred_boxes_list, gt_boxes_list,
            iou_threshold=iou_threshold
        )
    else:
        mAP = 0.0
        accuracy = 0.0

    # 全プロセスで結果を共有
    mAP = accelerator.gather(torch.tensor([mAP])).mean().item()
    accuracy = accelerator.gather(torch.tensor([accuracy])).mean().item()

    return mAP, accuracy

def main():
    # Acceleratorの初期化
    accelerator = Accelerator()
    logger.info(accelerator.state)

    # 設定の読み込み
    with open("config/model/character_detection.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 乱数シードの設定
    set_seed(42)

    # データセットの準備
    transform = get_character_detection_transforms(config)
    train_dataset = CharacterDetectionDataset(
        annotation_file="data/processed/train_column_info.csv",
        target_width=config["model"]["input_size"][0],
        transform=transform,
    )
    val_dataset = CharacterDetectionDataset(
        annotation_file="data/processed/val_column_info.csv",
        target_width=config["model"]["input_size"][0],
        transform=None,  # 検証時はデータ拡張なし
    )

    # データローダーの準備
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,  # バッチ内のデータを適切に結合
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,  # バッチ内のデータを適切に結合
        pin_memory=True,
    )

    # モデルの準備
    config["model"]["num_classes"] = train_dataset.num_classes
    model = CharacterDetectionModel(config)

    # 最適化器とスケジューラの準備
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["scheduler"]["total_epochs"],
    )

    # Acceleratorによるデバイスへの転送
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # wandbの初期化
    if accelerator.is_local_main_process:
        wandb.init(
            project="kuzushiji-character-detection",
            config=config,
            name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    # 実験結果の保存ディレクトリを作成
    save_dir = Path(datetime.now().strftime(config["experiment"]["save_dir"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / config["experiment"]["log_dir"]
    checkpoint_dir = save_dir / config["experiment"]["checkpoint_dir"]
    eval_dir = save_dir / config["experiment"]["eval_dir"]
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    # 早期停止の設定
    early_stopping = EarlyStopping(patience=10)

    # 訓練ループ
    best_map = 0.0
    for epoch in range(config["training"]["scheduler"]["total_epochs"]):
        logger.info(f"Epoch {epoch + 1}")

        # 訓練
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, accelerator)
        logger.info(f"Train Loss: {train_loss:.4f}")

        # 検証
        val_map, char_acc = validate(
            model,
            val_loader,
            accelerator,
            visualization_dir=eval_dir / f"epoch_{epoch + 1}",
        )
        logger.info(f"Validation mAP: {val_map:.4f}, Character Accuracy: {char_acc:.4f}")

        # wandbにログを記録
        if accelerator.is_local_main_process:
            wandb.log({
                "train/loss": train_loss,
                "val/mAP": val_map,
                "val/char_acc": char_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # モデルの保存
        if val_map > best_map:
            best_map = val_map
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    unwrapped_model.state_dict(),
                    checkpoint_dir / "best_model.pth",
                )

        # 早期停止の判定
        if early_stopping(val_map):
            logger.info("Early stopping triggered")
            break

    # wandbの終了
    if accelerator.is_local_main_process:
        wandb.finish()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """バッチ内のデータを適切に結合する関数

    Args:
        batch (List[Dict[str, torch.Tensor]]): バッチ内のデータのリスト

    Returns:
        Dict[str, torch.Tensor]: 結合されたデータ
    """
    # バッチ内の最大高さを取得
    max_height = max(item["height"] for item in batch)

    images = []
    boxes = []
    labels = []
    image_ids = []
    image_paths = []

    for item in batch:
        # 画像の高さを揃える
        image = item["image"]  # [C, H, W]
        current_height = image.shape[1]
        if current_height < max_height:
            # 下部にパディングを追加
            padding = torch.zeros(3, max_height - current_height, image.shape[2])
            image = torch.cat([image, padding], dim=1)
        images.append(image)

        boxes.append(item["boxes"])
        labels.append(item["labels"])
        image_ids.append(item["image_id"])
        image_paths.append(item["image_path"])

    # 画像はスタックして1つのテンソルに
    images = torch.stack(images)

    return {
        "image": images,
        "boxes": boxes,  # リストのまま（各画像で検出数が異なるため）
        "labels": labels,  # リストのまま（各画像で文字数が異なるため）
        "image_id": torch.tensor(image_ids),
        "image_path": image_paths,
    }


if __name__ == "__main__":
    main()
