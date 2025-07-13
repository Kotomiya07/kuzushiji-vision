# train_character_detection.py

import os
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils.util import EasyDict


def get_project_root():
    """プロジェクトのルートディレクトリを取得"""
    return Path(__file__).parent.parent


def main():
    """文字位置検出モデルの学習を実行"""
    # プロジェクトルートディレクトリに移動
    os.chdir(get_project_root())

    # 設定の読み込み
    with open("src/configs/model/character_detection.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 実験ディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments/character_detection") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 設定の保存
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    config = EasyDict(config)

    # モデルの準備
    model = YOLO(f"{config.model.backbone}")  # YOLOモデルをロード

    # モデル構造の調整
    model.model.nc = config.model.num_classes

    # 学習の設定
    train_args = {
        "data": "src/configs/data/character_detection.yaml",
        "epochs": config.training.scheduler.total_epochs,
        "batch": config.training.batch_size,
        "patience": config.training.patience,
        "imgsz": config.model.input_size[0],
        "device": 0,
        "workers": 24,
        "project": "experiments/character_detection",
        "name": timestamp,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": config.training.optimizer,
        "lr0": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "label_smoothing": 0.0,
        "scale": 0.5,
        "warmup_epochs": config.training.scheduler.warmup_epochs,
        "close_mosaic": 10,  # モザイク拡張を終了するエポック
        "flipud": config.augmentation.vertical_flip,
        "fliplr": config.augmentation.horizontal_flip,
        "mosaic": 1.0,  # モザイク拡張の確率
        "mixup": 0.0,  # mixupは使用しない
        "copy_paste": 0.0,  # copy-pasteも使用しない
        "degrees": config.augmentation.rotation[1],  # 回転の最大角度
        "hsv_h": 0.0,  # 色相の変更なし
        "hsv_s": 0.0,  # 彩度の変更なし
        "hsv_v": config.augmentation.brightness,  # 明度の変更
        "single_cls": True,  # 単一クラス検出（文字位置のみ）
        "cache": False,
        "multi_scale": False,
        "profile": False,
        "plots": True,
    }

    # 学習の実行
    model.train(**train_args)

    # ベストモデルをコピー
    best_model_path = exp_dir / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"Best model saved at: {best_model_path}")
    else:
        print("Warning: Best model not found!")


if __name__ == "__main__":
    main()
