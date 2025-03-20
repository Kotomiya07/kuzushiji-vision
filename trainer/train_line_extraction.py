import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def get_project_root():
    """プロジェクトのルートディレクトリを取得"""
    return Path(__file__).parent.parent

def main():
    """列検出モデルの学習を実行"""
    # プロジェクトルートディレクトリに移動
    os.chdir(get_project_root())
    
    # 設定の読み込み
    with open("config/model/line_extraction.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 実験ディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments/line_extraction") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 設定の保存
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # モデルの準備
    model = YOLO(f'{config["model"]["backbone"]}.pt')  # YOLOモデルをロード
    
    # モデル構造の調整
    model.model.nc = config["model"]["num_classes"]

    # 学習の設定
    train_args = {
        "data": "config/data/line_extraction.yaml",  # データセット設定ファイル（相対パス）
        "epochs": config["training"]["scheduler"]["total_epochs"],
        "batch": config["training"]["batch_size"],
        "imgsz": config["model"]["input_size"][0],
        "device": 0,
        "workers": 24,
        "project": "experiments/line_extraction",
        "name": timestamp,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "label_smoothing": 0.0,
        "scale": 0.5,
        "warmup_epochs": config["training"]["scheduler"]["warmup_epochs"],
        "close_mosaic": 10,  # モザイク拡張を終了するエポック
        "flipud": config["augmentation"]["vertical_flip"],
        "fliplr": config["augmentation"]["horizontal_flip"],
        "mosaic": 1.0,  # モザイク拡張の確率
        "mixup": 0.0,   # mixupは使用しない
        "copy_paste": 0.0,  # copy-pasteも使用しない
        "degrees": config["augmentation"]["rotation"][1],  # 回転の最大角度
        "hsv_h": 0.0,  # 色相の変更なし
        "hsv_s": 0.0,  # 彩度の変更なし
        "hsv_v": config["augmentation"]["brightness"],  # 明度の変更
        "single_cls": False,
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
