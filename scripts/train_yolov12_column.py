"""YOLOv12を使用して列検出モデルをトレーニングするスクリプト

このスクリプトは、data/kuzushiji-column以下のデータを使用してYOLOv12をトレーニングします。
データはすべてtrainデータになっているため、8:1:1の比率でtrain:val:testに分割してから使用します。
トレーニング中にtrainデータとvalデータ双方に対して精度を計算してログします。
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split
from yolov12.ultralytics import YOLO


def get_project_root() -> Path:
    """プロジェクトのルートディレクトリを取得"""
    return Path(__file__).parent.parent


def split_dataset(
    data_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """データセットをtrain:val:testに分割

    Args:
        data_dir: データセットのルートディレクトリ
        train_ratio: trainデータの割合（デフォルト: 0.8）
        val_ratio: valデータの割合（デフォルト: 0.1）
        test_ratio: testデータの割合（デフォルト: 0.1）
        seed: 乱数シード

    Returns:
        (train_files, val_files, test_files): 各セットのファイル名リスト（拡張子なし）
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    images_dir = data_dir / "train" / "images"
    labels_dir = data_dir / "train" / "labels"

    # 画像ファイルのリストを取得（拡張子なし）
    image_files = sorted([f.stem for f in images_dir.glob("*.jpg") if (labels_dir / f"{f.stem}.txt").exists()])

    if len(image_files) == 0:
        raise ValueError(f"No matching image-label pairs found in {images_dir}")

    print(f"Found {len(image_files)} image-label pairs")

    # まずtrainとtemp（val+test）に分割
    train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True)

    # tempをvalとtestに分割
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_test_ratio), random_state=seed, shuffle=True)

    print(f"Split dataset: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    return train_files, val_files, test_files


def copy_files_to_split_dirs(data_dir: Path, train_files: list[str], val_files: list[str], test_files: list[str]) -> None:
    """ファイルをval/testディレクトリにコピー

    Args:
        data_dir: データセットのルートディレクトリ
        train_files: trainファイル名リスト（拡張子なし、使用しない）
        val_files: valファイル名リスト（拡張子なし）
        test_files: testファイル名リスト（拡張子なし）

    Note:
        trainファイルは既にdata_dir/train/に存在するため、コピーしません。
        valとtestのファイルのみをコピーします。
    """
    source_images_dir = data_dir / "train" / "images"
    source_labels_dir = data_dir / "train" / "labels"

    # trainファイルは既に存在するため、valとtestのみをコピー
    for split_name, file_list in [("val", val_files), ("test", test_files)]:
        split_images_dir = data_dir / split_name / "images"
        split_labels_dir = data_dir / split_name / "labels"

        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for file_stem in file_list:
            # 画像ファイルをコピー（拡張子を探す）
            for ext in [".jpg", ".png", ".jpeg"]:
                src_img = source_images_dir / f"{file_stem}{ext}"
                if src_img.exists():
                    dst_img = split_images_dir / f"{file_stem}{ext}"
                    shutil.copy2(src_img, dst_img)
                    break

            # ラベルファイルをコピー
            src_label = source_labels_dir / f"{file_stem}.txt"
            if src_label.exists():
                dst_label = split_labels_dir / f"{file_stem}.txt"
                shutil.copy2(src_label, dst_label)

        print(f"Copied {len(file_list)} files to {split_name} directory")

    print(f"Train files ({len(train_files)} files) are already in {data_dir / 'train'}, skipping copy")


def create_data_yaml(data_dir: Path, output_path: Path, num_classes: int = 1, class_names: list[str] | None = None) -> None:
    """YOLO用のdata.yamlファイルを作成

    Args:
        data_dir: データセットのルートディレクトリ
        output_path: 出力先のdata.yamlファイルパス
        num_classes: クラス数
        class_names: クラス名のリスト（Noneの場合は['column']を使用）
    """
    if class_names is None:
        class_names = ["column"]

    # 相対パスで設定
    data_config = {
        "path": str(data_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Created data.yaml at {output_path}")


def main():
    """列検出モデルの学習を実行"""
    # プロジェクトルートディレクトリに移動
    project_root = get_project_root()
    os.chdir(project_root)

    # データディレクトリの設定
    data_dir = project_root / "data" / "kuzushiji-column"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # データセットの分割
    print("Splitting dataset into train:val:test (8:1:1)...")
    train_files, val_files, test_files = split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # 既存の分割ディレクトリをクリーンアップ（オプション）
    for split_name in ["train", "val", "test"]:
        split_dir = data_dir / split_name
        if split_dir.exists() and split_name != "train":  # trainディレクトリは元のデータがあるので残す
            # val/testディレクトリのみ削除
            if (split_dir / "images").exists():
                shutil.rmtree(split_dir / "images")
            if (split_dir / "labels").exists():
                shutil.rmtree(split_dir / "labels")

    # ファイルを分割ディレクトリにコピー
    print("Copying files to split directories...")
    copy_files_to_split_dirs(data_dir, train_files, val_files, test_files)

    # data.yamlファイルを作成
    data_yaml_path = data_dir / "data.yaml"
    create_data_yaml(data_dir, data_yaml_path, num_classes=1, class_names=["column"])

    # 実験ディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments/yolov12_column") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # YOLOv12モデルの準備
    # YOLOv12のモデル名を試行（利用可能な場合）
    # 利用できない場合はYOLOv11にフォールバック
    model_names_to_try = ["yolo12x.pt", "yolo12s.pt", "yolo11n.pt"]
    model = None
    model_name = None

    for name in model_names_to_try:
        try:
            model = YOLO(name)
            model_name = name
            print(f"Successfully loaded model: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load any YOLO model. Tried: {model_names_to_try}")

    # 学習の設定
    train_args = {
        "data": str(data_yaml_path.absolute()),
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "device": "cuda:0",
        "workers": 8,
        "project": "experiments/yolov12_column",
        "name": timestamp,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 3,
        "label_smoothing": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,  # 回転なし（古文書のため）
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,  # 垂直反転なし
        "fliplr": 0.0,  # 水平反転なし（日本語文書のため）
        "hsv_h": 0.0,  # 色相変更なし
        "hsv_s": 0.0,  # 彩度変更なし
        "hsv_v": 0.1,  # 明度変更（軽微）
        "close_mosaic": 10,
        "single_cls": False,
        "cache": False,
        "multi_scale": False,
        "profile": False,
        "plots": True,
        "save": True,
        "save_period": 10,
        "val": True,  # 検証を有効化
    }

    # 学習の実行
    print(f"Starting training with model: {model_name}")
    print(f"Training arguments: {train_args}")
    results = model.train(**train_args)

    # 結果の保存
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

    # ベストモデルのパスを確認
    best_model_path = exp_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        # Ultralyticsのデフォルト保存場所を確認
        default_best = Path("experiments/yolov12_column") / timestamp / "weights" / "best.pt"
        if default_best.exists():
            best_model_path = default_best

    if best_model_path.exists():
        print(f"Best model saved at: {best_model_path}")
    else:
        print("Warning: Best model not found!")

    # 学習結果のサマリーを表示
    if hasattr(results, "results_dict"):
        print("\nTraining Results Summary:")
        for key, value in results.results_dict.items():
            print(f"  {key}: {value}")

    print(f"\nExperiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
