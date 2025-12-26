"""RF-DETRによる文字位置検出モデル学習スクリプト."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from src.utils.util import EasyDict, recursive_to_dict

MODEL_REGISTRY: dict[str, type[RFDETRBase]] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def get_project_root() -> Path:
    """プロジェクトのルートディレクトリを取得"""
    return Path(__file__).parent.parent


def load_config(config_path: Path) -> EasyDict:
    with config_path.open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return EasyDict(config_dict)


def resolve_experiment_dir(config: EasyDict) -> Path:
    save_dir_pattern = config.experiment.get("save_dir", "experiments/rfdetr_character_detection/%Y%m%d_%H%M%S")
    exp_dir = Path(datetime.now().strftime(save_dir_pattern))
    exp_dir.mkdir(parents=True, exist_ok=True)

    for key in ("log_dir", "checkpoint_dir", "eval_dir"):
        subdir = config.experiment.get(key)
        if subdir:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_config_snapshot(config: EasyDict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        yaml.safe_dump(recursive_to_dict(config), f, allow_unicode=True, sort_keys=False)


def build_model(config: EasyDict):
    model_cfg = config.model
    size_key = model_cfg.get("size", "base").lower()
    if size_key not in MODEL_REGISTRY:
        raise ValueError(f"未知のモデルサイズです: {size_key}. 利用可能なサイズ: {', '.join(MODEL_REGISTRY.keys())}")

    model_params = dict(model_cfg.get("params", {}))
    num_classes = model_cfg.get("num_classes")
    if num_classes is not None:
        model_params.setdefault("num_classes", num_classes)

    resolution = model_cfg.get("resolution")
    if resolution is not None:
        model_params.setdefault("resolution", resolution)

    training_device = config.training.get("device", "cuda")
    model_params.setdefault("device", training_device)

    model_cls = MODEL_REGISTRY[size_key]
    return model_cls(**model_params)


def prepare_train_arguments(config: EasyDict, exp_dir: Path) -> dict:
    training_cfg = config.training
    scheduler_cfg = training_cfg.get("scheduler", {})
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    augmentation_cfg = config.get("augmentation", {})
    data_cfg = config.data
    wandb_cfg = config.get("wandb", {})
    tensorboard_cfg = config.get("tensorboard", {})

    dataset_dir = Path(data_cfg.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = get_project_root() / dataset_dir

    run_name = None
    name_format = wandb_cfg.get("name_format")
    if name_format:
        run_name = datetime.now().strftime(name_format)

    train_args = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": data_cfg.get("dataset_file", "roboflow"),
        "epochs": scheduler_cfg.get("total_epochs", training_cfg.get("epochs", 100)),
        "lr": training_cfg.get("learning_rate", 1e-4),
        "lr_encoder": training_cfg.get("learning_rate_encoder", training_cfg.get("learning_rate", 1e-4)),
        "batch_size": training_cfg.get("batch_size", 4),
        "grad_accum_steps": training_cfg.get("grad_accum_steps", 1),
        "weight_decay": training_cfg.get("weight_decay", 1e-4),
        "lr_drop": training_cfg.get("lr_drop", scheduler_cfg.get("total_epochs", 100) - 1),
        "warmup_epochs": scheduler_cfg.get("warmup_epochs", 0),
        "ema_decay": training_cfg.get("ema_decay", 0.993),
        "ema_tau": training_cfg.get("ema_tau", 100),
        "checkpoint_interval": training_cfg.get("checkpoint_interval", 10),
        "num_workers": training_cfg.get("num_workers", 4),
        "device": training_cfg.get("device", "cuda"),
        "amp": training_cfg.get("amp", True),
        "output_dir": str(exp_dir),
        "seed": config.get("seed", 42),
        "multi_scale": augmentation_cfg.get("multi_scale", True),
        "expanded_scales": augmentation_cfg.get("expanded_scales", True),
        "do_random_resize_via_padding": augmentation_cfg.get("random_resize_via_padding", False),
        "square_resize_div_64": augmentation_cfg.get("square_resize_div_64", True),
        "tensorboard": tensorboard_cfg.get("enable", True),
        "wandb": wandb_cfg.get("enable", False),
        "project": wandb_cfg.get("project"),
        "run": run_name,
        "class_names": data_cfg.get("class_names"),
        "run_test": training_cfg.get("run_test", False),
        "early_stopping": early_stopping_cfg.get("enabled", False),
        "early_stopping_patience": early_stopping_cfg.get("patience", 10),
        "early_stopping_min_delta": early_stopping_cfg.get("min_delta", 0.001),
        "early_stopping_use_ema": early_stopping_cfg.get("use_ema", False),
    }

    extra_params = training_cfg.get("params", {})
    train_args.update(extra_params)

    return train_args


def configure_environment(config: EasyDict, exp_dir: Path) -> None:
    wandb_cfg = config.get("wandb", {})
    os.environ.setdefault("KUZUSHIJI_EXPERIMENT_DIR", str(exp_dir))

    if not wandb_cfg.get("enable", False):
        os.environ.setdefault("WANDB_MODE", "offline")
    else:
        project = wandb_cfg.get("project")
        if project:
            os.environ.setdefault("WANDB_PROJECT", str(project))
        entity = wandb_cfg.get("entity")
        if entity:
            os.environ.setdefault("WANDB_ENTITY", str(entity))


def main() -> None:
    import sys

    os.chdir(get_project_root())

    # コマンドライン引数から設定ファイルを指定可能に
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("src/configs/model/character_detection_rfdetr.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    config = load_config(config_path)
    exp_dir = resolve_experiment_dir(config)
    save_config_snapshot(config, exp_dir / "config.yaml")

    model = build_model(config)
    train_args = prepare_train_arguments(config, exp_dir)
    configure_environment(config, exp_dir)

    print("=== RF-DETR 学習設定 ===")
    print(f"モデルサイズ: {config.model.get('size', 'base')}")
    print(f"データセット: {train_args['dataset_dir']}")
    print(f"出力先: {exp_dir}")

    model.train(**train_args)

    best_model_path = exp_dir / "checkpoint_best_total.pth"
    if best_model_path.exists():
        print(f"ベストモデルを保存しました: {best_model_path}")
    else:
        print("警告: checkpoint_best_total.pth が見つかりませんでした。 訓練が正常に完了したか確認してください。")


if __name__ == "__main__":
    main()
