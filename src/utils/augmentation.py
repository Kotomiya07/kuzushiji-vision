import albumentations as A
import cv2  # Import OpenCV for interpolation flags
import torch
from albumentations.pytorch import ToTensorV2


class AlbumentationsToTorchTransform:
    """AlbumentationsのTransformをPyTorchのTransformに変換するクラス"""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # テンソルをnumpy配列に変換
        image = image.permute(1, 2, 0).numpy()

        # Albumentationsの変換を適用
        transformed = self.transform(image=image)
        image = transformed["image"]

        # numpy配列をテンソルに戻す
        return torch.from_numpy(image).permute(2, 0, 1)


def get_column_detection_transforms(config: dict) -> A.Compose:
    """列検出用のデータ拡張を取得する

    Args:
        config (Dict): 設定ファイルから読み込んだ設定

    Returns:
        A.Compose: データ拡張のパイプライン
    """
    aug_config = config["augmentation"]
    return A.Compose(
        [
            A.RandomRotate90(p=0.0),  # 日本語の文書なので90度回転は不要
            A.Rotate(limit=aug_config["rotation"], p=0.5),
            A.RandomScale(scale_limit=(aug_config["scale"][0] - 1.0, aug_config["scale"][1] - 1.0), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config["brightness"], contrast_limit=aug_config["contrast"], p=0.5
            ),
            A.GaussNoise(p=0.3),  # デフォルトのパラメータを使用
            A.GaussNoise(p=0.3),  # デフォルトのパラメータを使用
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
    )


def get_character_detection_train_transforms(config: dict) -> A.Compose:
    """文字検出用の訓練データ拡張を取得する (Albumentations)

    Args:
        config (dict): 設定ファイルから読み込んだ設定

    Returns:
        T.Compose: データ拡張のパイプライン
    """
    aug_config = config["augmentation"]
    target_width = config.model.input_size[0]

    return A.Compose(
        [
            # Resize is now handled in the Dataset __getitem__ before this transform
            # 1. Geometric and Color Augmentations (applied to image and bboxes)
            A.Rotate(limit=aug_config["rotation"], p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad with 0
            A.RandomBrightnessContrast(
                brightness_limit=aug_config["brightness"], contrast_limit=aug_config["contrast"], p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.2, blur_limit=7),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),  # Updated args
            A.GridDistortion(p=0.2, distort_limit=0.1, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad with 0
            # 3. Dropout Augmentations (applied after geometric transforms)
            # Note: Albumentations dropout might affect bbox visibility, use with caution or adjust params
            A.CoarseDropout(
                max_holes=8,
                max_height=int(target_width * 0.1),
                max_width=int(target_width * 0.1),  # Scale hole size with target_width
                min_holes=1,
                min_height=int(target_width * 0.02),
                min_width=int(target_width * 0.02),
                fill_value=0,  # Fill with 0 (black)
                mask_fill_value=None,  # Use fill_value for mask
                p=aug_config.get("coarse_dropout_p", 0.3),  # Use config value or default
            ),
            # GridDropout might be less common/useful here, consider removing or adjusting
            # A.GridDropout(ratio=0.1, random_offset=True, fill_value=0, p=0.2),
            # RandomErasing equivalent (applied to tensor later, or use CoarseDropout)
            # Albumentations doesn't have a direct RandomErasing equivalent that works well with bboxes easily.
            # CoarseDropout is similar. T.RandomErasing is applied later if needed, but it won't affect bboxes.
            # Let's rely on CoarseDropout for now.
            # 4. Normalization (applied to image only)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 5. Convert to Tensor (applied to image only)
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def get_character_detection_val_transforms(config: dict) -> A.Compose:
    """文字検出用の検証データ拡張（リサイズ、正規化、テンソル化のみ）を取得する

    Args:
        config (dict): 設定ファイルから読み込んだ設定

    Returns:
        A.Compose: データ拡張のパイプライン
    """
    config.model.input_size[0]

    return A.Compose(
        [
            # Resize is now handled in the Dataset __getitem__ before this transform
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
