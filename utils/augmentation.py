import albumentations as A
from typing import Dict

def get_column_detection_transforms(config: Dict) -> A.Compose:
    """列検出用のデータ拡張を取得する

    Args:
        config (Dict): 設定ファイルから読み込んだ設定

    Returns:
        A.Compose: データ拡張のパイプライン
    """
    aug_config = config['augmentation']
    return A.Compose(
        [
            A.RandomRotate90(p=0.0),  # 日本語の文書なので90度回転は不要
            A.Rotate(
                limit=aug_config['rotation'],
                p=0.5
            ),
            A.RandomScale(
                scale_limit=(
                    aug_config['scale'][0] - 1.0,
                    aug_config['scale'][1] - 1.0
                ),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness'],
                contrast_limit=aug_config['contrast'],
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(0, aug_config['gaussian_noise']),
                p=0.3
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=[]
        )
    )

def get_character_detection_transforms(config: Dict) -> A.Compose:
    """文字検出用のデータ拡張を取得する

    Args:
        config (Dict): 設定ファイルから読み込んだ設定

    Returns:
        A.Compose: データ拡張のパイプライン
    """
    aug_config = config['augmentation']
    return A.Compose(
        [
            A.RandomRotate90(p=0.0),  # 日本語の文書なので90度回転は不要
            A.Rotate(
                limit=aug_config['rotation'],
                p=0.5
            ),
            A.RandomScale(
                scale_limit=(
                    aug_config['scale'][0] - 1.0,
                    aug_config['scale'][1] - 1.0
                ),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness'],
                contrast_limit=aug_config['contrast'],
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(0, aug_config['gaussian_noise']),
                p=0.3
            ),
            A.RandomErasing(
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                p=aug_config['random_erase']
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    ) 