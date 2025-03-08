import cv2
import numpy as np
from typing import Tuple, Optional

def resize_keeping_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    target_width: Optional[int] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, float]:
    """アスペクト比を保持したままリサイズを行う

    Args:
        image (np.ndarray): 入力画像
        target_size (int): リサイズ後の長辺のサイズ（target_widthが指定されている場合は無視）
        target_width (Optional[int], optional): リサイズ後の横幅。指定された場合はこのサイズに合わせる。Defaults to None.
        interpolation (int, optional): 補間方法. Defaults to cv2.INTER_LINEAR.

    Returns:
        Tuple[np.ndarray, float]: リサイズされた画像とスケール比
    """
    h, w = image.shape[:2]
    
    if target_width is not None:
        # 横幅指定の場合
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    else:
        # 長辺指定の場合
        if h > w:
            scale = target_size / h
            new_h = target_size
            new_w = int(w * scale)
        else:
            scale = target_size / w
            new_w = target_size
            new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, scale

def extract_text_columns(
    image: np.ndarray,
    column_boxes: np.ndarray,
    target_width: int = 192
) -> list[np.ndarray]:
    """画像から文字列を抽出する

    Args:
        image (np.ndarray): 入力画像
        column_boxes (np.ndarray): 列のバウンディングボックス [N, 4] (x1, y1, x2, y2)
        target_width (int, optional): 出力画像の横幅. Defaults to 192.

    Returns:
        list[np.ndarray]: 抽出された列画像のリスト
    """
    column_images = []
    for box in column_boxes:
        x1, y1, x2, y2 = box.astype(int)
        column_image = image[y1:y2, x1:x2]
        resized_column, _ = resize_keeping_aspect_ratio(
            column_image, 
            target_size=0,  # 無視される
            target_width=target_width
        )
        column_images.append(resized_column)
    return column_images

def normalize_image(image: np.ndarray) -> np.ndarray:
    """画像を正規化する（0-1の範囲に変換）

    Args:
        image (np.ndarray): 入力画像

    Returns:
        np.ndarray: 正規化された画像
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image 