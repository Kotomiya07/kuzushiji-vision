import ast
import json  # Add json import
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentation import (
    get_character_detection_train_transforms,
    get_character_detection_val_transforms,
)  # Import new transform functions
from .image_processing import normalize_image, resize_keeping_aspect_ratio  # Needed for ColumnDetectionDataset
from .util import EasyDict  # Import EasyDict


class ColumnDetectionDataset(Dataset):
    """列検出のためのデータセット"""

    def __init__(self, image_dir: str, annotation_file: str, target_size: int = 640, transform=None):
        """
        Args:
            image_dir (str): 画像ディレクトリのパス
            annotation_file (str): アノテーションファイルのパス
            target_size (int, optional): リサイズ後の長辺のサイズ. Defaults to 640.
            transform: データ拡張の関数. Defaults to None.
        """
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # アノテーション情報の取得
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_name"])

        # 画像の読み込みとリサイズ
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image, scale = resize_keeping_aspect_ratio(image, target_size=self.target_size)

        # バウンディングボックスのスケーリング
        boxes = np.array(eval(row["column_boxes"]))  # [N, 4]
        boxes = boxes * scale

        # 正規化
        resized_image = normalize_image(resized_image)

        # データ拡張
        if self.transform:
            transformed = self.transform(image=resized_image, bboxes=boxes)
            resized_image = transformed["image"]
            boxes = np.array(transformed["bboxes"])

        return EasyDict({"image": resized_image, "boxes": boxes, "scale": scale})


class CharacterDetectionDataset(Dataset):
    """文字検出のデータセット"""
    """文字検出のデータセット"""

    def __init__(
        self,
        annotation_file: str,
        image_base_dir: str,
        unicode_map_file: str,  # No default, should be provided
        config: EasyDict,  # Pass the whole config
        is_train: bool = True,  # Flag to determine train/val transforms
    ):
        """
        Args:
            annotation_file (str): アノテーションファイルのパス
            image_base_dir (str): 画像ファイルが格納されているベースディレクトリのパス
            unicode_map_file (str): Unicode IDとインデックスのマッピングJSONファイルのパス.
            config (EasyDict): The main configuration object.
            is_train (bool, optional): If True, applies training augmentations. Otherwise, applies validation transforms. Defaults to True.
        """
        super().__init__()
        self.df = pd.read_csv(annotation_file)
        self.image_base_dir = image_base_dir  # Store image base directory
        self.target_width = config.model.input_size[0]  # Store target width again
        self.config = config  # Store config

        # Select appropriate transform based on is_train flag
        if is_train:
            self.transform = get_character_detection_train_transforms(config)
        else:
            self.transform = get_character_detection_val_transforms(config)

        # UnicodeコードのマッピングをJSONファイルから読み込む
        try:
            with open(unicode_map_file, encoding="utf-8") as f:
                self.unicode_to_idx = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Unicode map file not found at {unicode_map_file}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Failed to decode JSON from {unicode_map_file}")
            raise

        # Unicodeコードのマッピングを作成
        self.unicode_to_idx = {}
        for unicode_ids in self.df["unicode_ids"]:
            for unicode_id in ast.literal_eval(unicode_ids):
                if unicode_id not in self.unicode_to_idx:
                    self.unicode_to_idx[unicode_id] = len(self.unicode_to_idx)

    def __len__(self) -> int:
        return len(self.df)
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:  # Add try block to catch errors within __getitem__
            # データの取得
            row = self.df.iloc[idx]
            # Construct the full image path using the base directory
            relative_image_path_from_csv = row["column_image"]
            # Remove potential redundant prefix like 'train/images/' or 'val/images/'
            # This assumes the base dir already points to '.../train/images' or '.../val/images' etc.
            parts = relative_image_path_from_csv.split("/")
            if len(parts) > 2 and parts[0] in ["train", "val", "test"] and parts[1] == "images":
                relative_image_path = os.path.join(*parts[2:])  # Join the rest of the path parts
            else:
                relative_image_path = relative_image_path_from_csv  # Use as is if pattern doesn't match
            image_path = os.path.join(self.image_base_dir, relative_image_path)

            # Try loading image first
            try:
                # Load image using OpenCV as Albumentations expects numpy arrays
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Image not found or failed to load: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape  # Get original shape
            except Exception as e:
                print(f"ERROR loading image at index {idx}, path: {image_path}: {e}")
                return None  # Return None if image loading fails

            # Try parsing annotations
            try:
                char_boxes_str = row["char_boxes_in_column"]
                unicode_ids_str = row["unicode_ids"]
                char_boxes = ast.literal_eval(char_boxes_str)  # Use correct column name
                unicode_ids = ast.literal_eval(unicode_ids_str)
            except Exception as e:
                print(f"ERROR parsing annotations at index {idx}, path: {image_path}: {e}")
                print(f"  char_boxes_str: {char_boxes_str}")
                print(f"  unicode_ids_str: {unicode_ids_str}")
                return None  # Return None if annotation parsing fails

            # Resize image and scale boxes manually before Albumentations
            try:
                if w == 0:  # Avoid division by zero
                    print(f"WARNING: Original image width is 0 at index {idx}, path: {image_path}. Skipping sample.")
                    return None
                scale = self.target_width / w
                target_height = max(1, int(h * scale))

                # Resize image using cv2
                resized_image = cv2.resize(image, (self.target_width, target_height), interpolation=cv2.INTER_LINEAR)

                # Process and scale boxes
                scaled_boxes = []
                processed_labels = []
                if isinstance(char_boxes, list) and isinstance(unicode_ids, list):
                    if len(char_boxes) != len(unicode_ids):
                        print(
                            f"WARNING: Mismatch between char_boxes ({len(char_boxes)}) and unicode_ids ({len(unicode_ids)}) at index {idx}. Skipping sample."
                        )
                        return None

                    for box, unicode_id in zip(char_boxes, unicode_ids, strict=False):
                        if unicode_id not in self.unicode_to_idx:
                            print(
                                f"WARNING: Unicode ID '{unicode_id}' not found in mapping at index {idx}, path: {image_path}. Skipping this character."
                            )
                            continue  # Skip this specific character

                        if len(box) != 4:
                            print(
                                f"WARNING: Invalid box format {box} at index {idx}, path: {image_path}. Skipping this character."
                            )
                            continue

                        x1, y1, x2, y2 = box
                        # Scale coordinates
                        x1_s = x1 * scale
                        y1_s = y1 * scale
                        x2_s = x2 * scale
                        y2_s = y2 * scale
                        # Clip coordinates to resized image boundaries
                        x1_s = np.clip(x1_s, 0, self.target_width)
                        y1_s = np.clip(y1_s, 0, target_height)
                        x2_s = np.clip(x2_s, 0, self.target_width)
                        y2_s = np.clip(y2_s, 0, target_height)
                        # Ensure x1 < x2 and y1 < y2 after clipping
                        if x1_s >= x2_s or y1_s >= y2_s:
                            print(
                                f"WARNING: Box collapsed after scaling/clipping at index {idx}, path: {image_path}. Original: {box}, Scaled: {[x1_s, y1_s, x2_s, y2_s]}. Skipping character."
                            )
                            continue

                        scaled_boxes.append([x1_s, y1_s, x2_s, y2_s])
                        processed_labels.append(self.unicode_to_idx[unicode_id])

                # Convert to numpy arrays for Albumentations
                scaled_boxes = np.array(scaled_boxes, dtype=np.float32)
                processed_labels = np.array(processed_labels if processed_labels else [], dtype=np.int64)

                # Ensure boxes have shape (N, 4) even if N=0
                if scaled_boxes.ndim == 1 and scaled_boxes.shape[0] == 0:
                    scaled_boxes = scaled_boxes.reshape(0, 4)

            except Exception as e:
                print(f"ERROR processing boxes/labels at index {idx}, path: {image_path}: {e}")
                import traceback

                traceback.print_exc()
                return None

            # Apply Albumentations transforms (handles augmentations, normalize, ToTensor)
            try:
                if self.transform:
                    # Pass resized image and scaled boxes/labels to the transform
                    transformed = self.transform(image=resized_image, bboxes=scaled_boxes, labels=processed_labels)
                    image_tensor = transformed["image"]
                    # Ensure tensors are created correctly even if bboxes/labels are empty
                    boxes_tensor = torch.tensor(
                        transformed["bboxes"] if len(transformed["bboxes"]) > 0 else [], dtype=torch.float32
                    )
                    labels_tensor = torch.tensor(
                        transformed["labels"] if len(transformed["labels"]) > 0 else [], dtype=torch.long
                    )

                    # Ensure boxes have shape (N, 4) even if N=0 after transform
                    if boxes_tensor.ndim == 1 and boxes_tensor.shape[0] == 0:
                        boxes_tensor = boxes_tensor.reshape(0, 4)
                    # Ensure labels have shape (N,) even if N=0
                    if labels_tensor.ndim == 0 and labels_tensor.numel() == 0:  # Handle case where tensor([]) is created
                        labels_tensor = torch.tensor([], dtype=torch.long)

                else:
                    # Fallback if no transform (should not happen)
                    print(f"WARNING: No transform defined for dataset at index {idx}. Returning resized/scaled data.")
                    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
                    boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
                    labels_tensor = torch.tensor(processed_labels, dtype=torch.long)

                # Final height is the height after manual resize
                final_h = target_height

            except Exception as e:
                print(f"ERROR applying transforms at index {idx}, path: {image_path}: {e}")
                import traceback

                traceback.print_exc()
                return None

            # Final check for empty tensors which might still cause issues downstream depending on collate/loss
            # Final check for empty tensors after transformation
            if boxes_tensor.shape[0] == 0:  # Check only boxes, labels might be empty if all chars were skipped
                # print(f"DEBUG: Empty boxes tensor after transform at index {idx}, path: {image_path}. Boxes: {boxes_tensor.shape}")
                # Decide whether to skip samples with no valid boxes after transform/filtering
                # return None # Option to skip
                pass  # Keep empty tensors for now

            # print(f"DEBUG: Index {idx}, Image Path: {image_path}, Final Image: {image_tensor.shape}, Boxes: {boxes_tensor.shape}, Labels: {labels_tensor.shape}") # デバッグ用ログ
            return EasyDict(
                {
                    "image": image_tensor,
                    "boxes": boxes_tensor,
                    "labels": labels_tensor,
                    "image_id": idx,
                    "image_path": image_path,  # Add image path to the output dict
                    "height": final_h,  # Add final height to the output dict
                }
            )
        except Exception as e:  # Catch any other unexpected errors in __getitem__
            print(f"UNEXPECTED ERROR in __getitem__ at index {idx}: {e}")
            import traceback

            traceback.print_exc()
            return None  # Return None on any error

    # This line intentionally left blank after removing misplaced code

    @property
    def num_classes(self) -> int:
        """文字クラスの数を返す"""
        return len(self.unicode_to_idx)
