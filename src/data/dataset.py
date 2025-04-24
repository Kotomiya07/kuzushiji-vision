import ast
import json
import os

from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ..utils.augmentation import (
    get_character_detection_train_transforms,
    get_character_detection_val_transforms,
)
from ..utils.image_processing import normalize_image, resize_keeping_aspect_ratio
from ..utils.util import EasyDict 


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

        # 画像の読み込みとリサイズ (Pillowを使用)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"ERROR loading image with Pillow at index {idx}, path: {image_path}: {e}")
            # Return a dummy dict or raise an error, depending on desired behavior
            # For now, let's return None and handle it in the collate_fn
            return None # Or handle appropriately

        # Pillow ImageをNumPy配列に変換してリサイズ関数に渡す
        image_np = np.array(image)
        resized_image_np, scale = resize_keeping_aspect_ratio(image_np, target_size=self.target_size)

        # バウンディングボックスのスケーリング
        boxes = np.array(eval(row["column_boxes"]))  # [N, 4]
        boxes = boxes * scale

        # 正規化 (NumPy配列に対して行う)
        resized_image_np = normalize_image(resized_image_np)

        # データ拡張 (AlbumentationsはNumPy配列を期待)
        if self.transform:
            transformed = self.transform(image=resized_image_np, bboxes=boxes)
            resized_image_np = transformed["image"]
            boxes = np.array(transformed["bboxes"])

        # NumPy配列を返す
        return EasyDict({"image": resized_image_np, "boxes": boxes, "scale": scale})


class CharacterDetectionDataset(Dataset):
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

        # Unicodeコードのマッピングを作成 (This part seems redundant if loaded from file, consider removing or adjusting logic)
        # self.unicode_to_idx = {} # Comment out or remove if loading from file is the primary method
        # for unicode_ids in self.df["unicode_ids"]:
        #     try: # Add try-except for potential eval errors
        #         ids_list = ast.literal_eval(unicode_ids)
        #         for unicode_id in ids_list:
        #             if unicode_id not in self.unicode_to_idx:
        #                 self.unicode_to_idx[unicode_id] = len(self.unicode_to_idx)
        #     except (ValueError, SyntaxError) as e:
        #         print(f"Warning: Could not parse unicode_ids: {unicode_ids}. Error: {e}")
        #         continue # Skip problematic rows

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None: # Return None on error
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

            # Try loading image first using Pillow
            try:
                # Load image using Pillow and convert to RGB
                image = Image.open(image_path).convert("RGB")
                w, h = image.size # Get original shape using Pillow's size attribute
            except FileNotFoundError:
                print(f"ERROR: Image file not found at index {idx}, path: {image_path}")
                return None
            except Exception as e:
                print(f"ERROR loading image with Pillow at index {idx}, path: {image_path}: {e}")
                return None  # Return None if image loading fails

            # Try parsing annotations
            try:
                char_boxes_str = row["char_boxes_in_column"]
                unicode_ids_str = row["unicode_ids"]
                # Use safer eval or preferably json.loads if the format allows
                try:
                    char_boxes = ast.literal_eval(char_boxes_str)
                    unicode_ids = ast.literal_eval(unicode_ids_str)
                except (ValueError, SyntaxError) as eval_e:
                     print(f"ERROR parsing annotations with ast.literal_eval at index {idx}, path: {image_path}: {eval_e}")
                     print(f"  char_boxes_str: {char_boxes_str}")
                     print(f"  unicode_ids_str: {unicode_ids_str}")
                     return None # Return None if annotation parsing fails

                # Basic type checking after eval
                if not isinstance(char_boxes, list) or not isinstance(unicode_ids, list):
                    print(f"ERROR: Parsed annotations are not lists at index {idx}, path: {image_path}")
                    print(f"  char_boxes type: {type(char_boxes)}")
                    print(f"  unicode_ids type: {type(unicode_ids)}")
                    return None

            except Exception as e: # Catch other potential errors during access or eval
                print(f"ERROR accessing or parsing annotations at index {idx}, path: {image_path}: {e}")
                # Optionally print problematic strings if they exist
                try:
                    print(f"  char_boxes_str: {row.get('char_boxes_in_column', 'N/A')}")
                    print(f"  unicode_ids_str: {row.get('unicode_ids', 'N/A')}")
                except Exception:
                    pass # Ignore errors printing the strings themselves
                return None  # Return None if annotation parsing fails

            # Resize image and scale boxes manually before Albumentations
            try:
                if w == 0:  # Avoid division by zero
                    print(f"WARNING: Original image width is 0 at index {idx}, path: {image_path}. Skipping sample.")
                    return None
                scale = self.target_width / w
                target_height = max(1, int(h * scale)) # Ensure target height is at least 1

                # Resize image using Pillow
                # Use Resampling.BILINEAR for Pillow >= 9.0.0, or Image.BILINEAR for older versions
                try:
                    resampling_method = Image.Resampling.BILINEAR
                except AttributeError:
                    resampling_method = Image.BILINEAR # Fallback for older Pillow
                resized_image = image.resize((self.target_width, target_height), resampling_method)

                # Convert resized Pillow image to NumPy array for Albumentations
                resized_image_np = np.array(resized_image)

                # Process and scale boxes
                scaled_boxes = []
                processed_labels = []
                # Ensure char_boxes and unicode_ids are lists before checking length
                if isinstance(char_boxes, list) and isinstance(unicode_ids, list):
                    if len(char_boxes) != len(unicode_ids):
                        print(
                            f"WARNING: Mismatch between char_boxes ({len(char_boxes)}) and unicode_ids ({len(unicode_ids)}) at index {idx}, path: {image_path}. Skipping sample."
                        )
                        # Consider if skipping the whole sample is desired, or just skipping processing boxes/labels
                        return None # Skip sample for now

                    for box, unicode_id in zip(char_boxes, unicode_ids, strict=False): # strict=False might hide issues if lengths differ
                        # Check if unicode_id exists in the map
                        if unicode_id not in self.unicode_to_idx:
                            # print(f"WARNING: Unicode ID '{unicode_id}' not found in mapping at index {idx}, path: {image_path}. Skipping this character.")
                            continue  # Skip this specific character silently or log verbosely

                        # Validate box format
                        if not isinstance(box, (list, tuple)) or len(box) != 4:
                            print(
                                f"WARNING: Invalid box format {box} at index {idx}, path: {image_path}. Skipping this character."
                            )
                            continue

                        # Ensure box coordinates are numeric before scaling
                        try:
                            x1, y1, x2, y2 = map(float, box) # Convert to float just in case
                        except (ValueError, TypeError):
                             print(
                                f"WARNING: Non-numeric box coordinates {box} at index {idx}, path: {image_path}. Skipping this character."
                            )
                             continue

                        # Scale coordinates
                        x1_s = x1 * scale
                        y1_s = y1 * scale
                        x2_s = x2 * scale
                        y2_s = y2 * scale
                        # Clip coordinates to resized image boundaries
                        x1_s = np.clip(x1_s, 0, self.target_width - 1) # Use target_width-1 and target_height-1 for 0-based indexing safety
                        y1_s = np.clip(y1_s, 0, target_height - 1)
                        x2_s = np.clip(x2_s, 0, self.target_width - 1)
                        y2_s = np.clip(y2_s, 0, target_height - 1)

                        # Ensure x1 < x2 and y1 < y2 after clipping (allow for minimal size)
                        min_box_size = 1e-3 # Define a minimum size to avoid zero-area boxes
                        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
                            # print(
                            #     f"WARNING: Box collapsed after scaling/clipping at index {idx}, path: {image_path}. Original: {box}, Scaled: {[x1_s, y1_s, x2_s, y2_s]}. Skipping character."
                            # )
                            continue # Skip collapsed boxes silently or log

                        scaled_boxes.append([x1_s, y1_s, x2_s, y2_s])
                        processed_labels.append(self.unicode_to_idx[unicode_id])
                else:
                     print(f"WARNING: char_boxes or unicode_ids are not lists at index {idx}, path: {image_path}. Skipping box processing.")
                     # Handle case where annotations were invalid earlier but didn't cause return None


                # Convert to numpy arrays for Albumentations
                # Ensure correct dtype even for empty lists
                scaled_boxes = np.array(scaled_boxes if scaled_boxes else [], dtype=np.float32)
                processed_labels = np.array(processed_labels if processed_labels else [], dtype=np.int64)

                # Ensure boxes have shape (N, 4) even if N=0
                if scaled_boxes.ndim == 1 and scaled_boxes.shape[0] == 0:
                    scaled_boxes = scaled_boxes.reshape(0, 4)
                elif scaled_boxes.ndim == 1 and scaled_boxes.shape[0] == 4: # Handle case of single box
                     scaled_boxes = scaled_boxes.reshape(1, 4)
                elif scaled_boxes.ndim != 2 or (scaled_boxes.shape[0] > 0 and scaled_boxes.shape[1] != 4):
                    print(f"ERROR: scaled_boxes has unexpected shape {scaled_boxes.shape} at index {idx}, path: {image_path}. Skipping sample.")
                    return None


            except Exception as e:
                print(f"ERROR processing boxes/labels at index {idx}, path: {image_path}: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Apply Albumentations transforms (handles augmentations, normalize, ToTensor)
            try:
                if self.transform:
                    # Pass resized image (as NumPy array) and scaled boxes/labels to the transform
                    transformed = self.transform(image=resized_image_np, bboxes=scaled_boxes, labels=processed_labels)
                    image_tensor = transformed["image"] # Should be a Tensor after ToTensorV2
                    # Ensure tensors are created correctly even if bboxes/labels are empty
                    # Albumentations usually returns lists, convert them safely
                    transformed_bboxes = transformed.get("bboxes", [])
                    transformed_labels = transformed.get("labels", [])

                    boxes_tensor = torch.tensor(
                        transformed_bboxes if transformed_bboxes else [], dtype=torch.float32
                    )
                    labels_tensor = torch.tensor(
                        transformed_labels if transformed_labels else [], dtype=torch.long
                    )

                    # Ensure boxes have shape (N, 4) even if N=0 after transform
                    if boxes_tensor.ndim == 1 and boxes_tensor.shape[0] == 0:
                        boxes_tensor = boxes_tensor.reshape(0, 4)
                    elif boxes_tensor.ndim == 1 and boxes_tensor.shape[0] == 4: # Handle single box case
                        boxes_tensor = boxes_tensor.reshape(1, 4)
                    elif boxes_tensor.ndim != 2 or (boxes_tensor.shape[0] > 0 and boxes_tensor.shape[1] != 4):
                         print(f"ERROR: boxes_tensor after transform has unexpected shape {boxes_tensor.shape} at index {idx}, path: {image_path}. Skipping sample.")
                         return None


                    # Ensure labels have shape (N,) even if N=0
                    if labels_tensor.ndim == 0 and labels_tensor.numel() == 0:  # Handle case where tensor([]) is created
                        labels_tensor = torch.tensor([], dtype=torch.long)
                    elif labels_tensor.ndim != 1:
                         print(f"ERROR: labels_tensor after transform has unexpected shape {labels_tensor.shape} at index {idx}, path: {image_path}. Skipping sample.")
                         return None


                else:
                    # Fallback if no transform (should not happen with current init logic)
                    print(f"WARNING: No transform defined for dataset at index {idx}. Returning resized/scaled data without ToTensor.")
                    # Convert NumPy array to Tensor manually if needed (assuming normalize is not done)
                    image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).float() / 255.0 # Basic ToTensor
                    boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
                    labels_tensor = torch.tensor(processed_labels, dtype=torch.long)

                # Final height is the height after manual resize
                final_h = target_height # Height of the image passed to albumentations

            except Exception as e:
                print(f"ERROR applying transforms at index {idx}, path: {image_path}: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Final check for empty tensors after transformation
            # Depending on the loss function, empty boxes/labels might be acceptable or problematic
            if boxes_tensor.shape[0] == 0:
                # print(f"DEBUG: Empty boxes tensor after transform at index {idx}, path: {image_path}. Boxes: {boxes_tensor.shape}")
                # Decide whether to skip samples with no valid boxes after transform/filtering
                # return None # Option to skip if downstream cannot handle 0 boxes
                pass # Keep empty tensors for now, assuming collate_fn or model can handle it

            # print(f"DEBUG: Index {idx}, Image Path: {image_path}, Final Image: {image_tensor.shape}, Boxes: {boxes_tensor.shape}, Labels: {labels_tensor.shape}") # デバッグ用ログ
            return EasyDict(
                {
                    "image": image_tensor,
                    "boxes": boxes_tensor,
                    "labels": labels_tensor,
                    "image_id": idx,
                    "image_path": image_path,  # Add image path to the output dict
                    "height": final_h,  # Add final height to the output dict
                    "width": self.target_width # Add final width
                }
            )
        except Exception as e:  # Catch any other unexpected errors in __getitem__
            print(f"UNEXPECTED ERROR in __getitem__ at index {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None  # Return None on any error

    @property
    def num_classes(self) -> int:
        """文字クラスの数を返す"""
        # Ensure unicode_to_idx is initialized before accessing len
        return len(self.unicode_to_idx) if hasattr(self, 'unicode_to_idx') else 0
