import glob
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class OneLineOCRDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        tokenizer: AutoTokenizer,
        transform=None,
        max_label_len: int = 256,
        image_height: int = 64,
        image_width: int = 1024,
        image_channels: int = 1,
    ):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.tokenizer = tokenizer
        self.max_label_len = max_label_len
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels  # 1 for grayscale, 3 for RGB

        self.image_label_bbox_triplets = []

        # File Discovery
        if not os.path.isdir(self.data_root_dir):
            raise FileNotFoundError(f"Data root directory not found: {self.data_root_dir}")

        # Determine the dataset type (train/val/test) from the data_root_dir path
        # This assumes data_root_dir might be like ".../train", ".../val", or ".../test"
        # or that a subdirectory like "train" exists directly under data_root_dir
        # For this implementation, we'll assume the "images" and "labels" subdirectories
        # are directly under data_root_dir, and "bounding_boxes" is parallel to "images" and "labels"
        # but potentially under a {train/val/test} parent if data_root_dir points to that level.

        # The user specified path like data/column_dataset_padded/{train/va/test}/bounding_boxes
        # This implies that data_root_dir might be data/column_dataset_padded/train (or val/test)
        # So, images are in data_root_dir/images, labels in data_root_dir/labels
        # and bboxes are in data_root_dir/bounding_boxes

        images_base_dir = os.path.join(self.data_root_dir, "images")
        labels_base_dir = os.path.join(self.data_root_dir, "labels")
        bboxes_base_dir = os.path.join(self.data_root_dir, "bounding_boxes")

        if not os.path.isdir(images_base_dir):
            # If images_base_dir itself doesn't exist, we can't proceed for this data_root_dir
            print(f"Warning: Images base directory not found: {images_base_dir}, skipping this data root.")
            # Depending on strictness, could raise error or return.
            # For now, this will lead to an empty dataset if no books are found.
            book_ids = []
        else:
            book_ids = [d for d in os.listdir(images_base_dir) if os.path.isdir(os.path.join(images_base_dir, d))]

        for book_id in book_ids:
            image_dir = os.path.join(images_base_dir, book_id)
            label_dir = os.path.join(labels_base_dir, book_id)
            bbox_dir = os.path.join(bboxes_base_dir, book_id)  # Bounding box directory for the book

            # Check for image_dir, label_dir is already done. Add check for bbox_dir if strictness requires.
            # For now, if label_dir or image_dir is missing, we skip. Bbox is optional.

            if not os.path.isdir(image_dir):
                # print(f"Warning: Image directory not found for BookID {book_id}, skipping.")
                continue
            if not os.path.isdir(label_dir):  # Labels are essential
                # print(f"Warning: Label directory not found for BookID {book_id}, skipping.")
                continue

            # Bbox dir is optional, so we don't skip if it's missing.
            # We'll check for individual bbox files later.

            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(image_dir, ext)))

            for img_path in image_files:
                img_basename = os.path.basename(img_path)
                img_name_no_ext, _ = os.path.splitext(img_basename)

                label_file_path = os.path.join(label_dir, f"{img_name_no_ext}.txt")
                bbox_file_path = os.path.join(bbox_dir, f"{img_name_no_ext}.json")  # Bbox JSON path

                if os.path.exists(label_file_path):  # Only proceed if label exists
                    # Check if bbox file exists, store path or None
                    actual_bbox_path = bbox_file_path if os.path.exists(bbox_file_path) else None
                    self.image_label_bbox_triplets.append((img_path, label_file_path, actual_bbox_path))
                else:
                    # print(f"Warning: Label file not found for image {img_path}, skipping.")
                    pass

        if not self.image_label_bbox_triplets:
            print(
                f"Warning: No image-label-bbox triplets found in {self.data_root_dir}. Check directory structure and file names."
            )

        # Default transform if none provided
        if transform is None:
            transform_list = [transforms.Resize((self.image_height, self.image_width)), transforms.ToTensor()]
            if self.image_channels == 1:
                transform_list.insert(1, transforms.Grayscale(num_output_channels=1))
            # Add normalization if necessary, e.g. transforms.Normalize(mean=[0.5], std=[0.5]) for grayscale
            # For now, keeping it simple. Model might handle normalization internally or expect raw [0,1] tensors.
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_label_bbox_triplets)

    def __getitem__(self, idx):
        if idx >= len(self.image_label_bbox_triplets):
            raise IndexError("Index out of bounds")

        image_path, label_file_path, bbox_file_path = self.image_label_bbox_triplets[idx]

        # Load Image
        try:
            if self.image_channels == 1:
                image = Image.open(image_path).convert("L")  # Grayscale
            elif self.image_channels == 3:
                image = Image.open(image_path).convert("RGB")  # Color
            else:
                # Fallback or error for unsupported channels
                image = Image.open(image_path).convert("L")
                # print(f"Warning: Unsupported image_channels {self.image_channels}. Defaulting to Grayscale.")

        except FileNotFoundError:
            # print(f"Error: Image file not found at {image_path}. Returning None or dummy data.")
            # This case should ideally not happen if file discovery is robust.
            # For robustness, one might return a dummy tensor or skip.
            # For now, let it raise an error or handle as per specific needs if it occurs.
            raise
        except Exception:
            # print(f"Error loading image {image_path}: {e}")
            # Handle other PIL errors
            raise

        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Basic conversion if no transform is specified (should be rare with default)
            image_tensor = transforms.ToTensor()(image)

        # Load Label
        try:
            with open(label_file_path, encoding="utf-8") as f:
                label_text = f.read().strip()
        except FileNotFoundError:
            # print(f"Error: Label file not found at {label_file_path}. Using empty label.")
            label_text = ""  # Or handle error more strictly
        except Exception:
            # print(f"Error reading label file {label_file_path}: {e}. Using empty label.")
            label_text = ""

        # Tokenize and pad label
        # self.tokenizer.encode already adds GO and EOS
        tokenized_label = self.tokenizer.encode(label_text)

        # Pad sequence
        padded_label = torch.full((self.max_label_len,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)

        # Truncate if tokenized_label is longer than max_label_len
        # Ensure EOS is at the end if truncated, and GO at the start
        # The tokenizer.encode should produce [GO_ID, char_ids..., EOS_ID]
        seq_len = min(len(tokenized_label), self.max_label_len)
        padded_label[:seq_len] = torch.tensor(tokenized_label[:seq_len], dtype=torch.long)

        # If truncation happened before EOS, ensure EOS is the last token.
        if seq_len < self.max_label_len and tokenized_label[seq_len - 1] != self.tokenizer.eos_token_id:
            if seq_len > 0:  # Only add EOS if there is space and it's not already there
                padded_label[seq_len - 1] = self.tokenizer.eos_token_id  # Overwrite last token with EOS if truncated before EOS
            # If seq_len is 0 (empty label), it will be all PAD_IDs, which is fine.
        elif seq_len == self.max_label_len and padded_label[self.max_label_len - 1] != self.tokenizer.eos_token_id:
            # If full and last token is not EOS, force it to be EOS
            padded_label[self.max_label_len - 1] = self.tokenizer.eos_token_id

        label_tensor = padded_label

        # Bounding Boxes
        target_bbox_len = self.max_label_len - 1  # Corresponds to label_tensor[1:]
        bbox_list = []

        if bbox_file_path and os.path.exists(bbox_file_path):
            try:
                with open(bbox_file_path, encoding="utf-8") as f:
                    loaded_bboxes = json.load(f)
                # Assuming loaded_bboxes is a list of lists, e.g., [[x1,y1,x2,y2], ...]
                # Or it could be a list of dicts, e.g. [{'box': [x1,y1,x2,y2], 'char': 'a'}, ...]
                # For now, assume it's a simple list of bbox coordinates.
                # We need to ensure each bbox has 4 coordinates.
                for bbox in loaded_bboxes:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox_list.append([float(c) for c in bbox])
                    # else: print(f"Warning: Malformed bbox entry in {bbox_file_path}: {bbox}")

            except json.JSONDecodeError:
                # print(f"Error: Could not decode JSON from {bbox_file_path}. Using dummy bboxes.")
                pass  # Handled by creating dummy bboxes later if bbox_list is empty
            except Exception:
                # print(f"Error loading or processing bbox file {bbox_file_path}: {e}. Using dummy bboxes.")
                pass  # Handled by creating dummy bboxes later

        # Pad or truncate bbox_list to target_bbox_len
        processed_bboxes = bbox_list[:target_bbox_len]
        # Pad with default bbox [0,0,0,0] if shorter
        while len(processed_bboxes) < target_bbox_len:
            processed_bboxes.append([0.0, 0.0, 0.0, 0.0])

        bbox_tensor = torch.tensor(processed_bboxes, dtype=torch.float)
        # Ensure the tensor shape is correct even if processed_bboxes was empty
        if bbox_tensor.shape[0] != target_bbox_len or (target_bbox_len > 0 and bbox_tensor.shape[1] != 4):
            # This case handles if processed_bboxes became empty or malformed leading to wrong shape
            # print(f"Warning: bbox_tensor shape mismatch for {image_path}. Resetting to zeros.")
            bbox_tensor = torch.zeros((target_bbox_len, 4), dtype=torch.float)
        # If target_bbox_len is 0 (e.g. max_label_len is 1), create an empty tensor correctly
        if target_bbox_len == 0:
            bbox_tensor = torch.empty((0, 4), dtype=torch.float)

        return image_tensor, label_tensor, bbox_tensor
