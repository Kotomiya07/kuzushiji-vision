import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Define a maximum label length (can be adjusted)
MAX_LABEL_LENGTH = 50
# Define default image dimensions (can be adjusted)
DEFAULT_TARGET_HEIGHT = 64
DEFAULT_TARGET_WIDTH = 512


class OCRDataset(Dataset):
    def __init__(self, data_dir, split, image_transform, char_to_int, max_label_length=MAX_LABEL_LENGTH):
        """
        Args:
            data_dir (str): Path to the root data directory (e.g., "data/column_dataset_padded").
            split (str): Data split, one of "train", "val", or "test".
            image_transform (callable): Transformations to apply to the image.
            char_to_int (dict): Dictionary mapping characters to integers.
            max_label_length (int): Maximum length for labels (padding/truncation).
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_transform = image_transform
        self.char_to_int = char_to_int
        self.max_label_length = max_label_length
        self.samples = []

        # Scan for samples
        self._prepare_samples()

    def _prepare_samples(self):
        """
        Scans the data directory to find image, label, and bounding box file triplets.
        Verifies that all corresponding files exist.
        Supports both flat structure and subdirectory structure.
        """
        image_dir = os.path.join(self.data_dir, self.split, "images")
        label_dir = os.path.join(self.data_dir, self.split, "labels")
        bbox_dir = os.path.join(self.data_dir, self.split, "bounding_boxes")

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        if not os.path.isdir(bbox_dir):
            raise FileNotFoundError(f"Bounding box directory not found: {bbox_dir}")

        # Check if we have subdirectories (book_id structure) or flat structure
        image_items = os.listdir(image_dir)
        has_subdirs = any(os.path.isdir(os.path.join(image_dir, item)) for item in image_items)

        if has_subdirs:
            # Handle subdirectory structure: images/book_id/file.jpg
            for book_id in os.listdir(image_dir):
                book_image_dir = os.path.join(image_dir, book_id)
                book_label_dir = os.path.join(label_dir, book_id)
                book_bbox_dir = os.path.join(bbox_dir, book_id)

                if not os.path.isdir(book_image_dir):
                    continue

                for image_filename in os.listdir(book_image_dir):
                    if not image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    base_filename, _ = os.path.splitext(image_filename)

                    label_filename = base_filename + ".txt"
                    bbox_filename = base_filename + ".json"

                    image_path = os.path.join(book_image_dir, image_filename)
                    label_path = os.path.join(book_label_dir, label_filename)
                    bbox_path = os.path.join(book_bbox_dir, bbox_filename)

                    if os.path.exists(label_path) and os.path.exists(bbox_path):
                        self.samples.append({"image_path": image_path, "label_path": label_path, "bbox_path": bbox_path})
                    else:
                        print(f"Warning: Skipping image {base_filename} due to missing label or bbox file.")
        else:
            # Handle flat structure: images/file.jpg
            for image_filename in os.listdir(image_dir):
                if not image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                base_filename, _ = os.path.splitext(image_filename)

                label_filename = base_filename + ".txt"
                bbox_filename = base_filename + ".json"

                image_path = os.path.join(image_dir, image_filename)
                label_path = os.path.join(label_dir, label_filename)
                bbox_path = os.path.join(bbox_dir, bbox_filename)

                if os.path.exists(label_path) and os.path.exists(bbox_path):
                    self.samples.append({"image_path": image_path, "label_path": label_path, "bbox_path": bbox_path})
                else:
                    print(f"Warning: Skipping image {base_filename} due to missing label or bbox file.")

    def __len__(self):
        """Return the total number of verified samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads an image, its label, and bounding boxes, applies transformations,
        and prepares them for the model.
        """
        sample_info = self.samples[idx]

        # Load image
        image = Image.open(sample_info["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # Load label
        with open(sample_info["label_path"], encoding="utf-8") as f:
            label_text = f.read().strip()

        actual_label_len = len(label_text)
        # Cap actual_label_len at max_label_length, as the label itself will be truncated.
        # The CTC loss and other downstream processes should not consider characters beyond this limit.
        if actual_label_len > self.max_label_length:
            actual_label_len = self.max_label_length

        # Convert label to integers and pad/truncate
        encoded_label = [
            self.char_to_int.get(char, self.char_to_int["<UNK>"]) for char in label_text[:actual_label_len]
        ]  # Use capped length

        # Pad label
        if len(encoded_label) < self.max_label_length:  # Pad up to max_label_length
            encoded_label.extend([self.char_to_int["<PAD>"]] * (self.max_label_length - len(encoded_label)))
        # No else needed for truncation here as label_text was sliced by actual_label_len (which is <= max_label_length)
        # and encoded_label is built from that.

        label_tensor = torch.tensor(encoded_label, dtype=torch.long)

        # Load bounding boxes
        with open(sample_info["bbox_path"]) as f:
            bboxes_data = json.load(f)  # Assuming format is [[x_min, y_min, x_max, y_max], ...]

        # Convert bounding boxes to tensor and pad/truncate
        # Each bbox is [x_min, y_min, x_max, y_max]
        # Pad with [0,0,0,0] if fewer boxes than max_label_length
        # Truncate if more boxes than max_label_length (aligned with label truncation)
        processed_bboxes = []
        # Use actual_label_len (already capped at max_label_length) for bbox alignment
        # This ensures that we don't try to access bboxes beyond the (potentially truncated) label length.

        for i in range(self.max_label_length):
            if i < actual_label_len and i < len(bboxes_data):
                # Ensure bboxes_data has an entry for character i
                processed_bboxes.append(bboxes_data[i])
            else:
                processed_bboxes.append([0, 0, 0, 0])  # Padding box

        bbox_tensor = torch.tensor(processed_bboxes, dtype=torch.float32)

        return {
            "image": image,
            "label": label_tensor,
            "bounding_boxes": bbox_tensor,
            "label_lengths": torch.tensor(actual_label_len, dtype=torch.long),  # Add actual_label_len
        }


def build_char_to_int_map(data_dir):
    """
    Builds character-to-integer and integer-to-character mappings from label files.
    Supports both flat structure and subdirectory structure.

    Args:
        data_dir (str): Path to the root data directory.

    Returns:
        tuple: (char_to_int, int_to_char) dictionaries.
    """
    unique_chars = set()
    splits = ["train", "val", "test"]

    for split in splits:
        label_dir = os.path.join(data_dir, split, "labels")
        if not os.path.isdir(label_dir):
            print(f"Warning: Label directory not found for split '{split}': {label_dir}")
            continue

        # Check if we have subdirectories (book_id structure) or flat structure
        label_items = os.listdir(label_dir)
        has_subdirs = any(os.path.isdir(os.path.join(label_dir, item)) for item in label_items)

        if has_subdirs:
            # Handle subdirectory structure: labels/book_id/file.txt
            for book_id in os.listdir(label_dir):
                book_label_dir = os.path.join(label_dir, book_id)
                if not os.path.isdir(book_label_dir):
                    continue

                for label_filename in os.listdir(book_label_dir):
                    if label_filename.endswith(".txt"):
                        label_path = os.path.join(book_label_dir, label_filename)
                        try:
                            with open(label_path, encoding="utf-8") as f:
                                unique_chars.update(f.read().strip())
                        except Exception as e:
                            print(f"Warning: Could not read label file {label_path}: {e}")
        else:
            # Handle flat structure: labels/file.txt
            for label_filename in os.listdir(label_dir):
                if label_filename.endswith(".txt"):
                    label_path = os.path.join(label_dir, label_filename)
                    try:
                        with open(label_path, encoding="utf-8") as f:
                            unique_chars.update(f.read().strip())
                    except Exception as e:
                        print(f"Warning: Could not read label file {label_path}: {e}")

    # Create mappings
    char_to_int = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
    # Start other characters from index 2
    for i, char in enumerate(sorted(unique_chars), start=2):
        char_to_int[char] = i

    int_to_char = {i: char for char, i in char_to_int.items()}

    return char_to_int, int_to_char


def get_data_loader(
    data_dir,
    split,
    batch_size,
    num_workers,
    char_to_int,
    max_label_length=MAX_LABEL_LENGTH,
    shuffle=None,
    target_height=DEFAULT_TARGET_HEIGHT,
    target_width=DEFAULT_TARGET_WIDTH,
):
    """
    Creates a PyTorch DataLoader for the OCR dataset.

    Args:
        data_dir (str): Path to the root data directory.
        split (str): Data split ("train", "val", or "test").
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        char_to_int (dict): Dictionary mapping characters to integers.
        max_label_length (int): Maximum length for labels.
        shuffle (bool, optional): Whether to shuffle data. Defaults to True for train, False otherwise.
        target_height (int): Target height for image resizing.
        target_width (int): Target width for image resizing.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    if shuffle is None:
        shuffle = split == "train"

    # Define image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = OCRDataset(
        data_dir=data_dir,
        split=split,
        image_transform=image_transform,
        char_to_int=char_to_int,
        max_label_length=max_label_length,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Recommended for performance with CUDA
    )

    return data_loader


if __name__ == "__main__":
    # Example Usage (assuming you have a dataset in 'data/column_dataset_padded')
    # Create dummy data for testing if it doesn't exist
    EXAMPLE_DATA_DIR = "data/column_dataset_padded_example"  # Use a different dir for example

    def create_dummy_data(base_dir, splits=None):
        if splits is None:
            splits = ["train", "val", "test"]
        if os.path.exists(base_dir):
            print(f"Dummy data directory {base_dir} already exists. Skipping creation.")
            # return # If you want to avoid re-creation, uncomment this

        print(f"Creating dummy data in {base_dir}...")
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"

        for split in splits:
            img_dir = os.path.join(base_dir, split, "images")
            lbl_dir = os.path.join(base_dir, split, "labels")
            bbox_dir = os.path.join(base_dir, split, "bounding_boxes")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)

            for i in range(5):  # Create 5 dummy samples per split
                img_filename = f"sample_{i}.png"
                txt_filename = f"sample_{i}.txt"
                json_filename = f"sample_{i}.json"

                # Create a dummy image (e.g., a 100x30 black image)
                try:
                    dummy_img = Image.new("RGB", (100, 30), color="black")
                    dummy_img.save(os.path.join(img_dir, img_filename))
                except Exception as e:
                    print(f"Could not create dummy image {img_filename}: {e}")
                    continue

                # Create a dummy label
                label_text = "".join(chars[j % len(chars)] for j in range(10))  # Dummy text
                with open(os.path.join(lbl_dir, txt_filename), "w") as f:
                    f.write(label_text)

                # Create dummy bounding boxes
                # [[x_min, y_min, x_max, y_max], ...]
                bboxes = [[j * 10, 5, (j * 10) + 8, 25] for j in range(len(label_text))]
                with open(os.path.join(bbox_dir, json_filename), "w") as f:
                    json.dump(bboxes, f)
        print(f"Dummy data creation finished in {base_dir}.")

    # Create dummy data if it doesn't exist (for local testing)
    # Important: This example creates files. In a real environment, data would pre-exist.
    create_dummy_data(EXAMPLE_DATA_DIR)

    print(f"Building char map from: {EXAMPLE_DATA_DIR}")
    char_to_int_map, int_to_char_map = build_char_to_int_map(EXAMPLE_DATA_DIR)
    print("Character to Integer Mapping:", char_to_int_map)
    print("Integer to Character Mapping:", int_to_char_map)

    if not char_to_int_map or len(char_to_int_map) <= 2:  # Only <PAD> and <UNK>
        print(
            "Warning: char_to_int_map is empty or only contains special tokens. "
            "This might happen if dummy data was not created or is empty."
        )
        # Fallback for char_to_int_map if dummy data creation failed or was skipped and no real data
        if not char_to_int_map or len(char_to_int_map) <= 2:
            print("Using a fallback char_to_int_map for loader demonstration.")
            char_to_int_map = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "c": 4}  # Minimal example

    print(f"\nGetting data loader for 'train' split from: {EXAMPLE_DATA_DIR}")
    try:
        train_loader = get_data_loader(
            data_dir=EXAMPLE_DATA_DIR,
            split="train",
            batch_size=2,
            num_workers=0,  # Set to 0 for easier debugging in main, >0 for performance
            char_to_int=char_to_int_map,
            max_label_length=MAX_LABEL_LENGTH,
        )

        print(f"Number of batches in train_loader: {len(train_loader)}")

        # Iterate over a few batches to test
        for i, batch in enumerate(train_loader):
            print(f"\nBatch {i + 1}:")
            print("Image shape:", batch["image"].shape)  # Expected: [batch_size, C, H, W]
            print("Label shape:", batch["label"].shape)  # Expected: [batch_size, max_label_length]
            print("Bounding boxes shape:", batch["bounding_boxes"].shape)  # Expected: [batch_size, max_label_length, 4]
            print("Label lengths:", batch["label_lengths"])  # Expected: [batch_size]

            print("Sample Label (int):", batch["label"][0])
            print("Sample Label length:", batch["label_lengths"][0])
            print("Sample BBoxes (first 5):", batch["bounding_boxes"][0][:5])
            if i >= 1:  # Show 2 batches
                break

        if len(train_loader) == 0:
            print("Train loader is empty. This might be due to issues with data paths or OCRDataset.")

    except FileNotFoundError as e:
        print(f"Error during DataLoader creation or iteration: {e}")
        print(
            "Please ensure that the EXAMPLE_DATA_DIR path is correct and contains the expected subdirectories "
            "(train/images, train/labels, train/bounding_boxes)."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nData loader script finished.")
