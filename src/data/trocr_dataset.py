import ast
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ResizeWithPadding:
    """Resize image while preserving aspect ratio and pad to target size"""

    def __init__(self, target_size: tuple[int, int], fill_color: tuple[int, int, int] = (255, 255, 255)):
        """
        Args:
            target_size: (height, width) target size
            fill_color: RGB color for padding (default: white)
        """
        self.target_height, self.target_width = target_size
        self.fill_color = fill_color

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Resize image while preserving aspect ratio and pad to target size
        """
        # Get original dimensions
        orig_width, orig_height = image.size

        # Calculate scaling factor to fit within target size
        scale_w = self.target_width / orig_width
        scale_h = self.target_height / orig_height
        scale = min(scale_w, scale_h)

        # Calculate new size
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and fill color
        new_image = Image.new("RGB", (self.target_width, self.target_height), self.fill_color)

        # Calculate position to paste resized image (center)
        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2

        # Paste resized image onto new image
        new_image.paste(image, (paste_x, paste_y))

        return new_image


class TrOCRDataset(Dataset):
    """Dataset for TrOCR training with column images and text annotations"""

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        tokenizer_path: str,
        image_size: tuple[int, int] = (1024, 64),  # Default to column image size
        max_length: int = 128,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Args:
            csv_path: Path to column_info.csv
            image_root: Root directory containing column images
            tokenizer_path: Path to tokenizer
            image_size: Target image size (height, width)
            max_length: Maximum sequence length for tokenization
            split: "train", "val", or "test"
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.image_size = image_size
        self.max_length = max_length
        self.split = split

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load and process data
        self.data = self._load_data()
        self.data = self._split_data(train_ratio, val_ratio, test_ratio)

        # Define image transforms with aspect ratio preservation
        self.transform = transforms.Compose(
            [
                # Resize while preserving aspect ratio and pad to exact size
                ResizeWithPadding(target_size=image_size, fill_color=(255, 255, 255)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # RGB normalization
            ]
        )

    def _load_data(self) -> pd.DataFrame:
        """Load and process the CSV data"""
        df = pd.read_csv(self.csv_path)

        # Filter out rows with missing data
        df = df.dropna()

        # Convert unicode_ids string to list
        df["unicode_ids"] = df["unicode_ids"].apply(ast.literal_eval)

        # Convert Unicode IDs to text
        df["text"] = df["unicode_ids"].apply(self._unicode_ids_to_text)

        # Filter out empty texts
        df = df[df["text"].str.len() > 0]

        # Create full image paths
        def create_full_path(column_image_path):
            if column_image_path.startswith("/"):
                return column_image_path
            elif column_image_path.startswith("processed_v2/"):
                # Replace processed_v2/ with data/processed_v2/
                return column_image_path.replace("processed_v2/", "data/processed_v2/", 1)
            else:
                return os.path.join(os.getcwd(), column_image_path)

        df["full_image_path"] = df["column_image"].apply(create_full_path)

        # Filter out non-existent images
        df = df[df["full_image_path"].apply(os.path.exists)]

        return df.reset_index(drop=True)

    def _unicode_ids_to_text(self, unicode_ids: list[str]) -> str:
        """Convert list of Unicode IDs to text"""
        text = ""
        for unicode_id in unicode_ids:
            if unicode_id.startswith("U+"):
                try:
                    # Convert Unicode ID to character
                    char_code = int(unicode_id[2:], 16)
                    char = chr(char_code)
                    text += char
                except ValueError:
                    continue
        return text

    def _split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.DataFrame:
        """Split data into train/val/test"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        total_samples = len(self.data)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        if self.split == "train":
            return self.data[:train_end]
        elif self.split == "val":
            return self.data[train_end:val_end]
        elif self.split == "test":
            return self.data[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample"""
        row = self.data.iloc[idx]

        # Load and process image
        image_path = row["full_image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            pixel_values = torch.zeros(3, *self.image_size)

        # Process text
        text = row["text"]

        # Tokenize text
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Prepare labels (input_ids shifted for causal LM)
        labels = encoding["input_ids"].squeeze(0)  # Remove batch dimension

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "image_path": image_path,
        }


def create_trocr_dataloaders(
    csv_path: str,
    image_root: str,
    tokenizer_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: tuple[int, int] = (1024, 64),  # Default to column image size
    max_length: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders"""

    # Create datasets
    train_dataset = TrOCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer_path=tokenizer_path,
        image_size=image_size,
        max_length=max_length,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    val_dataset = TrOCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer_path=tokenizer_path,
        image_size=image_size,
        max_length=max_length,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    test_dataset = TrOCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer_path=tokenizer_path,
        image_size=image_size,
        max_length=max_length,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    csv_path = "data/processed_v2/column_info.csv"
    image_root = "data/processed_v2/column_images"
    tokenizer_path = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-200000"

    dataset = TrOCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer_path=tokenizer_path,
        image_size=(1024, 64),  # Column images are tall and narrow
        split="train",
    )

    print(f"Dataset size: {len(dataset)}")

    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['pixel_values'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Text: {sample['text']}")
