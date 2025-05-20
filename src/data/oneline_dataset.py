import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms # Added for default transformations
# Assuming src is in PYTHONPATH, adjust if necessary for your environment
from ..utils.tokenizer import Vocab, PAD_ID, GO_ID, EOS_ID

class OneLineOCRDataset(Dataset):
    def __init__(self, data_root_dir: str, tokenizer: Vocab, transform=None, 
                 max_label_len: int = 256, image_height: int = 64, 
                 image_width: int = 1024, image_channels: int = 1):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.tokenizer = tokenizer
        self.max_label_len = max_label_len
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels # 1 for grayscale, 3 for RGB

        self.image_label_pairs = []

        # File Discovery
        if not os.path.isdir(self.data_root_dir):
            raise FileNotFoundError(f"Data root directory not found: {self.data_root_dir}")

        book_ids = [d for d in os.listdir(self.data_root_dir) if os.path.isdir(os.path.join(self.data_root_dir, d))]

        for book_id in book_ids:
            image_dir = os.path.join(self.data_root_dir, book_id, "images")
            label_dir = os.path.join(self.data_root_dir, book_id, "labels")

            if not os.path.isdir(image_dir):
                # print(f"Warning: Image directory not found for BookID {book_id}, skipping.")
                continue
            if not os.path.isdir(label_dir):
                # print(f"Warning: Label directory not found for BookID {book_id}, skipping.")
                continue

            # Consider common image extensions. Add more if needed.
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            
            for img_path in image_files:
                img_basename = os.path.basename(img_path)
                img_name_no_ext, _ = os.path.splitext(img_basename)
                
                # Assume label file has .txt extension
                label_file_path = os.path.join(label_dir, f"{img_name_no_ext}.txt")

                if os.path.exists(label_file_path):
                    self.image_label_pairs.append((img_path, label_file_path))
                else:
                    # print(f"Warning: Label file not found for image {img_path}, skipping.")
                    pass # Silently skip or log as per requirements

        if not self.image_label_pairs:
            print(f"Warning: No image-label pairs found in {self.data_root_dir}. Check directory structure and file names.")

        # Default transform if none provided
        if transform is None:
            transform_list = [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor()
            ]
            if self.image_channels == 1:
                transform_list.insert(1, transforms.Grayscale(num_output_channels=1))
            # Add normalization if necessary, e.g. transforms.Normalize(mean=[0.5], std=[0.5]) for grayscale
            # For now, keeping it simple. Model might handle normalization internally or expect raw [0,1] tensors.
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        if idx >= len(self.image_label_pairs):
            raise IndexError("Index out of bounds")
            
        image_path, label_file_path = self.image_label_pairs[idx]

        # Load Image
        try:
            if self.image_channels == 1:
                image = Image.open(image_path).convert('L') # Grayscale
            elif self.image_channels == 3:
                image = Image.open(image_path).convert('RGB') # Color
            else:
                # Fallback or error for unsupported channels
                image = Image.open(image_path).convert('L') 
                # print(f"Warning: Unsupported image_channels {self.image_channels}. Defaulting to Grayscale.")

        except FileNotFoundError:
            # print(f"Error: Image file not found at {image_path}. Returning None or dummy data.")
            # This case should ideally not happen if file discovery is robust.
            # For robustness, one might return a dummy tensor or skip.
            # For now, let it raise an error or handle as per specific needs if it occurs.
            raise
        except Exception as e:
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
            with open(label_file_path, 'r', encoding='utf-8') as f:
                label_text = f.read().strip()
        except FileNotFoundError:
            # print(f"Error: Label file not found at {label_file_path}. Using empty label.")
            label_text = "" # Or handle error more strictly
        except Exception as e:
            # print(f"Error reading label file {label_file_path}: {e}. Using empty label.")
            label_text = ""

        # Tokenize and pad label
        # self.tokenizer.encode already adds GO and EOS
        tokenized_label = self.tokenizer.encode(label_text) 
        
        # Pad sequence
        padded_label = torch.full((self.max_label_len,), fill_value=PAD_ID, dtype=torch.long)
        
        # Truncate if tokenized_label is longer than max_label_len
        # Ensure EOS is at the end if truncated, and GO at the start
        # The tokenizer.encode should produce [GO_ID, char_ids..., EOS_ID]
        seq_len = min(len(tokenized_label), self.max_label_len)
        padded_label[:seq_len] = torch.tensor(tokenized_label[:seq_len], dtype=torch.long)

        # If truncation happened before EOS, ensure EOS is the last token.
        if seq_len < self.max_label_len and tokenized_label[seq_len-1] != EOS_ID :
             if seq_len > 0 : #Only add EOS if there is space and it's not already there
                 padded_label[seq_len-1] = EOS_ID # Overwrite last token with EOS if truncated before EOS
             # If seq_len is 0 (empty label), it will be all PAD_IDs, which is fine.
        elif seq_len == self.max_label_len and padded_label[self.max_label_len-1] != EOS_ID:
             # If full and last token is not EOS, force it to be EOS
             padded_label[self.max_label_len-1] = EOS_ID


        label_tensor = padded_label

        # Bounding Boxes (Placeholder)
        # Create dummy bboxes for each potential character slot in the label, excluding SOS.
        # The target sequence for char loss is typically target_ids[:, 1:] (excluding SOS)
        # So, bboxes should correspond to these N-1 tokens.
        dummy_bbox_tensor = torch.zeros((self.max_label_len -1, 4), dtype=torch.float)

        return image_tensor, label_tensor, dummy_bbox_tensor
