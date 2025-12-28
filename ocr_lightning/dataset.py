import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms # Added for ToTensor

class OcrDataset(Dataset):
    def __init__(self, data_split_dir, image_transforms=None, char_to_idx=None):
        self.data_split_dir = data_split_dir
        self.image_transforms = image_transforms
        self.char_to_idx = char_to_idx
        self.file_samples = []

        images_dir = os.path.join(self.data_split_dir, 'images')
        labels_dir = os.path.join(self.data_split_dir, 'labels')
        bounding_boxes_dir = os.path.join(self.data_split_dir, 'bounding_boxes')

        for book_id in os.listdir(images_dir):
            book_images_dir = os.path.join(images_dir, book_id)
            if not os.path.isdir(book_images_dir):
                continue

            for image_name in os.listdir(book_images_dir):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(book_images_dir, image_name)
                
                base_name = os.path.splitext(image_name)[0]
                label_file_name = f"{base_name}.txt"
                bbox_file_name = f"{base_name}.json"

                label_path = os.path.join(labels_dir, book_id, label_file_name)
                bbox_path = os.path.join(bounding_boxes_dir, book_id, bbox_file_name)

                if not os.path.exists(label_path):
                    print(f"Warning: Label file not found for image {image_path}, skipping sample.")
                    continue
                
                if not os.path.exists(bbox_path):
                    print(f"Warning: Bounding box file not found for image {image_path}, skipping sample.")
                    continue
                
                self.file_samples.append((image_path, label_path, bbox_path))

    def __len__(self):
        return len(self.file_samples)

    def __getitem__(self, idx):
        image_path, label_path, bbox_path = self.file_samples[idx]

        image = Image.open(image_path).convert('RGB')
        
        with open(label_path, 'r', encoding='utf-8') as f:
            label_text = f.read().strip()
        
        with open(bbox_path, 'r', encoding='utf-8') as f:
            bounding_boxes = json.load(f)

        if self.image_transforms:
            image = self.image_transforms(image)
        else:
            image = transforms.ToTensor()(image) # Apply ToTensor if no transforms are provided
        
        # Character encoding will be handled in a future step
        # if self.char_to_idx:
        #     pass

        return {
            'image': image,
            'label_text': label_text,
            'bounding_boxes': bounding_boxes,
            'image_path': image_path  # Added for collate_fn and debugging
        }

import torch

def ocr_collate_fn(batch):
    images = [item['image'] for item in batch]
    label_texts = [item['label_text'] for item in batch]
    bounding_boxes_list = [item['bounding_boxes'] for item in batch]
    image_paths = [item['image_path'] for item in batch] 

    # Pad images (assuming images are C x H x W tensors)
    max_h = 0
    max_w = 0
    for img in images:
        if img.shape[1] > max_h: # C, H, W
            max_h = img.shape[1]
        if img.shape[2] > max_w:
            max_w = img.shape[2]

    padded_images = []
    for img in images:
        c, h, w = img.shape
        # padding: (padding_left, padding_right, padding_top, padding_bottom)
        padding = (0, max_w - w, 0, max_h - h) 
        padded_img = torch.nn.functional.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
    
    images_tensor = torch.stack(padded_images)

    # Pad bounding boxes
    bbox_counts = [len(bboxes) for bboxes in bounding_boxes_list]
    max_bboxes = max(bbox_counts) if bbox_counts else 0 # Handle empty batch or no bboxes
    
    padded_bounding_boxes = []
    # Using [-1,-1,-1,-1] as dummy box, assuming valid coords are non-negative
    dummy_box = [-1, -1, -1, -1] 

    for bboxes in bounding_boxes_list:
        # Create a copy before extending, as item['bounding_boxes'] might be a reference
        current_boxes = list(bboxes) 
        while len(current_boxes) < max_bboxes:
            current_boxes.append(dummy_box)
        padded_bounding_boxes.append(current_boxes)

    bounding_boxes_tensor = torch.tensor(padded_bounding_boxes, dtype=torch.float32)

    # Target lengths (can be len of text or actual encoded target lengths later)
    target_lengths = [len(text) for text in label_texts]

    return {
        'images': images_tensor,
        'label_texts': label_texts, 
        'bounding_boxes_batch': bounding_boxes_tensor,
        'target_lengths': target_lengths,
        'bbox_counts': bbox_counts,
        'image_paths': image_paths 
    }
