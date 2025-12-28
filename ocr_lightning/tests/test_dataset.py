import unittest
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile
from pathlib import Path
import json
import shutil
import os

from ocr_lightning.dataset import OcrDataset, ocr_collate_fn

class TestOcrDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "tmp_data_dir"
        
        # Create directory structure
        self.images_base_dir = self.data_dir / "images" / "book1"
        self.labels_base_dir = self.data_dir / "labels" / "book1"
        self.bboxes_base_dir = self.data_dir / "bounding_boxes" / "book1"

        os.makedirs(self.images_base_dir)
        os.makedirs(self.labels_base_dir)
        os.makedirs(self.bboxes_base_dir)

        # Dummy image dimensions
        self.img_width = 32
        self.img_height = 32

        # Create dummy files
        self.sample_names = ["img1", "img2"]
        self.dummy_labels = ["test1", "test2"]
        self.dummy_bboxes = [[[0,0,5,5], [1,1,6,6]], [[10,10,15,15]]] # img1 has 2 boxes, img2 has 1

        for i, name in enumerate(self.sample_names):
            # Dummy image
            img_path = self.images_base_dir / f"{name}.png"
            Image.new('RGB', (self.img_width, self.img_height), color = 'black').save(img_path, format='PNG')
            
            # Dummy label
            label_path = self.labels_base_dir / f"{name}.txt"
            with open(label_path, 'w') as f:
                f.write(self.dummy_labels[i])
            
            # Dummy bounding box
            bbox_path = self.bboxes_base_dir / f"{name}.json"
            with open(bbox_path, 'w') as f:
                json.dump(self.dummy_bboxes[i], f)

        self.char_to_idx = {'<blank>':0, 't':1, 'e':2, 's':3, '1':4, '2':5, ' ':6} # Added space for collate test
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_loading(self):
        # OcrDataset applies ToTensor by default if image_transforms is None
        dataset = OcrDataset(data_split_dir=self.data_dir, char_to_idx=self.char_to_idx, image_transforms=None)
        self.assertEqual(len(dataset), 2)
        
        sample0 = dataset[0]
        self.assertIsInstance(sample0['image'], torch.Tensor)
        self.assertEqual(sample0['image'].shape, (3, self.img_height, self.img_width)) # C, H, W after ToTensor
        self.assertEqual(sample0['label_text'], self.dummy_labels[0])
        self.assertEqual(sample0['bounding_boxes'], self.dummy_bboxes[0])
        expected_path0 = str(self.images_base_dir / f"{self.sample_names[0]}.png")
        self.assertEqual(str(sample0['image_path']), expected_path0)

        sample1 = dataset[1]
        self.assertIsInstance(sample1['image'], torch.Tensor)
        self.assertEqual(sample1['image'].shape, (3, self.img_height, self.img_width))
        self.assertEqual(sample1['label_text'], self.dummy_labels[1])
        self.assertEqual(sample1['bounding_boxes'], self.dummy_bboxes[1])
        expected_path1 = str(self.images_base_dir / f"{self.sample_names[1]}.png")
        self.assertEqual(str(sample1['image_path']), expected_path1)


    def test_dataset_file_missing_label(self):
        # Remove one label file
        os.remove(self.labels_base_dir / f"{self.sample_names[1]}.txt")
        
        # Suppress print warnings during test
        # Here we are testing the skipping behavior, so warnings are expected.
        # A more sophisticated way might involve redirecting stdout.
        dataset = OcrDataset(data_split_dir=self.data_dir, char_to_idx=self.char_to_idx)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['label_text'], self.dummy_labels[0])

    def test_dataset_file_missing_bbox(self):
        # Remove one bbox file
        os.remove(self.bboxes_base_dir / f"{self.sample_names[0]}.json")
        dataset = OcrDataset(data_split_dir=self.data_dir, char_to_idx=self.char_to_idx)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['label_text'], self.dummy_labels[1])
        
    def test_dataset_file_missing_image_folder_level(self):
        # Create a new structure where an image is missing a corresponding label/bbox dir
        images_book2_dir = self.data_dir / "images" / "book2"
        os.makedirs(images_book2_dir)
        Image.new('RGB', (self.img_width, self.img_height), color = 'blue').save(images_book2_dir / "img3.png", format='PNG')
        
        dataset = OcrDataset(data_split_dir=self.data_dir, char_to_idx=self.char_to_idx)
        self.assertEqual(len(dataset), 2) # Should still load the 2 valid samples from book1

    def test_collate_fn(self):
        # Using image_transforms=None, so OcrDataset will apply ToTensor
        dataset = OcrDataset(data_split_dir=self.data_dir, char_to_idx=self.char_to_idx, image_transforms=None)
        samples = [dataset[0], dataset[1]]
        
        batch = ocr_collate_fn(samples)
        
        # Images
        self.assertIsInstance(batch['images'], torch.Tensor)
        # Assuming all dummy images are same size, so no padding needed for H, W
        self.assertEqual(batch['images'].shape, (2, 3, self.img_height, self.img_width))
        
        # Labels
        self.assertIsInstance(batch['label_texts'], list)
        self.assertEqual(len(batch['label_texts']), 2)
        self.assertEqual(batch['label_texts'][0], self.dummy_labels[0])
        self.assertEqual(batch['label_texts'][1], self.dummy_labels[1])
        
        # Bounding boxes
        self.assertIsInstance(batch['bounding_boxes_batch'], torch.Tensor)
        # Max boxes between the two samples: sample0 has 2, sample1 has 1. So max_boxes_in_batch = 2
        # Shape should be (batch_size, max_boxes_in_batch, 4)
        self.assertEqual(batch['bounding_boxes_batch'].shape, (2, 2, 4)) 
        
        # Check padding for the second sample's bboxes
        expected_bbox_sample0 = torch.tensor(self.dummy_bboxes[0], dtype=torch.float32)
        self.assertTrue(torch.equal(batch['bounding_boxes_batch'][0], expected_bbox_sample0))
        
        expected_bbox_sample1_padded = list(self.dummy_bboxes[1]) # copy
        expected_bbox_sample1_padded.append([-1,-1,-1,-1]) # Add one dummy box
        self.assertTrue(torch.equal(batch['bounding_boxes_batch'][1], torch.tensor(expected_bbox_sample1_padded, dtype=torch.float32)))

        # Bbox counts
        self.assertIsInstance(batch['bbox_counts'], list)
        self.assertEqual(batch['bbox_counts'], [len(self.dummy_bboxes[0]), len(self.dummy_bboxes[1])])

        # Target lengths (based on label_texts)
        self.assertIsInstance(batch['target_lengths'], list)
        self.assertEqual(batch['target_lengths'], [len(self.dummy_labels[0]), len(self.dummy_labels[1])])

        # Image paths
        self.assertIsInstance(batch['image_paths'], list)
        self.assertEqual(len(batch['image_paths']), 2)


if __name__ == '__main__':
    unittest.main()
