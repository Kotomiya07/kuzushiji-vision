import unittest
import torch
from ocr_lightning.model import OCRModel

class TestOCRModel(unittest.TestCase):
    def setUp(self):
        # Define a simple vocabulary for testing
        VOCAB = '<blank>' + 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
        self.idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}
        
        # num_chars is derived by the model from len(char_to_idx)
        # blank_char_idx is also derived by the model

        self.max_boxes = 10 # Max boxes for localization head
        self.learning_rate = 1e-4 # Not used in forward pass tests, but required by model init

        # Initialize model
        self.model = OCRModel(
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            learning_rate=self.learning_rate,
            max_boxes=self.max_boxes
        )
        # Set model to evaluation mode
        self.model.eval()

    def test_model_forward_pass(self):
        # Create a dummy input image tensor
        # Batch size 2, 3 channels (RGB), height 64, width 128
        dummy_images = torch.randn(2, 3, 64, 128) 
        
        with torch.no_grad(): # Disable gradient calculations for inference
            output = self.model(dummy_images)
        
        # Check for 'pred_boxes'
        self.assertIn('pred_boxes', output)
        # Check shape of 'pred_boxes': (batch_size, max_boxes, 4)
        self.assertEqual(output['pred_boxes'].shape, (2, self.max_boxes, 4))
        
        # Check for 'pred_logits'
        self.assertIn('pred_logits', output)
        # Check shape of 'pred_logits': (batch_size, sequence_length, num_chars)
        # Current model uses sequence_length = 1
        # num_chars should be len(self.char_to_idx)
        self.assertEqual(output['pred_logits'].shape, (2, 1, len(self.char_to_idx)))

    def test_model_batch_consistency(self):
        # Create two separate dummy input image tensors, each with batch size 1
        dummy_image1 = torch.randn(1, 3, 64, 128)
        dummy_image2 = torch.randn(1, 3, 64, 128)
        
        with torch.no_grad():
            # Get output for the first image
            output1 = self.model(dummy_image1)
            
            # Get output for the second image
            output2 = self.model(dummy_image2)
            
            # Create a batched input from the two images
            batched_input = torch.cat([dummy_image1, dummy_image2], dim=0)
            
            # Get output for the batched input
            batched_output = self.model(batched_input)
        
        # Check consistency for 'pred_boxes'
        # Compare first element of batched output with output1
        self.assertTrue(torch.allclose(batched_output['pred_boxes'][0], output1['pred_boxes'].squeeze(0), atol=1e-6))
        # Compare second element of batched output with output2
        self.assertTrue(torch.allclose(batched_output['pred_boxes'][1], output2['pred_boxes'].squeeze(0), atol=1e-6))
        
        # Check consistency for 'pred_logits'
        # Compare first element of batched output with output1
        self.assertTrue(torch.allclose(batched_output['pred_logits'][0], output1['pred_logits'].squeeze(0), atol=1e-6))
        # Compare second element of batched output with output2
        self.assertTrue(torch.allclose(batched_output['pred_logits'][1], output2['pred_logits'].squeeze(0), atol=1e-6))

    def test_model_hyperparameters_saved(self):
        # Test if hyperparameters are accessible via self.model.hparams
        self.assertEqual(self.model.hparams.max_boxes, self.max_boxes)
        self.assertEqual(len(self.model.hparams.char_to_idx), len(self.char_to_idx))
        self.assertEqual(self.model.hparams.char_to_idx['a'], self.char_to_idx['a'])
        self.assertEqual(self.model.hparams.idx_to_char[1], self.idx_to_char[1])
        self.assertEqual(self.model.hparams.learning_rate, self.learning_rate)
        self.assertEqual(self.model.hparams.num_chars, len(self.char_to_idx))
        self.assertEqual(self.model.hparams.blank_char_idx, self.char_to_idx['<blank>'])


if __name__ == '__main__':
    unittest.main()
