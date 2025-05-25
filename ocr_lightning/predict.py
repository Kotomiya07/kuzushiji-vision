import torch
import pytorch_lightning as pl
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms

from ocr_lightning.dataset import OcrDataset, ocr_collate_fn
from ocr_lightning.model import OCRModel

# Define Character Set (Placeholder - should match training)
# This should ideally be loaded from model hparams or a shared config.
# For now, define it as in train.py for consistency if hparams are not fully populated.
DEFAULT_VOCAB = '<blank>' + 'abcdefghijklmnopqrstuvwxyz0123456789' + '帝都書肆尚書堂梓 .,:;!?\'"`-()'
DEFAULT_CHAR_TO_IDX = {char: idx for idx, char in enumerate(DEFAULT_VOCAB)}
DEFAULT_IDX_TO_CHAR = {idx: char for idx, char in enumerate(DEFAULT_VOCAB)}
DEFAULT_BLANK_CHAR_IDX = DEFAULT_CHAR_TO_IDX.get('<blank>', 0)


def decode_ctc_output(logits, idx_to_char, blank_idx):
    """
    Decodes CTC output.
    Input logits: tensor of shape (sequence_length, num_classes)
    Input idx_to_char: mapping from character indices to characters.
    Input blank_idx: index of the blank character.
    """
    # The current model has seq_len=1, so logits is (1, num_classes)
    if logits.ndim == 2 and logits.size(0) == 1: 
        pred_indices = torch.argmax(logits, dim=1)
        char_idx = pred_indices[0].item()
        if char_idx == blank_idx:
            return ""
        return idx_to_char.get(char_idx, "") # Return char or empty string if not found
    
    # Standard CTC decode for potentially longer sequences in future models
    pred_indices = torch.argmax(logits, dim=1)
    decoded_sequence = []
    last_char_idx = None
    for char_idx_tensor in pred_indices:
        char_idx = char_idx_tensor.item()
        if char_idx == blank_idx:
            last_char_idx = None
            continue
        if char_idx == last_char_idx: # Collapse repeated characters not separated by blank
            continue
        
        character = idx_to_char.get(char_idx)
        if character:
            decoded_sequence.append(character)
        last_char_idx = char_idx # Update last_char_idx to current char_idx
            
    return "".join(decoded_sequence)


def predict_single_image(model, image_path, image_transforms_func, idx_to_char, blank_idx, device):
    model.eval()
    
    try:
        original_image_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None, None


    if image_transforms_func:
        image_tensor = image_transforms_func(original_image_pil)
    else:
        image_tensor = transforms.ToTensor()(original_image_pil) # Default if none provided

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_boxes_single = output['pred_boxes']    # (1, max_boxes, 4)
        pred_logits_single = output['pred_logits']  # (1, seq_len, num_classes)

    # Squeeze batch dimension
    pred_boxes_single = pred_boxes_single.squeeze(0).cpu().numpy() 
    pred_logits_single = pred_logits_single.squeeze(0) # Shape (seq_len, num_classes)

    decoded_text = decode_ctc_output(pred_logits_single, idx_to_char, blank_idx)
    
    return original_image_pil, pred_boxes_single, decoded_text


def draw_predictions_on_image(image_pil, boxes, text, font_path=None, default_font_size=20):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(font_path, default_font_size)
    except (IOError, TypeError): 
        font = ImageFont.load_default()

    text_position = (10, 10)
    text_bbox = draw.textbbox(text_position, text, font=font)
    draw.rectangle(text_bbox, fill="black") # Add background for text
    draw.text(text_position, text, fill="green", font=font)

    img_width, img_height = image_pil.size
    for box in boxes:
        # Assuming box format [x1, y1, x2, y2] from model.
        # Filter out padding boxes (all -1) or invalid boxes.
        if list(box) == [-1, -1, -1, -1] or list(box) == [0,0,0,0]:
            continue
        
        x1, y1, x2, y2 = box
        # Clamp coordinates to image dimensions
        x1_c = max(0, min(x1, img_width -1))
        y1_c = max(0, min(y1, img_height -1))
        x2_c = max(0, min(x2, img_width -1))
        y2_c = max(0, min(y2, img_height -1))

        if x1_c >= x2_c or y1_c >= y2_c : # Skip invalid or zero-area boxes after clamping
            # print(f"Skipping invalid box: original {box}, clamped {[x1_c, y1_c, x2_c, y2_c]}")
            continue
        
        draw.rectangle([(x1_c,y1_c), (x2_c,y2_c)], outline='red', width=2)
        
    return image_pil


def main(args):
    try:
        model = OCRModel.load_from_checkpoint(args.checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
    except Exception as e:
        print(f"Error loading model from checkpoint {args.checkpoint_path}: {e}")
        return

    model.eval() # Ensure model is in eval mode

    # Determine device
    if args.accelerator == 'gpu' and torch.cuda.is_available():
        device_str = f"cuda:{args.devices.split(',')[0]}" if isinstance(args.devices, str) and args.devices != 'auto' and args.devices.split(',')[0].isdigit() else "cuda"
    elif args.accelerator == 'mps' and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
        if args.accelerator == 'gpu' and not torch.cuda.is_available():
            print("Warning: GPU specified but CUDA not available. Using CPU.")
        elif args.accelerator == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS specified but not available. Using CPU.")

    device = torch.device(device_str)
    model.to(device)
    print(f"Using device: {device}")

    # Get vocab info from hparams, otherwise use defaults
    idx_to_char = getattr(model.hparams, 'idx_to_char', DEFAULT_IDX_TO_CHAR)
    blank_idx = getattr(model.hparams, 'blank_char_idx', DEFAULT_BLANK_CHAR_IDX)
    char_to_idx = getattr(model.hparams, 'char_to_idx', DEFAULT_CHAR_TO_IDX)

    if idx_to_char == DEFAULT_IDX_TO_CHAR or not hasattr(model.hparams, 'idx_to_char'):
        print("Warning: Using default vocabulary. Ensure this matches the training vocabulary for correct decoding.")
        # If these are not in hparams, the model might not have saved them correctly.
        # For this script to work, these mappings are essential.

    # Define image transforms (must be consistent with training)
    # For simplicity, using ToTensor. If normalization or specific sizing was used, it must be replicated.
    image_transforms_func = transforms.ToTensor() 
    # Example if normalization was used:
    # image_transforms_func = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])


    if args.test_data_dir:
        print(f"Running in batch test mode on data from: {args.test_data_dir}")
        test_dataset = OcrDataset(
            data_split_dir=Path(args.test_data_dir),
            char_to_idx=char_to_idx, 
            image_transforms=image_transforms_func # Pass the transform function
        )
        if len(test_dataset) == 0:
            print(f"Error: Test dataset at {args.test_data_dir} is empty or not found.")
            return

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=ocr_collate_fn,
            num_workers=args.num_workers
        )
        
        trainer_accelerator = args.accelerator if args.accelerator != 'auto' else None
        trainer_devices = 'auto'
        if args.accelerator in ['gpu', 'mps'] and args.devices and args.devices != 'auto':
            if args.devices.isdigit():
                trainer_devices = int(args.devices)
            elif isinstance(args.devices, str) and ',' in args.devices:
                 trainer_devices = [int(d.strip()) for d in args.devices.split(',')]
            else: # specific device string like 'cuda:0'
                 trainer_devices = args.devices
        
        trainer = pl.Trainer(
            accelerator=trainer_accelerator, 
            devices=trainer_devices, 
            logger=False # No extensive logging for test mode
        )
        print(f"Starting trainer.test() with accelerator '{trainer_accelerator}' and devices '{trainer_devices}'...")
        trainer.test(model, test_loader)

    elif args.image_path:
        print(f"Running single image prediction for: {args.image_path}")
        pil_image, pred_boxes, pred_text = predict_single_image(
            model, 
            args.image_path, 
            image_transforms_func, 
            idx_to_char, 
            blank_idx, 
            device
        )

        if pil_image is None: # Error handled in predict_single_image
            return

        print(f"Predicted text: '{pred_text}'")
        
        output_image_pil = draw_predictions_on_image(
            pil_image.copy(), 
            pred_boxes, # Pass all boxes, draw_predictions filters them
            pred_text, 
            font_path=args.font_path
        )

        if args.output_path:
            try:
                output_image_pil.save(args.output_path)
                print(f"Saved visualized output to: {args.output_path}")
            except Exception as e:
                print(f"Error saving output image to {args.output_path}: {e}")
        else:
            try:
                print("Attempting to display image (this might not work in all environments e.g. remote server)...")
                output_image_pil.show()
            except Exception as e:
                print(f"Could not display image: {e}. Consider providing an --output_path to save the image.")
    else:
        print("Neither --test_data_dir nor --image_path provided. Please specify one. Exiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test OCR Model and Visualize Predictions")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt file).")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test_data_dir", type=str, help="Path to test data directory (for running pl.Trainer.test()).")
    group.add_argument("--image_path", type=str, help="Path to a single image for prediction and visualization.")
    
    parser.add_argument("--output_path", type=str, help="Path to save the visualized output image (used with --image_path).")
    parser.add_argument("--font_path", type=str, default=None, help="Path to a .ttf font file for drawing text on the image.")
    
    parser.add_argument("--accelerator", type=str, default='auto', choices=['cpu', 'gpu', 'mps', 'auto'], help="Accelerator to use ('cpu', 'gpu', 'mps', 'auto').")
    parser.add_argument("--devices", type=str, default='auto', help="Devices to use (e.g., 'auto', '1' for 1 GPU, '0,1' for specific GPUs, or a specific device string like 'cuda:0').")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (used with --test_data_dir).")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of workers for DataLoader (used with --test_data_dir).")

    args = parser.parse_args()

    if args.output_path:
        os.makedirs(Path(args.output_path).parent, exist_ok=True)
        
    main(args)
