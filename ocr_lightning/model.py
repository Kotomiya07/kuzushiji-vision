import torch
import torch.nn as nn
import torch.nn.functional as F # Added for log_softmax
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl

class OCRModel(pl.LightningModule):
    def __init__(self, char_to_idx, idx_to_char, learning_rate=1e-4, max_boxes=50):
        super().__init__()
        
        # Store hyperparameters
        # blank_char_idx is derived and stored, num_chars is derived and stored
        _num_chars = len(char_to_idx)
        _blank_char_idx = char_to_idx.get('<blank>', 0) # Assuming '<blank>' is your blank character, default to 0 if not found
        if _blank_char_idx !=0 and list(char_to_idx.keys())[0] != '<blank>':
             print(f"Warning: blank_char_idx is {_blank_char_idx} but char_to_idx suggests it might be 0. Ensure consistency for CTCLoss.")

        self.save_hyperparameters(
            {
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'learning_rate': learning_rate,
                'max_boxes': max_boxes,
                'blank_char_idx': _blank_char_idx, # Ensure this is correct for your char_to_idx
                'num_chars': _num_chars
            }
        )

        # Feature Extractor
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]) 
        self.resnet_output_features = resnet.fc.in_features 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Localization Head
        self.localization_head = nn.Linear(self.resnet_output_features, self.hparams.max_boxes * 4)

        # Recognition Head
        self.recognition_rnn = nn.LSTM(
            input_size=self.resnet_output_features,
            hidden_size=256, 
            num_layers=2,    
            bidirectional=True, 
            batch_first=True 
        )
        self.recognition_fc = nn.Linear(256 * 2, self.hparams.num_chars)

        # Loss Functions
        self.localization_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.recognition_loss_fn = nn.CTCLoss(
            blank=self.hparams.blank_char_idx, 
            zero_infinity=True, 
            reduction='mean'
        )

        # Loss Weights
        self.loc_loss_weight = 1.0
        self.rec_loss_weight = 1.0

    def forward(self, images):
        # images shape: (batch_size, channels, height, width)
        features = self.feature_extractor(images) # (batch_size, resnet_output_features, H/32, W/32)
        
        # Global features for localization and simplified recognition
        pooled_features = self.avgpool(features) # (batch_size, resnet_output_features, 1, 1)
        pooled_features_flat = torch.flatten(pooled_features, 1) # (batch_size, resnet_output_features)

        # Localization
        raw_box_predictions = self.localization_head(pooled_features_flat) # (batch_size, max_boxes * 4)
        # Reshape to (batch_size, max_boxes, 4)
        box_predictions = raw_box_predictions.view(-1, self.hparams.max_boxes, 4) 

        # Recognition (Simplified: using pooled features as a sequence of length 1)
        # Input for RNN: (batch_size, seq_len, input_size)
        # Here, seq_len = 1
        rnn_input = pooled_features_flat.unsqueeze(1) # (batch_size, 1, resnet_output_features)
        
        rnn_output, _ = self.recognition_rnn(rnn_input) # (batch_size, 1, 256*2 for bidirectional)
        
        # Output for CTC: (batch_size, sequence_length, num_classes) if batch_first=True for RNN
        # or (sequence_length, batch_size, num_classes) if CTCLoss expects that.
        # PyTorch's CTCLoss default is (Time, Batch, Class), i.e. (seq_len, batch, num_classes)
        # If self.recognition_rnn has batch_first=True, its output is (Batch, Time, Feat)
        # So self.recognition_fc output is (Batch, Time, Class)
        char_logits = self.recognition_fc(rnn_output) # (batch_size, 1, self.hparams.num_chars)
        
        return {'pred_boxes': box_predictions, 'pred_logits': char_logits}

    def _shared_step(self, batch, batch_idx, step_name):
        images, label_texts, bounding_boxes_batch, _, bbox_counts, _ = \
            batch['images'], batch['label_texts'], batch['bounding_boxes_batch'], \
            batch['target_lengths'], batch['bbox_counts'], batch['image_paths']

        output_dict = self(images)
        pred_boxes = output_dict['pred_boxes']
        pred_logits = output_dict['pred_logits']

        batch_size = images.size(0)

        # Localization Loss
        batch_loc_loss = 0.0
        valid_samples_for_loc_loss = 0 # Counts samples that have at least one box to compare

        for i in range(batch_size):
            num_gt_boxes_for_sample = bbox_counts[i].item()
            # pred_boxes are already (B, self.hparams.max_boxes, 4)
            # gt_boxes_padded (bounding_boxes_batch) are (B, max_collated_boxes, 4)
            
            boxes_to_compare = min(num_gt_boxes_for_sample, self.hparams.max_boxes)

            if boxes_to_compare == 0:
                continue

            current_gt_boxes = bounding_boxes_batch[i, :boxes_to_compare, :]
            current_pred_boxes = pred_boxes[i, :boxes_to_compare, :]
            
            loss_for_sample = self.localization_loss_fn(current_pred_boxes, current_gt_boxes)
            batch_loc_loss += loss_for_sample
            valid_samples_for_loc_loss += 1
        
        loc_loss = batch_loc_loss / valid_samples_for_loc_loss if valid_samples_for_loc_loss > 0 \
            else torch.tensor(0.0, device=self.device, requires_grad=True)


        # Recognition Loss
        raw_logits = pred_logits # (B, 1, self.hparams.num_chars)
        log_probs = F.log_softmax(raw_logits, dim=2)
        log_probs_for_ctc = log_probs.permute(1, 0, 2) # (1, B, num_chars) (Time, Batch, Class)

        max_encoded_len = 0
        encoded_labels_list = []
        actual_target_lengths = []

        for text in label_texts:
            encoded = [self.hparams.char_to_idx.get(char, self.hparams.blank_char_idx) for char in text]
            encoded_labels_list.append(torch.tensor(encoded, dtype=torch.long, device=self.device))
            actual_target_lengths.append(len(encoded))
            max_encoded_len = max(max_encoded_len, len(encoded))

        # Pad encoded labels
        # Only create padded_targets if there's something to pad, otherwise CTC can error with empty targets
        if max_encoded_len > 0:
            padded_targets = torch.full((batch_size, max_encoded_len), 
                                        fill_value=self.hparams.blank_char_idx, 
                                        dtype=torch.long, device=self.device)
            for i, l_tensor in enumerate(encoded_labels_list):
                padded_targets[i, :l_tensor.size(0)] = l_tensor
        else: # All label_texts were empty
            padded_targets = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)


        input_lengths_for_ctc = torch.full(size=(batch_size,), 
                                           fill_value=log_probs_for_ctc.size(0), # Sequence length of 1
                                           dtype=torch.long, device=self.device)
        
        target_lengths_for_ctc = torch.tensor(actual_target_lengths, dtype=torch.long, device=self.device)

        valid_targets_mask = target_lengths_for_ctc > 0
        
        if torch.all(~valid_targets_mask) or max_encoded_len == 0 : # All targets are empty or all labels were empty strings
            rec_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Filter inputs for CTCLoss
            log_probs_ctc_filtered = log_probs_for_ctc[:, valid_targets_mask, :]
            
            # Ensure padded_targets_filtered is not empty before max()
            target_lengths_filtered = target_lengths_for_ctc[valid_targets_mask]
            max_len_filtered = target_lengths_filtered.max().item() if len(target_lengths_filtered) > 0 else 0
            
            padded_targets_filtered = padded_targets[valid_targets_mask, :max(1, max_len_filtered)] # Slice up to max length among valid targets
            
            input_lengths_ctc_filtered = input_lengths_for_ctc[valid_targets_mask]

            if padded_targets_filtered.numel() == 0 and target_lengths_filtered.sum() == 0:
                 # This case handles if all valid targets were actually empty strings (which should be caught by max_encoded_len == 0)
                 # or if somehow a target_length > 0 but encoded string was empty.
                 rec_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                rec_loss = self.recognition_loss_fn(
                    log_probs_ctc_filtered, 
                    padded_targets_filtered, 
                    input_lengths_ctc_filtered, 
                    target_lengths_filtered
                )

        # Combined Loss
        total_loss = (self.loc_loss_weight * loc_loss) + (self.rec_loss_weight * rec_loss)

        # Logging
        self.log(f'{step_name}/loc_loss', loc_loss, on_step=(step_name=="train"), on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f'{step_name}/rec_loss', rec_loss, on_step=(step_name=="train"), on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f'{step_name}/total_loss', total_loss, on_step=(step_name=="train"), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "val")
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch['images'].size(0)) # Duplicate with _shared_step logging
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "test")
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch['images'].size(0)) # Duplicate
        return loss
