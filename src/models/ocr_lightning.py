import pytorch_lightning as pl
import torch
import torch.nn as nn
from jiwer import compute_measures
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..utils.metrics import compute_iou
from .unet_transformer_encoder import UNetTransformerEncoder


class LitOCRModel(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, tokenizer: AutoTokenizer):  # Added tokenizer argument
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.vocab = tokenizer  # Use the passed tokenizer instance

        # Decoder (Hugging Face) - Initialize this first to get its hidden_size
        # Ensure the model name is appropriate or use a placeholder
        decoder_config_path = model_config.get("decoder_path", "KoichiYasuoka/roberta-small-japanese-aozora-char")
        decoder_config_obj = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config_obj.is_decoder = True
        decoder_config_obj.add_cross_attention = True  # Enable cross-attention
        decoder_config_obj.output_hidden_states = True  # Ensure hidden_states are outputted
        # Important: Tie decoder vocabulary size to our tokenizer
        decoder_config_obj.vocab_size = len(self.vocab)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_config_path, config=decoder_config_obj)
        # Resize token embeddings in case the pretrained model's vocab size differs
        self.decoder.resize_token_embeddings(len(self.vocab))

        # Configure the decoder's generation config directly
        self.decoder.generation_config.max_length = self.model_config.get("max_gen_len", 128)
        self.decoder.generation_config.pad_token_id = self.vocab.pad_token_id
        self.decoder.generation_config.eos_token_id = self.vocab.eos_token_id
        # Assuming bos_token_id is the bos_token_id for generation
        self.decoder.generation_config.bos_token_id = self.vocab.bos_token_id
        self.decoder.generation_config.output_hidden_states = True
        self.decoder.generation_config.return_dict_in_generate = True

        # Encoder (UNetTransformerEncoder)
        # The out_channels_decoder of the encoder must match the decoder's hidden_size
        self.encoder = UNetTransformerEncoder(
            in_channels=model_config.get("encoder_in_channels", 1),
            out_channels_decoder=self.decoder.config.hidden_size,  # Use decoder's actual hidden size
            initial_filters=model_config.get("encoder_initial_filters", 64),
            num_unet_layers=model_config.get("encoder_num_unet_layers", 4),
            num_transformer_layers=model_config.get("encoder_num_transformer_layers", 4),
            transformer_heads=model_config.get("encoder_transformer_heads", 8),
            transformer_mlp_dim=model_config.get("encoder_transformer_mlp_dim", 2048),
        )

        # Bounding box predictor head
        # The input dimension should match the decoder's hidden size
        self.bbox_predictor = nn.Linear(self.decoder.config.hidden_size, 4)  # 4 for (x_min, y_min, x_max, y_max)

        # Loss functions
        self.criterion_char = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_token_id)  # Use pad_id from tokenizer
        self.criterion_bbox = nn.L1Loss()

        # Filter out tokenizer from hparams to avoid saving complex object
        hparams_to_save = {k: v for k, v in self.hparams.items() if k != "tokenizer"}
        self.hparams.clear()
        self.hparams.update(hparams_to_save)
        # self.save_hyperparameters(ignore=['tokenizer']) # Alternative if PTL version supports it well

    def forward(self, images, target_ids=None):
        # Encoder pass (placeholder)
        encoder_output = self.encoder(images)  # Assuming images are preprocessed

        # Decoder pass
        if target_ids is not None:
            # Teacher forcing: provide target_ids as decoder_input_ids
            # encoder_output from UNetTransformerEncoder is already (B, N, D_encoder)
            # This is the expected shape for encoder_hidden_states.

            # Create attention_mask from target_ids (mask out padding tokens)
            # target_ids shape: (batch_size, seq_len)
            # attention_mask shape: (batch_size, seq_len)
            attention_mask = (target_ids != self.vocab.pad_token_id).long()
            token_type_ids = torch.zeros_like(target_ids, device=target_ids.device)

            decoder_outputs = self.decoder(
                input_ids=target_ids,
                attention_mask=attention_mask,  # Pass the attention_mask
                token_type_ids=token_type_ids,
                encoder_hidden_states=encoder_output,  # Directly use encoder_output
                # encoder_attention_mask can be passed if needed.
                # If encoder_output has padding, a mask is crucial.
                # UNetTransformerEncoder output length depends on input image size.
                # Assuming all images in batch are same size, so N is consistent.
                # If UNetTransformerEncoder output is unpadded, mask might be optional or all ones.
            )
            char_logits = decoder_outputs.logits
        else:
            # Inference: use generate method
            # encoder_output from UNetTransformerEncoder is (B, N, D_encoder)
            # This is the expected shape for encoder_hidden_states in generate method.
            dummy_input_ids = torch.full(
                (images.size(0), 1), fill_value=self.vocab.bos_token_id, dtype=torch.long, device=self.device
            )
            generation_output = self.decoder.generate(
                input_ids=dummy_input_ids,
                encoder_hidden_states=encoder_output,
                # Removed explicit generation_config, model's own config will be used
            )

            if hasattr(generation_output, "sequences"):
                char_logits = None  # Or find a way to get them from generate if needed
            else:
                char_logits = None  # Or find a way to get them from generate if needed

        # Bounding box prediction
        # Use the decoder's last hidden state or specific output for bbox prediction
        # Assuming decoder_outputs.last_hidden_state is available if target_ids were passed
        if target_ids is not None:
            # hidden_states shape: [batch_size, seq_len, hidden_size]
            predicted_bboxes = self.bbox_predictor(decoder_outputs.hidden_states[-1])
        else:
            # For generation, bbox prediction is more complex.
            # One way is to take the hidden states from the `generate` method if it returns them,
            # or pass the generated sequence back through the model to get hidden states.
            # For now, placeholder:
            predicted_bboxes = torch.zeros((images.size(0), self.model_config.get("max_gen_len", 50), 4), device=self.device)

        return char_logits, predicted_bboxes

    def training_step(self, batch, batch_idx):
        # Assuming OneLineOCRDataset returns: images, target_ids (full, with GO and EOS), target_bboxes (dummy)
        # target_mask is not provided by the current OneLineOCRDataset
        images, target_ids_full, target_bboxes_full = batch

        # Prepare inputs for model and loss calculation
        decoder_input_ids = target_ids_full[:, :-1]  # From <go> to token before <eos>
        target_char_ids = target_ids_full[:, 1:]  # From first char to <eos>

        # Create mask for loss calculation based on target_char_ids (excluding <go>, including <eos>)
        # Mask should be True for non-PAD tokens
        # target_bboxes_full are (B, max_label_len-1, 4)
        # predicted_bboxes will be (B, max_label_len-1, 4)
        # target_char_ids is (B, max_label_len-1)
        loss_mask = target_char_ids != self.vocab.pad_token_id  # Shape: (B, max_label_len-1)

        char_logits, predicted_bboxes = self.forward(images, target_ids=decoder_input_ids)

        # Character loss
        # char_logits: [B, S_dec, V] where S_dec = max_label_len-1
        # target_char_ids: [B, S_dec]
        char_loss = self.criterion_char(char_logits.reshape(-1, char_logits.size(-1)), target_char_ids.reshape(-1))

        # Bounding box loss
        # predicted_bboxes: [B, S_dec, 4]
        # target_bboxes_full: [B, S_dec, 4] (dummy zeros from dataset)
        # Apply mask to bbox loss
        # loss_mask needs to be expanded for the last dimension of bboxes if criterion needs it,
        # but direct boolean indexing is fine.
        masked_predicted_bboxes = predicted_bboxes[loss_mask]
        masked_target_bboxes = target_bboxes_full[loss_mask]

        if masked_predicted_bboxes.numel() > 0:  # Only compute loss if there are non-padded items
            bbox_loss = self.criterion_bbox(masked_predicted_bboxes, masked_target_bboxes)
        else:
            bbox_loss = torch.tensor(0.0, device=char_loss.device)

        loss = char_loss + self.model_config.get("bbox_loss_weight", 0.05) * bbox_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_char_loss", char_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_bbox_loss", bbox_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_ids_full, target_bboxes_full = batch

        # Prepare inputs for model and loss calculation
        decoder_input_ids = target_ids_full[:, :-1]  # From <go> to token before <eos>
        target_char_ids = target_ids_full[:, 1:]  # From first char to <eos>
        loss_mask = target_char_ids != self.vocab.pad_token_id  # Shape: (B, max_label_len-1)

        # Forward pass for loss calculation (using teacher forcing)
        char_logits, predicted_bboxes_for_loss = self.forward(images, target_ids=decoder_input_ids)

        # Calculate losses
        char_loss = self.criterion_char(char_logits.reshape(-1, char_logits.size(-1)), target_char_ids.reshape(-1))

        masked_predicted_bboxes_loss = predicted_bboxes_for_loss[loss_mask]
        masked_target_bboxes_loss = target_bboxes_full[loss_mask]
        if masked_predicted_bboxes_loss.numel() > 0:
            bbox_loss = self.criterion_bbox(masked_predicted_bboxes_loss, masked_target_bboxes_loss)
        else:
            bbox_loss = torch.tensor(0.0, device=char_loss.device)

        loss = char_loss + self.model_config.get("bbox_loss_weight", 0.05) * bbox_loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_char_loss", char_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_bbox_loss", bbox_loss, on_epoch=True, logger=True, sync_dist=True)

        # --- Inference for CER and IoU ---
        # Note: predicted_bboxes_for_loss are from teacher-forced inputs.
        # For a "true" inference IoU, one would need bboxes predicted based on generated_ids.
        # The current self.forward with target_ids=None returns placeholder bboxes.
        # For now, we use predicted_bboxes_for_loss for IoU calculation as per typical validation setup.

        # Generate IDs for CER
        encoder_output_val = self.encoder(images)
        generated_ids_output = self.decoder.generate(
            input_ids=torch.full((images.size(0), 1), self.vocab.bos_token_id, dtype=torch.long, device=self.device),
            encoder_hidden_states=encoder_output_val,
            max_length=self.model_config.get("max_gen_len", 128),  # Use 128 consistent with other parts
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            # bos_token_id=self.vocab.go_token_id, # Already part of input_ids
            # output_hidden_states=True, # Already set in generation_config
            # return_dict_in_generate=True # Already set in generation_config
        )
        actual_generated_ids = generated_ids_output.sequences  # Access sequences attribute

        pred_texts = [self.vocab.decode(ids.tolist(), join=True) for ids in actual_generated_ids]
        gt_texts = [self.vocab.decode(ids.tolist(), join=True) for ids in target_char_ids]  # Use target_char_ids here

        val_cer = 0.0
        valid_pairs_count = 0
        batch_cer_sum = 0
        for gt, pred in zip(gt_texts, pred_texts, strict=False):
            if gt:  # Only compute CER for non-empty ground truth strings
                # jiwer.compute_measures returns a dict, e.g. {'wer': ..., 'mer': ..., 'wil': ...}
                # We use 'wer' as Character Error Rate (assuming no spaces in characters)
                measures = compute_measures(gt, pred)
                batch_cer_sum += measures["wer"]
                valid_pairs_count += 1
        if valid_pairs_count > 0:
            val_cer = batch_cer_sum / valid_pairs_count

        # IoU Calculation
        # Using predicted_bboxes_for_loss and target_bboxes_full, masked by loss_mask
        batch_iou_sum = 0.0
        num_valid_bboxes = 0

        for i in range(images.size(0)):  # Iterate through batch
            sample_mask = loss_mask[i]  # Mask for current sample (S_dec,)
            sample_pred_bboxes = predicted_bboxes_for_loss[i][sample_mask]  # (N_valid, 4)
            sample_gt_bboxes = target_bboxes_full[i][sample_mask]  # (N_valid, 4)

            if sample_pred_bboxes.numel() > 0:
                # compute_iou expects (N, 4) and (M, 4), returns (N, M)
                # Here, we compare element-wise, so N=M.
                # We need to iterate if compute_iou doesn't do broadcasting element-wise for (N,4) vs (N,4) -> (N,)
                # Assuming compute_iou(A, B) where A is (N,4) and B is (M,4)
                # If we want direct 1-to-1, we might need a loop or diagonal.
                # Let's assume compute_iou handles (N,4) and (N,4) to give (N,) ious or we take diagonal.
                # For now, let's iterate for clarity.
                for j in range(sample_pred_bboxes.size(0)):
                    iou = compute_iou(sample_pred_bboxes[j].unsqueeze(0), sample_gt_bboxes[j].unsqueeze(0))
                    batch_iou_sum += iou.item()  # compute_iou returns tensor
                num_valid_bboxes += sample_pred_bboxes.size(0)

        val_iou = batch_iou_sum / num_valid_bboxes if num_valid_bboxes > 0 else 0.0

        self.log("val_cer", val_cer, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_iou", val_iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, target_ids_full, target_bboxes_full = batch

        # Prepare inputs for model and loss calculation
        decoder_input_ids = target_ids_full[:, :-1]
        target_char_ids = target_ids_full[:, 1:]
        loss_mask = target_char_ids != self.vocab.pad_token_id

        # Forward pass for loss calculation
        char_logits, predicted_bboxes_for_loss = self.forward(images, target_ids=decoder_input_ids)

        char_loss = self.criterion_char(char_logits.reshape(-1, char_logits.size(-1)), target_char_ids.reshape(-1))

        masked_predicted_bboxes_loss = predicted_bboxes_for_loss[loss_mask]
        masked_target_bboxes_loss = target_bboxes_full[loss_mask]
        if masked_predicted_bboxes_loss.numel() > 0:
            bbox_loss = self.criterion_bbox(masked_predicted_bboxes_loss, masked_target_bboxes_loss)
        else:
            bbox_loss = torch.tensor(0.0, device=char_loss.device)

        loss = char_loss + self.model_config.get("bbox_loss_weight", 0.05) * bbox_loss

        self.log("test_loss", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_char_loss", char_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_bbox_loss", bbox_loss, on_epoch=True, logger=True, sync_dist=True)

        # Inference for CER and IoU
        encoder_output_test = self.encoder(images)
        generated_ids_output = self.decoder.generate(
            input_ids=torch.full((images.size(0), 1), self.vocab.bos_token_id, dtype=torch.long, device=self.device),
            encoder_hidden_states=encoder_output_test,
            max_length=self.model_config.get("max_gen_len", 128),  # Use 128
            pad_token_id=self.vocab.pad_token_id,
            eos_token_id=self.vocab.eos_token_id,
            # bos_token_id=self.vocab.go_token_id,
            # output_hidden_states=True,
            # return_dict_in_generate=True
        )
        actual_generated_ids = generated_ids_output.sequences  # Access sequences attribute

        pred_texts = [self.vocab.decode(ids.tolist(), join=True) for ids in actual_generated_ids]
        gt_texts = [self.vocab.decode(ids.tolist(), join=True) for ids in target_char_ids]

        test_cer = 0.0
        valid_pairs_count = 0
        batch_cer_sum = 0
        for gt, pred in zip(gt_texts, pred_texts, strict=False):
            if gt:
                measures = compute_measures(gt, pred)
                batch_cer_sum += measures["wer"]
                valid_pairs_count += 1
        if valid_pairs_count > 0:
            test_cer = batch_cer_sum / valid_pairs_count

        # IoU Calculation
        batch_iou_sum = 0.0
        num_valid_bboxes = 0
        for i in range(images.size(0)):  # Iterate through batch
            sample_mask = loss_mask[i]
            sample_pred_bboxes = predicted_bboxes_for_loss[i][sample_mask]
            sample_gt_bboxes = target_bboxes_full[i][sample_mask]

            if sample_pred_bboxes.numel() > 0:
                for j in range(sample_pred_bboxes.size(0)):
                    iou = compute_iou(sample_pred_bboxes[j].unsqueeze(0), sample_gt_bboxes[j].unsqueeze(0))
                    batch_iou_sum += iou.item()
                num_valid_bboxes += sample_pred_bboxes.size(0)

        test_iou = batch_iou_sum / num_valid_bboxes if num_valid_bboxes > 0 else 0.0

        self.log("test_cer", test_cer, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_iou", test_iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Example optimizer, replace with one from optimizer_config if needed
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-4),
            weight_decay=self.optimizer_config.get("weight_decay", 0.01),
        )
        # Learning rate scheduler (optional)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optimizer_config.get("lr_scheduler_factor", 0.1),
            patience=self.optimizer_config.get("lr_scheduler_patience", 10),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
