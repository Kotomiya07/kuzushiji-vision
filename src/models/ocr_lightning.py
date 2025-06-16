import pytorch_lightning as pl
import schedulefree
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
        if self.vocab.pad_token_id is not None:  # Ensure pad_token_id is set in the config
            decoder_config_obj.pad_token_id = self.vocab.pad_token_id

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
        self.bbox_predictor = nn.Linear(self.decoder.config.hidden_size, 4)
        # Loss functions
        self.criterion_char = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_token_id)  # Use pad_id from tokenizer
        self.criterion_bbox = nn.L1Loss()

        # Filter out tokenizer from hparams to avoid saving complex object
        hparams_to_save = {k: v for k, v in self.hparams.items() if k != "tokenizer"}
        self.hparams.clear()
        self.hparams.update(hparams_to_save)
        # self.save_hyperparameters(ignore=['tokenizer']) # Alternative if PTL version supports it well

    def forward(self, images, labels=None, bbox_values=None):
        # Encoder
        encoder_outputs = self.encoder(images)

        # Decoder pass
        if labels is not None:
            # Assertions to check label validity
            if torch.any(labels >= self.vocab.vocab_size):
                raise ValueError(f"Label token ID {labels.max().item()} exceeds vocab size {self.vocab.vocab_size}")
            if torch.any(labels < 0):
                raise ValueError(f"Label token ID {labels.min().item()} is negative")

            # Teacher forcing: provide target_ids as decoder_input_ids
            # encoder_output from UNetTransformerEncoder is already (B, N, D_encoder)
            # This is the expected shape for encoder_hidden_states.

            # Create attention_mask from target_ids (mask out padding tokens)
            # target_ids shape: (batch_size, seq_len)
            # attention_mask shape: (batch_size, seq_len)
            attention_mask = (labels != self.vocab.pad_token_id).long()
            # Use logger instead of print for debugging
            # self.log_dict({"attention_mask_shape": str(attention_mask.shape)}, on_step=False, on_epoch=True)
            token_type_ids = torch.zeros_like(labels, device=labels.device)

            decoder_outputs = self.decoder(
                input_ids=labels,
                attention_mask=attention_mask,  # Pass the attention_mask
                token_type_ids=token_type_ids,
                encoder_hidden_states=encoder_outputs,  # Directly use encoder_output
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
                encoder_hidden_states=encoder_outputs,
                # Removed explicit generation_config, model's own config will be used
                # The model's generation_config already has:
                # output_hidden_states = True
                # return_dict_in_generate = True
            )

            char_logits = None  # In typical generation, logits for each step are not directly output by .generate()

            # Bounding box prediction for generated sequence
            if hasattr(generation_output, "hidden_states") and generation_output.hidden_states is not None:
                # generation_output.hidden_states is a tuple of hidden states for each decoder layer.
                # Each element of the tuple has shape (batch_size, generated_sequence_length, hidden_size).
                # We use the hidden states from the last decoder layer.
                last_layer_hidden_states = generation_output.hidden_states[-1]
                # `last_layer_hidden_states` corresponds to the embeddings of the generated tokens (including BOS if it was input).
                # If dummy_input_ids was (batch_size, 1) with BOS, then generated_sequence_length includes this BOS.
                predicted_bboxes = self.bbox_predictor(last_layer_hidden_states[:, 1:, :])

                # Ensure the sequence length of predicted_bboxes matches expectations if necessary.
                # For example, if a fixed length is needed downstream, padding/truncation might be required.
                # Here, it will have shape (batch_size, generated_sequence_length, 4).
            else:
                # Fallback if hidden_states are not available as expected or if an error occurs.
                # Use the actual generated sequence length if available, otherwise use max_gen_len.
                current_max_len = self.decoder.generation_config.max_length
                if hasattr(generation_output, "sequences"):
                    current_max_len = generation_output.sequences.size(1)
                predicted_bboxes = torch.zeros((images.size(0), current_max_len, 4), device=self.device)

        return char_logits, predicted_bboxes

    def training_step(self, batch, batch_idx):
        image_tensor, label_tensor, bbox_tensor = batch
        loss, logits, _ = self.forward(images=image_tensor, labels=label_tensor, bbox_values=bbox_tensor)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensor, label_tensor, bbox_tensor = batch
        loss, logits, generated_ids = self.forward(images=image_tensor, labels=label_tensor, bbox_values=bbox_tensor)

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        image_tensor, label_tensor, bbox_tensor = batch
        loss, logits, generated_ids = self.forward(images=image_tensor, labels=label_tensor, bbox_values=bbox_tensor)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        # Example optimizer, replace with one from optimizer_config if needed
        optimizer = schedulefree.RAdamScheduleFree(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-4),
            weight_decay=self.optimizer_config.get("weight_decay", 0.01),
        )
        return optimizer

    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "train"):
            optimizer.train()

    def on_validation_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_test_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

    def on_predict_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()
