import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """
    A basic convolutional block: (Conv2D -> BatchNorm2D -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DownsampleBlock(nn.Module):
    """
    Downsampling block: MaxPool2D -> ConvBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

# Note: UpsampleBlock might not be strictly needed for this encoder if output is from Transformer
# but defining it for completeness or future U-Net variations.
class UpsampleBlock(nn.Module):
    """
    Upsampling block: ConvTranspose2D -> Concatenate skip -> ConvBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # The in_channels for ConvTranspose2d is the number of channels from the previous layer in the decoder
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The input to the ConvBlock will be (in_channels // 2 + skip_channels)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Pad x if its spatial dimensions are smaller than skip_connection's
        # This can happen if the input image size is not perfectly divisible by 2^N
        diff_y = skip_connection.size()[2] - x.size()[2]
        diff_x = skip_connection.size()[3] - x.size()[3]

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


class UNetTransformerEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels_decoder=768, 
                 initial_filters=64, 
                 num_unet_layers=4, # Number of downsampling stages in U-Net
                 num_transformer_layers=4, 
                 transformer_heads=8, 
                 transformer_mlp_dim=2048,
                 bottleneck_patch_size=None): # Not used like ViT patch, but for sequence length determination
        super().__init__()

        self.in_channels = in_channels
        self.out_channels_decoder = out_channels_decoder
        self.initial_filters = initial_filters
        self.num_unet_layers = num_unet_layers

        # U-Net Encoder Path
        self.inc = ConvBlock(in_channels, initial_filters)
        
        self.down_blocks = nn.ModuleList()
        current_filters = initial_filters
        for i in range(num_unet_layers):
            self.down_blocks.append(DownsampleBlock(current_filters, current_filters * 2))
            current_filters *= 2
        
        # Bottleneck channels will be current_filters
        self.bottleneck_channels = current_filters

        # Positional Encoding for Transformer
        # The sequence length N will depend on the image input size and num_unet_layers
        # For an image of H_img x W_img, after num_unet_layers, H_bottleneck = H_img / (2^num_unet_layers)
        # W_bottleneck = W_img / (2^num_unet_layers)
        # N = H_bottleneck * W_bottleneck
        # This needs to be calculated dynamically in forward or set based on expected input size.
        # For now, let's assume a fixed max sequence length or initialize dynamically.
        # We'll use learnable positional embeddings.
        # Max sequence length placeholder, ideally determined by input size.
        # e.g. if input is 64x1024, 4 layers: 64/(2^4)=4, 1024/(2^4)=64. N = 4*64=256
        # self.max_seq_len = (image_height // (2**num_unet_layers)) * (image_width // (2**num_unet_layers))
        # This will be set in the first forward pass or passed via config
        self.pos_embed = None # Will be nn.Parameter

        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.bottleneck_channels, 
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=0.1, # Standard dropout
            activation='relu',
            batch_first=True # Expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_transformer_layers
        )

        # Final projection layer to match decoder's expected input dimension
        self.output_projection = nn.Linear(self.bottleneck_channels, out_channels_decoder)
        
        self._dummy_input_size_for_pos_embed = None # Store input size to re-init pos_embed if it changes

    def _initialize_positional_embeddings(self, h_bottleneck, w_bottleneck, device):
        seq_len = h_bottleneck * w_bottleneck
        if self.pos_embed is None or self.pos_embed.shape[1] != seq_len:
            print(f"Initializing positional embeddings for sequence length: {seq_len}, feature dim: {self.bottleneck_channels}")
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.bottleneck_channels, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=.02) # Standard initialization for ViT pos_embed

    def forward(self, x):
        # Store input H, W to initialize pos_embed on first pass or if size changes
        input_h, input_w = x.shape[2], x.shape[3]
        if self._dummy_input_size_for_pos_embed is None or \
           self._dummy_input_size_for_pos_embed != (input_h, input_w) :
            self._dummy_input_size_for_pos_embed = (input_h, input_w)
            # Calculate H, W at bottleneck
            h_bottleneck = input_h // (2**self.num_unet_layers)
            w_bottleneck = input_w // (2**self.num_unet_layers)
            if h_bottleneck == 0 or w_bottleneck == 0:
                raise ValueError(
                    f"Input image size ({input_h}x{input_w}) is too small for {self.num_unet_layers} U-Net layers. "
                    f"Results in bottleneck size ({h_bottleneck}x{w_bottleneck})."
                )
            self._initialize_positional_embeddings(h_bottleneck, w_bottleneck, x.device)


        # U-Net Encoder Path
        x1 = self.inc(x)
        # Skip connections are not used in this setup as we only take bottleneck to transformer
        # If a U-Net decoder was used, these would be needed.
        # skips = [x1] 
        current_features = x1
        for i in range(self.num_unet_layers):
            current_features = self.down_blocks[i](current_features)
            # skips.append(current_features)
        
        bottleneck_features = current_features # (B, C, H_b, W_b)
        
        B, C, H_b, W_b = bottleneck_features.shape
        
        # Reshape for Transformer: (B, C, H_b, W_b) -> (B, H_b * W_b, C)
        # N = H_b * W_b (sequence length), D_feat = C (feature dimension)
        transformer_input = bottleneck_features.flatten(2).permute(0, 2, 1) # (B, N, C)

        # Add positional encoding
        if transformer_input.shape[1] != self.pos_embed.shape[1]:
            # This can happen if input image sizes vary between batches, which is generally not recommended
            # or if the initial calculation was off. Re-initialize.
            print(f"Warning: Sequence length mismatch. Input: {transformer_input.shape[1]}, PosEmbed: {self.pos_embed.shape[1]}. Re-initializing pos_embed.")
            self._initialize_positional_embeddings(H_b, W_b, x.device)
            
        transformer_input = transformer_input + self.pos_embed

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input) # (B, N, C)

        # Project to the decoder's expected feature dimension
        projected_output = self.output_projection(transformer_output) # (B, N, out_channels_decoder)
        
        return projected_output # Shape: (batch_size, sequence_length, feature_dim)

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Example parameters
    img_channels = 1
    decoder_dim = 768 # e.g., hidden size of a BERT-like decoder
    unet_layers = 4   # Results in H/16, W/16 bottleneck
    
    # Create model instance
    encoder = UNetTransformerEncoder(
        in_channels=img_channels,
        out_channels_decoder=decoder_dim,
        initial_filters=64,
        num_unet_layers=unet_layers,
        num_transformer_layers=6,
        transformer_heads=8,
        transformer_mlp_dim=2048
    )

    # Dummy input tensor (Batch, Channels, Height, Width)
    # Height and Width should be divisible by 2^num_unet_layers for simplicity
    # e.g., 64x256. After 4 layers: 64/16=4, 256/16=16. SeqLen = 4*16 = 64
    dummy_image = torch.randn(4, img_channels, 64, 256) 
    
    print(f"Input image shape: {dummy_image.shape}")

    # Forward pass
    try:
        output_sequence = encoder(dummy_image)
        print(f"Output sequence shape: {output_sequence.shape}") # Expected: (B, N, decoder_dim)
        # For 64x256 input, N = (64/16) * (256/16) = 4 * 16 = 64
        # So, expected output: (4, 64, 768)
        expected_seq_len = (64 // (2**unet_layers)) * (256 // (2**unet_layers))
        assert output_sequence.shape == (dummy_image.size(0), expected_seq_len, decoder_dim)
        print("UNetTransformerEncoder test passed.")

    except Exception as e:
        print(f"Error during UNetTransformerEncoder test: {e}")
        raise

    # Test with a different input size to check dynamic pos_embed initialization
    dummy_image_2 = torch.randn(2, img_channels, 32, 128)
    print(f"Input image 2 shape: {dummy_image_2.shape}")
    try:
        output_sequence_2 = encoder(dummy_image_2)
        print(f"Output sequence 2 shape: {output_sequence_2.shape}")
        expected_seq_len_2 = (32 // (2**unet_layers)) * (128 // (2**unet_layers))
        # 32/16=2, 128/16=8. SeqLen = 2*8 = 16
        # Expected output: (2, 16, 768)
        assert output_sequence_2.shape == (dummy_image_2.size(0), expected_seq_len_2, decoder_dim)
        print("UNetTransformerEncoder test with different input size passed.")
    except Exception as e:
        print(f"Error during UNetTransformerEncoder test with different input size: {e}")
        raise

    # Test with input size that is not perfectly divisible
    dummy_image_3 = torch.randn(2, img_channels, 70, 270) # H, W = 70, 270
    # H_b = 70 // 16 = 4
    # W_b = 270 // 16 = 16
    # N = 4 * 16 = 64
    print(f"Input image 3 shape: {dummy_image_3.shape}")
    try:
        output_sequence_3 = encoder(dummy_image_3)
        print(f"Output sequence 3 shape: {output_sequence_3.shape}")
        expected_seq_len_3 = (70 // (2**unet_layers)) * (270 // (2**unet_layers))
        assert output_sequence_3.shape == (dummy_image_3.size(0), expected_seq_len_3, decoder_dim)
        print("UNetTransformerEncoder test with non-divisible input size passed.")
    except Exception as e:
        print(f"Error during UNetTransformerEncoder test with non-divisible input size: {e}")
        raise
        
    # Test with too small input
    dummy_image_4 = torch.randn(2, img_channels, 8, 32) # H, W = 8, 32
    # H_b = 8 // 16 = 0
    # W_b = 32 // 16 = 2
    print(f"Input image 4 shape: {dummy_image_4.shape}")
    try:
        output_sequence_4 = encoder(dummy_image_4)
        print(f"Output sequence 4 shape: {output_sequence_4.shape}")
        print("UNetTransformerEncoder test with too small input size: This should have failed but didn't if this line is printed.")
    except ValueError as e:
        print(f"Successfully caught expected error for too small input: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for too small input: {e}")
        raise

    print("All UNetTransformerEncoder tests finished.")

# To make this runnable standalone for testing:
# python -m src.models.unet_transformer_encoder
# (Assuming your project root is in PYTHONPATH)
# Or from project root: python src/models/unet_transformer_encoder.py

# To integrate into LitOCRModel, this UNetTransformerEncoder will replace the nn.Identity() encoder.
# The LitOCRModel's __init__ will need to pass appropriate parameters.
# The `out_channels_decoder` of this encoder must match the `hidden_size` expected by
# the HuggingFace decoder's cross-attention mechanism.
# The `encoder_output` in LitOCRModel's forward method will be the output of this module.
# Its shape (B, N, D) is what cross-attention decoders typically expect for `encoder_hidden_states`.
# The `target_ids.shape[1]` used for repeating encoder_output in LitOCRModel will need adjustment
# as the Transformer output already has a sequence dimension (N).
# The `encoder_hidden_states` should be directly `transformer_output` from this module.
# The logic in LitOCRModel:
# if encoder_output.ndim == 2: ...
# elif encoder_output.ndim == 3 and encoder_output.shape[1] == 1 : ...
# will need to be updated because this encoder always outputs (B, N, D).
# It should directly be: `encoder_hidden_states=encoder_output`
# (where encoder_output is the output of this UNetTransformerEncoder).
# The same applies to the `generate` call.
# No need to unsqueeze or repeat unless the specific Hugging Face decoder expects a different format
# for encoder_hidden_states than (B, N_encoder, D_encoder).
# Standard cross-attention uses this format.

# The dummy_bbox_tensor in LitOCRModel might also need to align its sequence length
# with N (the sequence length from this encoder) rather than max_label_len,
# or be handled more carefully based on how bboxes are predicted relative to the encoder's output sequence.
# For now, LitOCRModel's bbox part uses decoder's hidden states, so its length is tied to target_ids length.
# This is fine for training with teacher forcing. For inference, bbox prediction would be more complex
# and might need alignment with the generated sequence.
# This is a detail for LitOCRModel's bbox head, not this encoder itself.
