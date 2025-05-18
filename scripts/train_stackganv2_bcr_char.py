import datetime
import glob
import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from schedulefree import RAdamScheduleFree # Keep existing optimizer
from torch.autograd import Variable # For CA_NET eps

from src.callbacks.ema import EMACallback

# --- 設定項目 (Keep your existing settings, adjust as needed) ---
DATA_ROOT = "data/onechannel"
#OUTPUT_DIR = f"experiments/stackgan_bcr_char_v2/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"experiments/simple_vae/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"{OUTPUT_DIR}/lightning_logs"

NUM_UNICODES = -1
UNICODE_TO_INT = {}
INT_TO_UNICODE = {}

LATENT_DIM = 100       # StackGAN Z_DIM is often 100
CHAR_EMBED_DIM = 128   # Dimension of initial character embedding (input to CA_NET)
CA_EMBED_DIM = 128     # Output dimension of CA_NET (cfg.GAN.EMBEDDING_DIM in StackGAN++)
GF_DIM = 64           # Base channel multiplier for Generators (like cfg.GAN.GF_DIM)
DF_DIM = 32            # Base channel multiplier for Discriminators (like cfg.GAN.DF_DIM)
R_NUM = 2              # Number of residual blocks in G_NET2 (like cfg.GAN.R_NUM)


IMG_SIZE_S1 = 32
IMG_SIZE_S2 = 64
CHANNELS = 1
BATCH_SIZE = 2048 # Adjusted based on typical values and potential memory increase
LR_G = 1.6e-5 # Adjusted, StackGAN often uses 2e-4
LR_D = 1e-5 # Adjusted
B1 = 0.9 # StackGAN often uses 0.5 for Adam
B2 = 0.999
N_EPOCHS = 300
LAMBDA_BCR = 1.5
LAMBDA_KL = 1.0 # Coefficient for KL divergence loss (like cfg.TRAIN.COEFF.KL)
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1

# --- データ拡張 (bCR用) ---
bcr_transform = T.Compose([
    T.RandomAffine(degrees=7, translate=(0.07, 0.07), scale=(0.93, 1.07), shear=(-5, 5, -5, 5)),
])

# --- StackGAN++ Helper Modules & Functions ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Upsample the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        nn.GLU(dim=1)
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes): # Used in NEXT_STAGE_G
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        nn.GLU(dim=1)
    )
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            nn.GLU(dim=1),
            conv3x3(channel_num, channel_num), # Output is channel_num
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class CA_NET(nn.Module):
    def __init__(self, initial_embed_dim, ca_output_dim):
        super(CA_NET, self).__init__()
        self.t_dim = initial_embed_dim
        self.ef_dim = ca_output_dim
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True) # For mu, logvar
        self.glu =nn.GLU() # StackGAN uses GLU here, but fc output would be ef_dim * 4. Sticking to simpler mu/logvar for now.

    def encode(self, char_embedding):
        x = self.glu(self.fc(char_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # Use torch.randn_like(std) for device placement
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, char_embedding):
        mu, logvar = self.encode(char_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Block for Discriminator (similar to Block3x3_leakRelu in model.py)
def D_Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# --- データセットクラス (No change needed from your original) ---
class CharUnicodeDataset(Dataset):
    def __init__(self, data_root, unicode_to_int_map, image_size=64, transform=None): # image_sizeはS2のサイズを想定
        self.data_root = Path(data_root)
        self.unicode_to_int_map = unicode_to_int_map
        self.image_size = image_size # S2 size
        self.image_paths = []
        self.labels = []

        if not self.unicode_to_int_map:
            raise ValueError("Unicode to int mapping is empty. Initialize it first.")

        for unicode_str, unicode_id in self.unicode_to_int_map.items():
            unicode_dir = self.data_root / unicode_str
            if unicode_dir.is_dir():
                for img_path in glob.glob(str(unicode_dir / "*.jpg")):
                    self.image_paths.append(img_path)
                    self.labels.append(unicode_id)

        if not self.image_paths:
            print(f"警告: 画像が {data_root} に見つかりませんでした。")


        self.base_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)), # S2 size
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.custom_transform = transform # For bCR, applied in training_step

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"画像読み込みエラー {img_path}: {e}. ダミー画像を返します。")
            image = Image.new("L", (self.image_size, self.image_size), color=0)

        image_s2 = self.base_transform(image)
        return image_s2, torch.tensor(label, dtype=torch.long)


# --- Generator Stage 1 (INIT_STAGE_G style) ---
class GeneratorStage1(nn.Module):
    def __init__(self, latent_dim, ca_embed_dim, channels, gf_dim, img_size_s1):
        super(GeneratorStage1, self).__init__()
        self.gf_dim = gf_dim
        self.in_dim = latent_dim + ca_embed_dim
        self.img_size_s1 = img_size_s1

        # FC layer to create initial 4x4 feature map for upsampling
        # Output of GLU will be self.gf_dim * (4x4 features)
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.gf_dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.gf_dim * 4 * 4 * 2),
            nn.GLU(dim=1)
        )

        # Upsampling blocks: 4x4 -> 8x8 -> 16x16 (for img_size_s1=16)
        self.upsample1 = upBlock(self.gf_dim, self.gf_dim // 2)       # Output: gf_dim//2, 8x8
        self.upsample2 = upBlock(self.gf_dim // 2, self.gf_dim // 4)  # Output: gf_dim//4, 16x16
        # Add more if IMG_SIZE_S1 is larger, e.g., for 32x32, one more upBlock

        current_channels_after_upsample = self.gf_dim // 4
        self.img_layer = nn.Sequential(
            conv3x3(current_channels_after_upsample, channels),
            nn.Tanh()
        )
        self.out_channels_for_g2 = current_channels_after_upsample # For G2 input

    def forward(self, noise, c_code):
        in_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(in_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4) # Reshape to 3D
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        # h_code is now the feature map before final image generation, e.g., (batch, gf_dim//4, 16, 16)
        
        fake_img = self.img_layer(h_code)
        return fake_img, h_code # Return features for G2

# --- Generator Stage 2 (NEXT_STAGE_G style) ---
class GeneratorStage2(nn.Module):
    def __init__(self, g1_out_channels, ca_embed_dim, channels, gf_dim, num_res_blocks):
        super(GeneratorStage2, self).__init__()
        self.gf_dim = gf_dim # Should be based on g1_out_channels or a new base for G2
        self.ef_dim = ca_embed_dim # Dimension of c_code from CA_NET

        # Initial processing of h_code from G1 and c_code
        # Input to jointConv: g1_out_channels + ca_embed_dim
        # Output of jointConv: self.gf_dim (e.g. could be g1_out_channels if gf_dim matches)
        self.jointConv = Block3x3_relu(g1_out_channels + self.ef_dim, self.gf_dim)
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock(self.gf_dim) for _ in range(num_res_blocks)]
        )
        
        # Upsampling (e.g., 16x16 from G1 -> 32x32 -> 64x64)
        # Each upBlock halves the channel dimension in its definition
        self.upsample1 = upBlock(self.gf_dim, self.gf_dim // 2) # Output: gf_dim//2
        self.upsample2 = upBlock(self.gf_dim // 2, self.gf_dim // 4) # Output: gf_dim//4

        current_channels_after_upsample = self.gf_dim // 4
        self.img_layer = nn.Sequential(
            conv3x3(current_channels_after_upsample, channels),
            nn.Tanh()
        )

    def forward(self, h_code_from_g1, c_code):
        s_size = h_code_from_g1.size(2) # Spatial size from G1's feature map
        # Spatially replicate c_code
        c_code_spatial = c_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, s_size, s_size)
        
        # Concatenate h_code from G1 and spatial c_code
        h_c_code = torch.cat((h_code_from_g1, c_code_spatial), 1)
        
        out_code = self.jointConv(h_c_code)
        out_code = self.residual_blocks(out_code)
        
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        
        fake_img = self.img_layer(out_code)
        return fake_img

# --- Discriminator Stage 1 (D_NET64 style for 16x16) ---
class DiscriminatorStage1(nn.Module):
    def __init__(self, ca_embed_dim, channels, df_dim, img_size_s1):
        super(DiscriminatorStage1, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ca_embed_dim
        self.img_size_s1 = img_size_s1

        # Image encoder: 16x16 -> 8x8 -> 4x4
        self.img_encoder = nn.Sequential(
            nn.Conv2d(channels, df_dim, kernel_size=4, stride=2, padding=1, bias=False), # 16->8
            nn.LeakyReLU(0.2, inplace=True),
            downBlock(df_dim, df_dim * 2),  # 8->4, out: df_dim*2 channels
        )
        encoded_feature_dim = df_dim * 2

        # Unconditional logits path
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(encoded_feature_dim, 1, kernel_size=4, stride=4), # 4x4 features to 1x1
            # Sigmoid applied in loss function (BCEWithLogitsLoss)
        )

        # Conditional logits path
        self.joint_conv = D_Block3x3_leakRelu(encoded_feature_dim + self.ef_dim, encoded_feature_dim)
        self.cond_logits = nn.Sequential(
            nn.Conv2d(encoded_feature_dim, 1, kernel_size=4, stride=4),
        )

    def forward(self, img, c_code): # c_code is mu from CA_NET in StackGAN trainer, or c_code itself
        x_code = self.img_encoder(img) # (batch, df_dim*2, 4, 4)

        # Unconditional score
        output_uncond = self.uncond_logits(x_code).view(-1)

        # Conditional score
        s_size = x_code.size(2)
        c_code_spatial = c_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((x_code, c_code_spatial), 1)
        h_c_code = self.joint_conv(h_c_code)
        output_cond = self.cond_logits(h_c_code).view(-1)
        
        return [output_cond, output_uncond]


# --- Discriminator Stage 2 (D_NET128 style for 64x64) ---
class DiscriminatorStage2(nn.Module):
    def __init__(self, ca_embed_dim, channels, df_dim, img_size_s2):
        super(DiscriminatorStage2, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ca_embed_dim
        self.img_size_s2 = img_size_s2

        # Image encoder: 64x64 -> 32 -> 16 -> 8 -> 4
        self.img_encoder = nn.Sequential(
            nn.Conv2d(channels, df_dim, kernel_size=4, stride=2, padding=1, bias=False), # 64->32
            nn.LeakyReLU(0.2, inplace=True),
            downBlock(df_dim, df_dim * 2),      # 32->16, out: df_dim*2
            downBlock(df_dim * 2, df_dim * 4),  # 16->8,  out: df_dim*4
            downBlock(df_dim * 4, df_dim * 8),  # 8->4,   out: df_dim*8
        )
        encoded_feature_dim = df_dim * 8

        self.uncond_logits = nn.Sequential(
            nn.Conv2d(encoded_feature_dim, 1, kernel_size=4, stride=4), # 4x4 features to 1x1
        )

        self.joint_conv = D_Block3x3_leakRelu(encoded_feature_dim + self.ef_dim, encoded_feature_dim)
        self.cond_logits = nn.Sequential(
            nn.Conv2d(encoded_feature_dim, 1, kernel_size=4, stride=4),
        )

    def forward(self, img, c_code):
        x_code = self.img_encoder(img)

        output_uncond = self.uncond_logits(x_code).view(-1)

        s_size = x_code.size(2)
        c_code_spatial = c_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((x_code, c_code_spatial), 1)
        h_c_code = self.joint_conv(h_c_code)
        output_cond = self.cond_logits(h_c_code).view(-1)
        
        return [output_cond, output_uncond]


# --- PyTorch Lightning Module (StackGAN version) ---
class StackCharGANLitModule(L.LightningModule):
    def __init__(
        self,
        latent_dim, char_embed_dim, ca_embed_dim, num_unicodes,
        lr_g, lr_d, b1, b2, lambda_bcr, lambda_kl,
        img_size_s1, img_size_s2, channels,
        gf_dim, df_dim, r_num, # StackGAN++ specific dimensions
        output_dir="gan_outputs",
        fixed_noise_samples=16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['fixed_noise_samples']) # fixed_noise_samples is not a direct model param
        self.automatic_optimization = False

        # Initial embedding layer for unicode IDs
        self.char_embedding_layer = nn.Embedding(num_unicodes, char_embed_dim)

        # Conditioning Augmentation Network
        self.ca_net = CA_NET(char_embed_dim, ca_embed_dim)

        # Stage 1 Models
        self.generator_s1 = GeneratorStage1(latent_dim, ca_embed_dim, channels, gf_dim, img_size_s1)
        g1_out_feat_channels = self.generator_s1.out_channels_for_g2 # Get feature channels for G2
        self.discriminator_s1 = DiscriminatorStage1(ca_embed_dim, channels, df_dim, img_size_s1)

        # Stage 2 Models
        # G2's gf_dim could be g1_out_feat_channels, or a new base like gf_dim for G1
        # For consistency with StackGAN++, NEXT_STAGE_G takes ngf (which is G1's output channel dim)
        # and ef_dim (ca_embed_dim). Let's use g1_out_feat_channels as the "base" for G2's ResBlocks etc.
        self.generator_s2 = GeneratorStage2(g1_out_feat_channels, ca_embed_dim, channels, g1_out_feat_channels, r_num)
        self.discriminator_s2 = DiscriminatorStage2(ca_embed_dim, channels, df_dim, img_size_s2)
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.bcr_aug_transform = bcr_transform # Your bCR transform
        self.s1_downsampler = T.Resize((img_size_s1, img_size_s1), antialias=True)

        num_fixed_samples = min(num_unicodes, fixed_noise_samples)
        self.fixed_noise = torch.randn(num_fixed_samples, self.hparams.latent_dim)
        self.fixed_labels = torch.arange(0, num_fixed_samples, dtype=torch.long)

        self.sample_dir = Path(self.hparams.output_dir) / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Initialize weights
        self.apply(weights_init)
        # self.ca_net.apply(weights_init) # Redundant if self.apply covers it
        # self.generator_s1.apply(weights_init)
        # self.discriminator_s1.apply(weights_init)
        # self.generator_s2.apply(weights_init)
        # self.discriminator_s2.apply(weights_init)


    def forward(self, z, labels_int): # For inference
        initial_char_embeds = self.char_embedding_layer(labels_int)
        c_code, _, _ = self.ca_net(initial_char_embeds)
        with torch.no_grad(): # G1 features are intermediate
            low_res_img, low_res_features = self.generator_s1(z, c_code)
        high_res_img = self.generator_s2(low_res_features, c_code)
        return high_res_img

    def training_step(self, batch, batch_idx):
        opt_g, opt_d1, opt_d2 = self.optimizers() # G optimizer for G1, G2, CA_NET

        opt_g.train()
        opt_d1.train()
        opt_d2.train()

        real_imgs_s2, labels_int = batch # labels_int are integer unicode IDs
        real_imgs_s1 = self.s1_downsampler(real_imgs_s2)
        
        batch_size = real_imgs_s2.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        # --- Get Conditioned Embeddings ---
        initial_char_embeds = self.char_embedding_layer(labels_int)
        c_code, mu, logvar = self.ca_net(initial_char_embeds)
        # In StackGAN++, discriminators often use 'mu' as the text condition, 
        # while generators use the reparameterized 'c_code'. Let's try c_code for D as well for simplicity first.
        # Or use mu for D:
        d_condition_code = mu # As per StackGAN trainer.py: netD(real_imgs, mu.detach())


        # --- STAGE 1 TRAINING ---
        self.generator_s1.train()
        self.discriminator_s1.train()
        self.ca_net.train() # CA_NET is part of G path

        # Train Discriminator D1
        opt_d1.zero_grad()
        
        # Real images for D1
        d1_preds_real = self.discriminator_s1(real_imgs_s1, d_condition_code.detach())
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d1_preds_real[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d1_preds_real[1].shape, device=self.device) * 0.3
        d1_loss_real_cond = self.adversarial_loss(d1_preds_real[0], real_label)
        d1_loss_real_uncond = self.adversarial_loss(d1_preds_real[1], fake_label)
        d1_loss_real = d1_loss_real_cond + d1_loss_real_uncond # Or weighted sum

        # Fake images for D1
        with torch.no_grad():
            fake_imgs_s1, _ = self.generator_s1(z, c_code.detach()) # Detach c_code as G1 is not updated here
        d1_preds_fake = self.discriminator_s1(fake_imgs_s1.detach(), d_condition_code.detach())
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d1_preds_fake[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d1_preds_fake[1].shape, device=self.device) * 0.3
        d1_loss_fake_cond = self.adversarial_loss(d1_preds_fake[0], real_label)
        d1_loss_fake_uncond = self.adversarial_loss(d1_preds_fake[1], fake_label)
        d1_loss_fake = d1_loss_fake_cond + d1_loss_fake_uncond

        d1_loss_standard = (d1_loss_real + d1_loss_fake) / 2 # Averaging factor might need adjustment
        d1_loss = d1_loss_standard
        
        # bCR Loss for D1 (applied to conditional part)
        if self.hparams.lambda_bcr > 0 and self.bcr_aug_transform:
            augmented_real_imgs_s1 = self.bcr_aug_transform(real_imgs_s1)
            with torch.no_grad():
                d1_preds_real_no_grad = self.discriminator_s1(real_imgs_s1, d_condition_code.detach())
            d1_preds_augmented = self.discriminator_s1(augmented_real_imgs_s1, d_condition_code.detach())
            # Apply bCR to the conditional output, or both
            d1_bcr_loss = nn.functional.mse_loss(d1_preds_augmented[0], d1_preds_real_no_grad[0])
            d1_loss += self.hparams.lambda_bcr * d1_bcr_loss
            self.log("loss_s1/d1_bcr", d1_bcr_loss, logger=True, sync_dist=True)

        self.log("loss_s1/d1_real_cond", d1_loss_real_cond, logger=True, sync_dist=True)
        self.log("loss_s1/d1_fake_cond", d1_loss_fake_cond, logger=True, sync_dist=True)
        self.log("loss_s1/d1_total", d1_loss, prog_bar=False, logger=True, sync_dist=True) # Prog_bar for G loss

        self.manual_backward(d1_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator_s1.parameters(), 1.0)
        opt_d1.step()

        # Train Generator G (G1 + CA_NET for this stage's loss components)
        opt_g.zero_grad()
        
        # G1's adversarial loss
        # We use c_code (reparameterized) for G, and d_condition_code (mu) for D.
        # For G's loss, D needs to see G's output with the same condition type D expects.
        fake_imgs_s1_for_g1, _ = self.generator_s1(z, c_code)
        d1_preds_fake_for_g1 = self.discriminator_s1(fake_imgs_s1_for_g1, d_condition_code) # D uses mu
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d1_preds_fake_for_g1[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d1_preds_fake_for_g1[1].shape, device=self.device) * 0.3
        g1_loss_adv_cond = self.adversarial_loss(d1_preds_fake_for_g1[0], real_label)
        g1_loss_adv_uncond = self.adversarial_loss(d1_preds_fake_for_g1[1], fake_label)
        g1_loss_adv = g1_loss_adv_cond + g1_loss_adv_uncond

        # KL Divergence loss from CA_NET
        kl_div_loss = KL_loss(mu, logvar)
        g1_loss_total = g1_loss_adv + self.hparams.lambda_kl * kl_div_loss
        
        self.log("loss_s1/g1_adv_cond", g1_loss_adv_cond, logger=True, sync_dist=True)
        self.log("loss_s1/g1_kl_div", kl_div_loss, logger=True, sync_dist=True)
        self.log("loss_s1/g1_total", g1_loss_total, prog_bar=True, logger=True, sync_dist=True)

        # Backward for G1 part of G loss (CA_NET grads will also accumulate)
        # self.manual_backward(g1_loss_total) # Will do combined backward for G1 and G2 loss later
        
        # --- STAGE 2 TRAINING ---
        self.generator_s2.train()
        self.discriminator_s2.train()

        # Detach G1's output features and c_code for D2 training and G2 input
        with torch.no_grad():
            _, low_res_features_for_s2 = self.generator_s1(z, c_code.detach())

        # Train Discriminator D2
        opt_d2.zero_grad()

        d2_preds_real = self.discriminator_s2(real_imgs_s2, d_condition_code.detach())
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d2_preds_real[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d2_preds_real[1].shape, device=self.device) * 0.3
        d2_loss_real_cond = self.adversarial_loss(d2_preds_real[0], real_label)
        d2_loss_real_uncond = self.adversarial_loss(d2_preds_real[1], fake_label)
        d2_loss_real = d2_loss_real_cond + d2_loss_real_uncond

        with torch.no_grad():
            fake_imgs_s2 = self.generator_s2(low_res_features_for_s2.detach(), c_code.detach())
        d2_preds_fake = self.discriminator_s2(fake_imgs_s2.detach(), d_condition_code.detach())
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d2_preds_fake[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d2_preds_fake[1].shape, device=self.device) * 0.3
        d2_loss_fake_cond = self.adversarial_loss(d2_preds_fake[0], real_label)
        d2_loss_fake_uncond = self.adversarial_loss(d2_preds_fake[1], fake_label)
        d2_loss_fake = d2_loss_fake_cond + d2_loss_fake_uncond
        
        d2_loss_standard = (d2_loss_real + d2_loss_fake) / 2
        d2_loss = d2_loss_standard

        if self.hparams.lambda_bcr > 0 and self.bcr_aug_transform:
            augmented_real_imgs_s2 = self.bcr_aug_transform(real_imgs_s2)
            with torch.no_grad():
                 d2_preds_real_no_grad = self.discriminator_s2(real_imgs_s2, d_condition_code.detach())
            d2_preds_augmented = self.discriminator_s2(augmented_real_imgs_s2, d_condition_code.detach())
            d2_bcr_loss = nn.functional.mse_loss(d2_preds_augmented[0], d2_preds_real_no_grad[0])
            d2_loss += self.hparams.lambda_bcr * d2_bcr_loss
            self.log("loss_s2/d2_bcr", d2_bcr_loss, logger=True, sync_dist=True)

        self.log("loss_s2/d2_real_cond", d2_loss_real_cond, logger=True, sync_dist=True)
        self.log("loss_s2/d2_fake_cond", d2_loss_fake_cond, logger=True, sync_dist=True)
        self.log("loss_s2/d2_total", d2_loss, prog_bar=False, logger=True, sync_dist=True)

        self.manual_backward(d2_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator_s2.parameters(), 1.0)
        opt_d2.step()

        # Train Generator G (G2 part, CA_NET already handled by G1 loss for KL)
        # opt_g.zero_grad() # Zeroed before G1 loss computation

        fake_imgs_s2_for_g2 = self.generator_s2(low_res_features_for_s2, c_code) # Use non-detached features/c_code for G2
        d2_preds_fake_for_g2 = self.discriminator_s2(fake_imgs_s2_for_g2, d_condition_code)
        # label smoothing
        # Discriminatorのrealラベルを 1.0 ではなく 0.9 や 0.7～1.2 の範囲の乱数に、fakeラベルを 0.0 ではなく 0.1 や 0.0～0.3 の範囲の乱数にする
        real_label = torch.rand(d2_preds_fake_for_g2[0].shape, device=self.device) * 0.3 + 0.7
        fake_label = torch.rand(d2_preds_fake_for_g2[1].shape, device=self.device) * 0.3
        g2_loss_adv_cond = self.adversarial_loss(d2_preds_fake_for_g2[0], real_label)
        g2_loss_adv_uncond = self.adversarial_loss(d2_preds_fake_for_g2[1], fake_label)
        g2_loss_adv = g2_loss_adv_cond + g2_loss_adv_uncond

        # Total Generator Loss
        # StackGAN++ sums losses from different stages for G. Or, train G1 and G2 sequentially within the step.
        # Here, G1 loss has KL. G2 loss is purely adversarial.
        # The optimzer `opt_g` updates G1, G2, and CA_NET.
        g_loss_total = g1_loss_total + g2_loss_adv # Combine losses for the single G optimizer
        
        self.log("loss_s2/g2_adv_cond", g2_loss_adv_cond, logger=True, sync_dist=True)
        self.log("loss_g_total", g_loss_total, prog_bar=True, logger=True, sync_dist=True) # Main G loss to track

        self.manual_backward(g_loss_total) # Backward for combined G losses
        torch.nn.utils.clip_grad_norm_(self.generator_s1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.generator_s2.parameters(), 1.0)
        opt_g.step()

        opt_g.eval()
        opt_d1.eval()
        opt_d2.eval()


    def configure_optimizers(self):
        # Optimizer for G1, G2, and CA_NET
        params_g = list(self.generator_s1.parameters()) + \
                   list(self.generator_s2.parameters()) + \
                   list(self.ca_net.parameters()) + \
                   list(self.char_embedding_layer.parameters()) # Char embedding is also trained
        
        optimizer_g = RAdamScheduleFree(params_g, lr=self.hparams.lr_g, betas=(self.hparams.b1, self.hparams.b2))
        optimizer_d1 = RAdamScheduleFree(self.discriminator_s1.parameters(), lr=self.hparams.lr_d, betas=(self.hparams.b1, self.hparams.b2))
        optimizer_d2 = RAdamScheduleFree(self.discriminator_s2.parameters(), lr=self.hparams.lr_d, betas=(self.hparams.b1, self.hparams.b2))
        
        # ScheduleFree recommends starting in eval mode if using warmup internally
        # However, your original code sets them to train() then eval() in training_step.
        # Let's stick to what ScheduleFree docs typically suggest for init if using its warmup.
        # If not using internal warmup, this might not be necessary.
        # For manual_optimization=True, Lightning doesn't call optimizer.eval/train.
        # We'll manage this in training_step if needed or rely on default Adam behavior.
        optimizer_g.eval()
        optimizer_d1.eval()
        optimizer_d2.eval()
        
        return optimizer_g, optimizer_d1, optimizer_d2

    def on_train_epoch_end(self):
        self.generator_s1.eval()
        self.generator_s2.eval()
        self.ca_net.eval()
        self.char_embedding_layer.eval()
        
        with torch.no_grad():
            fixed_noise_dev = self.fixed_noise.to(self.device)
            fixed_labels_dev = self.fixed_labels.to(self.device)

            initial_char_embeds_fixed = self.char_embedding_layer(fixed_labels_dev)
            c_code_fixed, _, _ = self.ca_net(initial_char_embeds_fixed)

            low_res_samples, low_res_features_fixed = self.generator_s1(fixed_noise_dev, c_code_fixed)
            generated_samples_s2 = self.generator_s2(low_res_features_fixed, c_code_fixed)
            
            # Denormalize for saving
            low_res_samples_denorm = (low_res_samples * 0.5) + 0.5
            generated_samples_s2_denorm = (generated_samples_s2 * 0.5) + 0.5

            grid_s1 = make_grid(low_res_samples_denorm, nrow=int(np.sqrt(len(low_res_samples_denorm))), normalize=False)
            save_path_s1 = self.sample_dir / f"epoch_{self.current_epoch+1:04d}_s1.png"
            save_image(grid_s1, save_path_s1)
            
            grid_s2 = make_grid(generated_samples_s2_denorm, nrow=int(np.sqrt(len(generated_samples_s2_denorm))), normalize=False)
            save_path_s2 = self.sample_dir / f"epoch_{self.current_epoch+1:04d}_s2.png"
            save_image(grid_s2, save_path_s2)

            if self.logger and self.logger.experiment is not None: # Check if logger (e.g. Wandb) is active
                try:
                    self.logger.log_image(key="Generated Samples Stage1", images=[str(save_path_s1)])
                    self.logger.log_image(key="Generated Samples Stage2", images=[str(save_path_s2)])
                except Exception as e:
                    print(f"Wandb logging error: {e}")


class SimpleVAELitModule(L.LightningModule):
    def __init__(
        self,
        input_channels=3,
        hidden_dim=128,
        latent_dim=32,
        learning_rate=LR_G,
        b1=B1,
        b2=B2
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # エンコーダー
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 潜在空間の平均と分散
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)  # 64x64画像の場合
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        
        # デコーダー
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x_recon, x, mu, log_var):
        # 再構成誤差（MSE損失）
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KLダイバージェンス
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # バッチサイズで正規化
        batch_size = x.size(0)
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size
        
        # 損失の合計
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def training_step(self, batch, batch_idx):
        opt = self.configure_optimizers()
        opt.train()
        x, _ = batch
        x_recon, mu, log_var = self(x)
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, log_var)
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        opt.eval()
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_kl_loss', kl_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        opt = self.configure_optimizers()
        opt.eval()
        x, _ = batch
        x_recon, mu, log_var = self(x)
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, log_var)
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_kl_loss', kl_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = RAdamScheduleFree(self.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.b1, self.hparams.b2))
        optimizer.eval()
        return optimizer




# --- Unicode マッピングの準備とメイン処理 (prepare_unicode_dataは変更なし) ---
def prepare_unicode_data(data_root_path):
    global NUM_UNICODES, UNICODE_TO_INT, INT_TO_UNICODE # Ensure these are global
    data_root = Path(data_root_path)
    if not data_root.is_dir():
        print(f"エラー: データルートディレクトリ {data_root_path} が見つかりません。")
        return False

    unicode_dirs = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if not unicode_dirs:
        print(f"エラー: {data_root_path} にUnicodeサブディレクトリが見つかりません。")
        return False

    NUM_UNICODES = len(unicode_dirs)
    UNICODE_TO_INT = {name: i for i, name in enumerate(unicode_dirs)}
    INT_TO_UNICODE = {i: name for i, name in enumerate(unicode_dirs)}
    print(f"Unicodeマッピング完了: {NUM_UNICODES} 種類の文字を発見。")
    return True

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # TF32 can be faster but might affect precision slightly.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    
    L.seed_everything(42)

    if not prepare_unicode_data(DATA_ROOT):
        exit(1)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    dataset = CharUnicodeDataset(data_root=DATA_ROOT, unicode_to_int_map=UNICODE_TO_INT, image_size=IMG_SIZE_S2)
    if len(dataset) == 0:
        print("エラー: データセットが空です。")
        exit(1)
    print(f"データセットサイズ: {len(dataset)} 画像")

    actual_num_workers = NUM_WORKERS if len(dataset) > BATCH_SIZE * NUM_WORKERS else 0
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=actual_num_workers,
        persistent_workers=True if actual_num_workers > 0 else False, #pin_memory=True
        pin_memory=torch.cuda.is_available()
    )

    # stackgan_model = StackCharGANLitModule(
    #     latent_dim=LATENT_DIM, char_embed_dim=CHAR_EMBED_DIM, ca_embed_dim=CA_EMBED_DIM,
    #     num_unicodes=NUM_UNICODES,
    #     lr_g=LR_G, lr_d=LR_D, b1=B1, b2=B2,
    #     lambda_bcr=LAMBDA_BCR, lambda_kl=LAMBDA_KL,
    #     img_size_s1=IMG_SIZE_S1, img_size_s2=IMG_SIZE_S2, channels=CHANNELS,
    #     gf_dim=GF_DIM, df_dim=DF_DIM, r_num=R_NUM,
    #     output_dir=OUTPUT_DIR,
    #     fixed_noise_samples=16 # This is used in init but not saved in hparams directly
    # )

    vae_model = SimpleVAELitModule(
        input_channels=1,
        hidden_dim=128,
        latent_dim=LATENT_DIM,
        learning_rate=LR_G,
        b1=B1,
        b2=B2
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=Path(OUTPUT_DIR) / "checkpoints",
    #     filename="stackchar-gan-{epoch:03d}-{loss_g_total:.2f}",
    #     save_top_k=3,
    #     monitor="loss_g_total", # Monitor combined G loss
    #     mode="min",
    #     save_last=True
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(OUTPUT_DIR) / "checkpoints",
        filename="vae-{epoch:03d}-{loss_total:.2f}",
        save_top_k=3,
        monitor="loss_total",
        mode="min",
        save_last=True
    )
    
    #ema_callback = EMACallback(decay=0.9999,target_module_names=["generator_s1", "generator_s2", "ca_net", "char_embedding_layer"])
    ema_callback = EMACallback(decay=0.9999)

    trainer_args = {
        "max_epochs": N_EPOCHS,
        "callbacks": [checkpoint_callback, ema_callback],
        "logger": L.pytorch.loggers.WandbLogger(project="simple_vae", name=Path(OUTPUT_DIR).name),
        "log_every_n_steps": 20, # Adjusted from 50
        "enable_progress_bar": True,
    }
    if torch.cuda.is_available():
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = 1 # Assuming single GPU, adjust if multi-GPU
        # "bf16-mixed" can be very beneficial if supported, otherwise "16-mixed" or "32-true"
        trainer_args["precision"] = "16-mixed" if torch.cuda.is_bf16_supported() else "32-true"

    else:
        trainer_args["accelerator"] = "cpu"
        print("警告: GPUが利用できません。CPUで学習を実行します。")

    trainer = L.Trainer(**trainer_args)

    print(f"学習を開始します... (エポック数: {N_EPOCHS}, バッチサイズ: {BATCH_SIZE})")
    print(f"ログは {LOG_DIR} に、成果物は {OUTPUT_DIR} に保存されます。")
    
    try:
        #trainer.fit(model=stackgan_model, train_dataloaders=dataloader)
        trainer.fit(model=vae_model, train_dataloaders=dataloader)
    except Exception as e:
        print(f"学習中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("学習が終了しました。")
