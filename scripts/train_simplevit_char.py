import datetime
import glob
import os
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from schedulefree import RAdamScheduleFree  # Keep existing optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

from src.callbacks.ema import EMACallback

# --- 設定項目 (Keep your existing settings, adjust as needed) ---
DATA_ROOT = "data/onechannel"
OUTPUT_DIR = f"experiments/simple_vae/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"{OUTPUT_DIR}/lightning_logs"

NUM_UNICODES = -1
UNICODE_TO_INT = {}
INT_TO_UNICODE = {}

LATENT_DIM = 100
# CHAR_EMBED_DIM = 128 # Not used in this VAE model
# CA_EMBED_DIM = 128 # Not used in this VAE model

IMG_SIZE = 128
CHANNELS = 1
BATCH_SIZE = 256  # Adjusted based on typical values and potential memory increase
LR = 1.6e-5
B1 = 0.9
B2 = 0.999
N_EPOCHS = 300
# LAMBDA_KL = 1.0 # Coefficient for KL divergence loss - current loss implies a weight of 1.0
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1


class CharUnicodeDataset(Dataset):
    def __init__(self, data_root: str, unicode_to_int_map: dict, image_size: int, channels: int = 1, transform=None):
        self.data_root = Path(data_root)
        self.unicode_to_int_map = unicode_to_int_map
        self.image_size = image_size
        self.channels = channels
        self.image_paths = []
        self.labels = []

        for unicode_char_dir, int_label in self.unicode_to_int_map.items():
            char_dir_path = self.data_root / unicode_char_dir
            if char_dir_path.is_dir():
                # Common image extensions
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]:
                    self.image_paths.extend(glob.glob(str(char_dir_path / ext)))
                    self.labels.extend([int_label] * len(glob.glob(str(char_dir_path / ext))))  # Add corresponding labels
            else:
                print(f"Warning: Directory not found for Unicode {unicode_char_dir}")

        # Filter out non-image files if any were accidentally caught by lenient glob patterns
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        filtered_paths_labels = [
            (p, l) for p, l in zip(self.image_paths, self.labels, strict=False) if Path(p).suffix.lower() in valid_extensions
        ]
        if filtered_paths_labels:
            self.image_paths, self.labels = zip(*filtered_paths_labels, strict=False)
        else:
            self.image_paths, self.labels = [], []

        if transform is None:
            transform_list = []
            if self.channels == 1:
                transform_list.append(T.Grayscale(num_output_channels=1))
            transform_list.extend(
                [
                    T.Resize((self.image_size, self.image_size), antialias=True),
                    T.ToTensor(),  # Scales to [0.0, 1.0]
                ]
            )
            # If VAE output uses tanh, add Normalize to [-1, 1]. Current uses Sigmoid [0,1].
            # if self.channels == 1:
            #     transform_list.append(T.Normalize(mean=[0.5], std=[0.5]))
            # else:
            #     transform_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            self.transform = T.Compose(transform_list)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Ensure image is converted to RGB first if it might be palette, then to L or RGB
            img = Image.open(img_path).convert("RGB")
            if self.channels == 1:
                image = img.convert("L")
            else:
                image = img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            dummy_image_tensor = torch.zeros((self.channels, self.image_size, self.image_size))
            return dummy_image_tensor, -1

        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleCVAELitModule(L.LightningModule):
    def __init__(
        self,
        input_channels=1,
        latent_dim=32,
        num_classes=NUM_UNICODES,  # Added number of classes
        learning_rate=LR,
        b1=B1,
        b2=B2,
    ):
        super().__init__()
        self.save_hyperparameters("input_channels", "latent_dim", "num_classes", "learning_rate", "b1", "b2")
        self.automatic_optimization = False

        # エンコーダー (Adjusted for 128x128 input)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
        )

        # 潜在空間の平均と分散 (条件付き)
        self.fc_mu = nn.Linear(512 * 4 * 4 + num_classes, latent_dim)
        self.fc_var = nn.Linear(512 * 4 * 4 + num_classes, latent_dim)

        # デコーダー (条件付き)
        self.decoder_fc = nn.Linear(latent_dim + num_classes, 512 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid(),  # Outputting images in [0, 1] range
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def encode(self, x, c):
        h = self.encoder(x)
        # Concatenate features with one-hot encoded class
        c_one_hot = F.one_hot(c, num_classes=self.hparams.num_classes).float()
        h_c = torch.cat([h, c_one_hot], dim=1)
        mu = self.fc_mu(h_c)
        log_var = self.fc_var(h_c)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate latent vector with one-hot encoded class
        c_one_hot = F.one_hot(c, num_classes=self.hparams.num_classes).float()
        z_c = torch.cat([z, c_one_hot], dim=1)
        h = F.relu(self.decoder_fc(z_c))
        return self.decoder_conv(h)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        BCE = self.criterion(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = BCE + beta * KLD
        return total_loss, BCE, KLD

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.train()

        x, labels = batch  # Now using labels
        x_recon, mu, log_var = self(x, labels)
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, log_var)

        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_kl_loss", kl_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        opt.eval()
        return total_loss

    def validation_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.eval()
        x, labels = batch
        x_recon, mu, log_var = self(x, labels)
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, log_var)

        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_recon_loss", recon_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # Log a few reconstructed images
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            if self.logger:
                num_images_to_log = min(x.size(0), 16)
                comparison = torch.cat([x[:num_images_to_log], x_recon[:num_images_to_log]])
                grid = make_grid(comparison, nrow=num_images_to_log)
                self.logger.log_image(key="samples", images=[grid])
                save_image(grid, Path(OUTPUT_DIR) / f"reconstruction_epoch_{self.current_epoch}.png")

        return total_loss

    def configure_optimizers(self):
        optimizer = RAdamScheduleFree(
            self.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.b1, self.hparams.b2)
        )
        optimizer.eval()
        return optimizer


# --- Unicode マッピングの準備とメイン処理 (prepare_unicode_dataは変更なし) ---
def prepare_unicode_data(data_root_path):
    global NUM_UNICODES, UNICODE_TO_INT, INT_TO_UNICODE  # Ensure these are global
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
    INT_TO_UNICODE = dict(enumerate(unicode_dirs))
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

    dataset = CharUnicodeDataset(
        data_root=DATA_ROOT, unicode_to_int_map=UNICODE_TO_INT, image_size=IMG_SIZE, channels=CHANNELS
    )
    if len(dataset) == 0:
        print("エラー: データセットが空です。 DATA_ROOT とその中の画像ファイルを確認してください。")
        exit(1)
    print(f"データセットサイズ: {len(dataset)} 画像")

    actual_num_workers = NUM_WORKERS if len(dataset) > BATCH_SIZE * NUM_WORKERS else 0
    # For single image per batch, persistent_workers can cause issues with small datasets if num_workers > 0
    use_persistent_workers = actual_num_workers > 0 and len(dataset) > BATCH_SIZE

    # train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=actual_num_workers,
        persistent_workers=use_persistent_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=actual_num_workers,
    #     persistent_workers=use_persistent_workers,
    #     pin_memory=torch.cuda.is_available()
    # )

    vae_model = SimpleCVAELitModule(
        input_channels=CHANNELS,
        latent_dim=LATENT_DIM,
        num_classes=NUM_UNICODES,
        learning_rate=LR,
        b1=B1,
        b2=B2,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(OUTPUT_DIR) / "checkpoints",
        filename="vae-{epoch:03d}",  # Monitor val_loss
        save_top_k=3,
        monitor="train_loss",  # Monitor validation loss
        mode="min",
        save_last=True,
    )

    ema_callback = EMACallback(decay=0.9999)  # This will be applied by Lightning

    # PrepareWandbLogger
    try:
        # Use the newer import path if available, fallback for compatibility
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        from lightning_fabric.loggers import WandbLogger  # Older path or fabric context

        print("Using WandbLogger from lightning_fabric.loggers")

    wandb_logger = WandbLogger(project="simple_vae_char", name=Path(OUTPUT_DIR).name, save_dir=LOG_DIR)

    trainer_args = {
        "max_epochs": N_EPOCHS,
        "callbacks": [checkpoint_callback, ema_callback],
        "logger": wandb_logger,
        "log_every_n_steps": 20,
        "enable_progress_bar": True,
    }
    if torch.cuda.is_available():
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = 1
        # User's original precision logic:
        # Use "16-mixed" (FP16) if BF16 hardware is present, otherwise "32-true".
        # A more common setup would be "bf16-mixed" if BF16 is supported, then "16-mixed", then "32-true".
        # Sticking to user's specified logic for now.
        if torch.cuda.is_bf16_supported():
            trainer_args["precision"] = "bf16-mixed"  # Prefer bf16 if available
        else:
            trainer_args["precision"] = "16-mixed"  # Fallback to fp16 if bf16 not available but GPU supports it
        # For GPUs that don't support bf16 or fp16 well, "32-true" is the safest.
        # The above assumes the GPU supports at least fp16 if bf16 is not available.
        # A safer default if unsure:
        # if torch.cuda.is_bf16_supported():
        #     trainer_args["precision"] = "bf16-mixed"
        # elif torch.cuda.get_device_capability(0)[0] >= 7: # Volta and later for good FP16
        #     trainer_args["precision"] = "16-mixed"
        # else:
        #     trainer_args["precision"] = "32-true"
        # For simplicity with user's original line, let's use:
        # trainer_args["precision"] = "16-mixed" if torch.cuda.is_bf16_supported() else "32-true"
        # Corrected common practice:
        # if torch.cuda.is_bf16_supported():
        #     trainer_args["precision"] = "bf16-mixed"
        # else: # No specific check for FP16, rely on Lightning to handle "16-mixed" if chosen or user can set manually
        # trainer_args["precision"] = "16-mixed" # if you know your GPU supports FP16 well
        # else: trainer_args["precision"] = "32-true"
        # The user had: "16-mixed" if torch.cuda.is_bf16_supported() else "32-true"
        # This means use fp16 if bf16 is available, which is slightly odd.
        # Let's assume they want highest available precision if bf16 isn't there, or meant to use bf16 when available.
        # For now, if bf16 is supported, use it. Otherwise, let lightning decide or default to 32.
        # Or more simply, let user configure "16-mixed" or "bf16-mixed" directly if desired.
        # To be safe and commonly effective:
        if torch.cuda.is_bf16_supported():
            print("Using 'bf16-mixed' precision as BFloat16 is supported.")
            trainer_args["precision"] = "bf16-mixed"
        else:
            print(
                "BFloat16 not supported. Consider '16-mixed' if your GPU supports FP16 well, or '32-true'. Defaulting to '32-true' for wider compatibility."
            )
            trainer_args["precision"] = "32-true"  # Fallback to 32 if bf16 is not supported. User can override.

    else:
        trainer_args["accelerator"] = "cpu"
        trainer_args["precision"] = "32-true"  # CPU usually uses 32-bit
        print("警告: GPUが利用できません。CPUで学習を実行します。")

    trainer = L.Trainer(**trainer_args)

    print(f"学習を開始します... (エポック数: {N_EPOCHS}, バッチサイズ: {BATCH_SIZE})")
    print(f"ログは {LOG_DIR} に、成果物は {OUTPUT_DIR} に保存されます。")

    try:
        trainer.fit(model=vae_model, train_dataloaders=train_dataloader)  # Using same dataloader for val for simplicity
    except Exception as e:
        print(f"学習中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("学習が終了しました。")
