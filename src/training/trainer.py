#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
from typing import Dict, Tuple, Optional

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.arch.kuzushiji_recognizer import create_model
from src.data.dataloader import create_data_loaders


class KuzushijiTrainer(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model = create_model(config)
        self.save_hyperparameters()
        
        # 損失関数
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config["training"]["loss"]["label_smoothing"]
        )
        
        # メトリクス計算用
        num_classes = config["architecture"]["transformer"]["decoder"]["vocab_size"]
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["optimizer"]["lr"],
            weight_decay=self.config["training"]["optimizer"]["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["scheduler"]["T_max"],
            eta_min=self.config["training"]["scheduler"]["eta_min"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.train_acc(y_hat.softmax(dim=-1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return {"loss": loss, "acc": acc}
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        acc = self.test_acc(y_hat.softmax(dim=-1), y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
        return {"test_loss": loss, "test_acc": acc}


def train(
    config_path: str,
    data_dir: str,
    exp_dir: str,
    max_epochs: Optional[int] = None
) -> None:
    # 設定の読み込み
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # データローダーの作成
    train_loader, val_loader, test_loader = create_data_loaders(config, data_dir)
    
    # モデルの作成
    model = KuzushijiTrainer(config)
    
    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(exp_dir, "checkpoints"),
            filename="kuzushiji-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ]
    
    # ロガーの設定
    logger = TensorBoardLogger(
        save_dir=os.path.join(exp_dir, "logs"),
        name="kuzushiji"
    )
    
    # トレーナーの設定と実行
    trainer = pl.Trainer(
        max_epochs=max_epochs or config["training"].get("max_epochs", 100),
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=logger,
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1)
    )
    
    # 訓練の実行
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # テストの実行
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="くずし字認識モデルの訓練")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="データセットのルートディレクトリ"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="実験結果を保存するディレクトリ"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="最大エポック数"
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_dir=args.data_dir,
        exp_dir=args.exp_dir,
        max_epochs=args.max_epochs
    )