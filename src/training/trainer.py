#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Tuple, Optional
import yaml

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.arch.kuzushiji_recognizer import create_model
from src.data.dataloader import create_data_loaders

class KuzushijiTrainer(pl.LightningModule):
    """くずし字ページ認識モデルの訓練用Lightning Module"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model = create_model(config)
        self.save_hyperparameters()
        
        # 損失関数の重み
        self.position_weight = config['training']['loss_weights']['position']
        self.unicode_weight = config['training']['loss_weights']['unicode']
        
        # メトリクスの初期化
        metrics = {}
        for stage in ['train', 'val', 'test']:
            metrics[f'{stage}_char_acc'] = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=config['output']['vocab_size']
            )
            metrics[f'{stage}_pos_iou'] = torchmetrics.JaccardIndex(
                task="multilabel",
                num_labels=1
            )
        self.metrics = torch.nn.ModuleDict(metrics)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def _calculate_position_loss(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        num_chars: torch.Tensor
    ) -> torch.Tensor:
        """位置の損失を計算"""
        # マスクの作成（有効な文字のみを考慮）
        batch_size = pred_pos.size(0)
        max_chars = pred_pos.size(1)
        mask = torch.arange(max_chars, device=pred_pos.device)[None, :] < num_chars[:, None]
        
        # L1損失の計算
        l1_loss = F.l1_loss(pred_pos, true_pos, reduction='none')
        l1_loss = (l1_loss * mask.unsqueeze(-1)).mean()
        
        return l1_loss
    
    def _calculate_unicode_loss(
        self,
        pred_unicode: torch.Tensor,
        true_unicode: torch.Tensor,
        num_chars: torch.Tensor
    ) -> torch.Tensor:
        """Unicode予測の損失を計算"""
        # マスクの作成
        batch_size = pred_unicode.size(0)
        max_chars = pred_unicode.size(1)
        mask = torch.arange(max_chars, device=pred_unicode.device)[None, :] < num_chars[:, None]
        
        # クロスエントロピー損失の計算
        pred_unicode = pred_unicode.view(-1, pred_unicode.size(-1))
        true_unicode = true_unicode.view(-1)
        mask = mask.view(-1)
        
        loss = F.cross_entropy(
            pred_unicode[mask],
            true_unicode[mask],
            label_smoothing=0.1
        )
        
        return loss
    
    def _calculate_metrics(
        self,
        stage: str,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        pred_unicode: torch.Tensor,
        true_unicode: torch.Tensor,
        num_chars: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """評価メトリクスの計算"""
        # マスクの作成
        batch_size = pred_pos.size(0)
        max_chars = pred_pos.size(1)
        mask = torch.arange(max_chars, device=pred_pos.device)[None, :] < num_chars[:, None]
        
        # 文字認識の精度
        pred_unicode_masked = pred_unicode[mask]
        true_unicode_masked = true_unicode[mask]
        self.metrics[f'{stage}_char_acc'](
            pred_unicode_masked.softmax(dim=-1),
            true_unicode_masked
        )
        
        # 位置推定のIoU
        pred_boxes = pred_pos[mask]
        true_boxes = true_pos[mask]
        self.metrics[f'{stage}_pos_iou'](pred_boxes, true_boxes)
        
        return {
            f"{stage}_char_acc": self.metrics[f'{stage}_char_acc'],
            f"{stage}_pos_iou": self.metrics[f'{stage}_pos_iou']
        }
    
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """訓練ステップ"""
        outputs = self(batch['image'])
        
        # 損失の計算
        pos_loss = self._calculate_position_loss(
            outputs['positions'],
            batch['positions'],
            batch['num_chars']
        )
        
        unicode_loss = self._calculate_unicode_loss(
            outputs['unicode_logits'],
            batch['unicodes'],
            batch['num_chars']
        )
        
        # 重み付け損失
        total_loss = (
            self.position_weight * pos_loss +
            self.unicode_weight * unicode_loss
        )
        
        # メトリクスの計算
        metrics = self._calculate_metrics(
            'train',
            outputs['positions'],
            batch['positions'],
            outputs['unicode_logits'],
            batch['unicodes'],
            batch['num_chars']
        )
        
        # ログの記録
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_pos_loss', pos_loss, prog_bar=True)
        self.log('train_unicode_loss', unicode_loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        
        return {
            'loss': total_loss,
            **metrics
        }
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """検証ステップ"""
        outputs = self(batch['image'])
        
        # 損失の計算
        pos_loss = self._calculate_position_loss(
            outputs['positions'],
            batch['positions'],
            batch['num_chars']
        )
        
        unicode_loss = self._calculate_unicode_loss(
            outputs['unicode_logits'],
            batch['unicodes'],
            batch['num_chars']
        )
        
        # 重み付け損失
        total_loss = (
            self.position_weight * pos_loss +
            self.unicode_weight * unicode_loss
        )
        
        # メトリクスの計算
        metrics = self._calculate_metrics(
            'val',
            outputs['positions'],
            batch['positions'],
            outputs['unicode_logits'],
            batch['unicodes'],
            batch['num_chars']
        )
        
        # ログの記録
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_pos_loss', pos_loss)
        self.log('val_unicode_loss', unicode_loss)
        self.log_dict(metrics, prog_bar=True)
        
        return {
            'val_loss': total_loss,
            **metrics
        }
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """テストステップ"""
        outputs = self(batch['image'])
        
        # メトリクスの計算
        metrics = self._calculate_metrics(
            'test',
            outputs['positions'],
            batch['positions'],
            outputs['unicode_logits'],
            batch['unicodes'],
            batch['num_chars']
        )
        
        # ログの記録
        self.log_dict(metrics)
        
        return metrics
    
    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['optimizer']['lr'],
            weight_decay=self.config['training']['optimizer']['weight_decay'],
            betas=self.config['training']['optimizer']['betas']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['scheduler']['T_max'],
            eta_min=self.config['training']['scheduler']['eta_min']
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train(
    config_path: str,
    data_dir: str,
    exp_dir: str,
    max_epochs: Optional[int] = None
) -> None:
    """モデルの訓練を実行"""
    
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
        max_epochs=max_epochs or config['training'].get('max_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        precision=16,  # 混合精度学習
        gradient_clip_val=config['training']['gradient_clip']['max_norm'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        deterministic=True  # 再現性確保
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