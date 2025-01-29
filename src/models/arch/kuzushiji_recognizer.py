#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import List, Dict, Optional, Tuple, Union

class PositionalEncoding(nn.Module):
    """位置エンコーディング（学習可能またはサイン波ベース）"""
    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1,
        type: str = "learned"
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.type = type
        
        if type == "learned":
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:  # sinusoidal
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == "learned":
            return self.dropout(x + self.pe[:, :x.size(1)])
        return self.dropout(x + self.pe[:, :x.size(1)])

class FPN(nn.Module):
    """特徴ピラミッドネットワーク"""
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # 各レベルの特徴量を変換するための1x1畳み込み
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(num_levels)
        ])
        
        # 3x3畳み込みで特徴量を滑らかにする
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(num_levels)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        prev_features = None
        
        for i in range(len(features) - 1, -1, -1):
            lateral_feat = self.lateral_convs[i](features[i])
            
            if prev_features is not None:
                prev_features = F.interpolate(
                    prev_features,
                    size=lateral_feat.shape[-2:],
                    mode='nearest'
                )
                lateral_feat = lateral_feat + prev_features
                
            output = self.smooth_convs[i](lateral_feat)
            results.insert(0, output)
            prev_features = lateral_feat
            
        return results

class TransformerBase(nn.Module):
    """Transformerの基本構造"""
    def __init__(self, config: Dict):
        super().__init__()
        transformer_config = config['architecture']['transformer']
        hidden_dim = transformer_config['hidden_dim']
        nhead = transformer_config['nhead']
        dropout = transformer_config['dropout']
        activation = transformer_config['activation']
        
        # エンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=transformer_config['encoder']['dim_feedforward'],
            dropout=dropout,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config['encoder']['num_layers']
        )
        
        # デコーダー
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=transformer_config['decoder']['dim_feedforward'],
            dropout=dropout,
            activation=activation
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_config['decoder']['num_layers']
        )

class KuzushijiRecognizer(nn.Module):
    """くずし字認識モデル（エンコーダー・デコーダー構造）"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        transformer_config = config['architecture']['transformer']
        hidden_dim = transformer_config['hidden_dim']
        
        # ResNetバックボーン
        backbone = resnet50(pretrained=config['architecture']['backbone']['pretrained'])
        if config['architecture']['backbone']['freeze_layers'] > 0:
            for i, param in enumerate(backbone.parameters()):
                if i < config['architecture']['backbone']['freeze_layers']:
                    param.requires_grad = False
        
        # 必要な層のみ抽出
        self.backbone_layers = nn.ModuleList([
            nn.Sequential(*list(backbone.children())[:5]),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])
        
        # 特徴ピラミッドネットワーク
        self.fpn = FPN(
            in_channels=config['architecture']['feature_pyramid']['in_channels'],
            out_channels=config['architecture']['feature_pyramid']['out_channels'],
            num_levels=config['architecture']['feature_pyramid']['num_levels']
        )
        
        # Transformer
        self.transformer = TransformerBase(config)
        
        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=transformer_config['positional_encoding']['max_sequence_length'],
            type=transformer_config['positional_encoding']['type']
        )
        
        # デコーダーの入力埋め込み
        self.target_embedding = nn.Embedding(
            transformer_config['decoder']['vocab_size'],
            hidden_dim
        )
        
        # 出力層
        self.output_projection = nn.Linear(
            hidden_dim,
            transformer_config['decoder']['vocab_size']
        )
        
        # 特殊トークン
        self.sos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        
    def create_mask(self, size: int) -> torch.Tensor:
        """デコーダーのマスクを作成"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """画像特徴のエンコード"""
        # バックボーンから特徴抽出
        features = []
        for layer in self.backbone_layers:
            x = layer(x)
            features.append(x)
        
        # FPNで特徴統合
        fpn_features = self.fpn(features[-3:])
        
        # 特徴量を結合してTransformerに入力
        transformed_features = []
        for feat in fpn_features:
            B, C, H, W = feat.shape
            feat = feat.view(B, C, -1).permute(0, 2, 1)
            transformed_features.append(feat)
        
        encoder_input = torch.cat(transformed_features, dim=1)
        encoder_input = self.pos_encoding(encoder_input)
        
        # Transformerエンコーダーで処理
        memory = self.transformer.encoder(encoder_input)
        
        return memory
    
    def decode(
        self,
        memory: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        max_length: int = 100
    ) -> torch.Tensor:
        """デコード処理"""
        batch_size = memory.size(0)
        device = memory.device
        
        if self.training and target_ids is not None:
            # 訓練時：教師強制
            tgt = self.target_embedding(target_ids)
            tgt = self.pos_encoding(tgt)
            
            tgt_mask = self.create_mask(target_ids.size(1)).to(device)
            
            decoder_output = self.transformer.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask
            )
            
            return self.output_projection(decoder_output)
        
        else:
            # 推論時：自己回帰的生成
            decoded_ids = []
            curr_ids = torch.full(
                (batch_size, 1),
                self.sos_token_id,
                dtype=torch.long,
                device=device
            )
            
            for _ in range(max_length):
                tgt = self.target_embedding(curr_ids)
                tgt = self.pos_encoding(tgt)
                
                tgt_mask = self.create_mask(curr_ids.size(1)).to(device)
                
                decoder_output = self.transformer.decoder(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask
                )
                
                next_token_logits = self.output_projection(decoder_output[:, -1])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                decoded_ids.append(next_token)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                
                if (next_token == self.eos_token_id).any():
                    break
            
            return torch.cat(decoded_ids, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        max_length: int = 100
    ) -> torch.Tensor:
        """順伝播"""
        memory = self.encode(x)
        return self.decode(memory, target_ids, max_length)

def create_model(config: Dict) -> KuzushijiRecognizer:
    """モデルを作成"""
    return KuzushijiRecognizer(config)

def load_model(checkpoint_path: str, config: Dict) -> KuzushijiRecognizer:
    """チェックポイントからモデルを読み込み"""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
