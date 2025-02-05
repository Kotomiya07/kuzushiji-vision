#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models import swin_v2_b

class PositionalEncoding2D(nn.Module):
    """2次元位置エンコーディング"""
    def __init__(
        self,
        d_model: int,
        max_h: int = 2048,
        max_w: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        h_pos = torch.arange(max_h).unsqueeze(1)
        w_pos = torch.arange(max_w).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model))
        
        h_pe = torch.zeros(max_h, d_model)
        h_pe[:, 0::2] = torch.sin(h_pos * div_term)
        h_pe[:, 1::2] = torch.cos(h_pos * div_term)
        
        w_pe = torch.zeros(max_w, d_model)
        w_pe[:, 0::2] = torch.sin(w_pos * div_term)
        w_pe[:, 1::2] = torch.cos(w_pos * div_term)
        
        self.register_buffer('h_pe', h_pe.unsqueeze(1))  # [H, 1, D]
        self.register_buffer('w_pe', w_pe.unsqueeze(0))  # [1, W, D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, D]
        Returns:
            [B, H, W, D]
        """
        x = x + self.h_pe[:x.size(1)] + self.w_pe[:, :x.size(2)]
        return self.dropout(x)

class LearnedPositionalEncoding2D(nn.Module):
    """学習可能な2次元位置エンコーディング"""
    def __init__(
        self,
        d_model: int,
        max_h: int = 2048,
        max_w: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.h_embed = nn.Embedding(max_h, d_model)
        self.w_embed = nn.Embedding(max_w, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, D]
        Returns:
            [B, H, W, D]
        """
        h_pos = torch.arange(x.size(1), device=x.device)
        w_pos = torch.arange(x.size(2), device=x.device)
        
        h_emb = self.h_embed(h_pos).unsqueeze(1)  # [H, 1, D]
        w_emb = self.w_embed(w_pos).unsqueeze(0)  # [1, W, D]
        
        x = x + h_emb + w_emb
        return self.dropout(x)

class CharacterQueryEmbedding(nn.Module):
    """文字検出用のクエリ埋め込み"""
    def __init__(
        self,
        d_model: int,
        num_queries: int = 1000
    ):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.pos_embed = nn.Embedding(num_queries, 4)  # x, y, width, height
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries: [B, N, D]
            pos_queries: [B, N, 4]
        """
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        pos_queries = self.pos_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return queries, pos_queries

class KuzushijiPageRecognizer(nn.Module):
    """くずし字ページ認識モデル"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # バックボーン (Swin Transformer V2)
        self.backbone = swin_v2_b(
            pretrained=config['architecture']['backbone']['pretrained'],
            window_size=config['architecture']['backbone']['window_size'],
        )
        
        # バックボーンの出力次元
        backbone_dim = self.backbone.num_features
        hidden_dim = config['architecture']['encoder']['hidden_dim']
        
        # 特徴マップの次元変換
        self.feature_proj = nn.Linear(backbone_dim, hidden_dim)
        
        # 位置エンコーディング
        pos_enc_type = config['architecture']['encoder']['position_encoding']['type']
        max_h = config['architecture']['encoder']['position_encoding']['max_h']
        max_w = config['architecture']['encoder']['position_encoding']['max_w']
        
        if pos_enc_type == 'learned_2d':
            self.pos_encoding = LearnedPositionalEncoding2D(
                hidden_dim, max_h, max_w,
                dropout=config['architecture']['encoder']['dropout']
            )
        else:
            self.pos_encoding = PositionalEncoding2D(
                hidden_dim, max_h, max_w,
                dropout=config['architecture']['encoder']['dropout']
            )
        
        # 文字クエリ埋め込み
        self.query_embed = CharacterQueryEmbedding(
            hidden_dim,
            num_queries=config['architecture']['decoder']['num_query_positions']
        )
        
        # Transformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config['architecture']['encoder']['nhead'],
            dim_feedforward=config['architecture']['encoder']['dim_feedforward'],
            dropout=config['architecture']['encoder']['dropout'],
            activation=config['architecture']['encoder']['activation']
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['architecture']['encoder']['num_layers']
        )
        
        # Transformerデコーダー
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=config['architecture']['decoder']['nhead'],
            dim_feedforward=config['architecture']['decoder']['dim_feedforward'],
            dropout=config['architecture']['decoder']['dropout'],
            activation=config['architecture']['decoder']['activation']
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['architecture']['decoder']['num_layers']
        )
        
        # 出力ヘッド
        self.position_head = nn.Linear(hidden_dim, 4)  # x, y, width, height
        self.unicode_head = nn.Linear(hidden_dim, config['output']['vocab_size'])
        
    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 入力画像 [B, C, H, W]
            output_attentions: アテンションマップを出力するかどうか
        Returns:
            positions: 文字位置 [B, N, 4] (x, y, width, height)
            unicode_logits: Unicode予測 [B, N, vocab_size]
            attentions: アテンションマップ (オプション)
        """
        B = x.size(0)
        
        # バックボーンで特徴抽出
        features = self.backbone.forward_features(x)  # [B, H*W, D]
        H = W = int(math.sqrt(features.size(1)))
        features = features.view(B, H, W, -1)  # [B, H, W, D]
        
        # 特徴次元の変換と位置エンコーディング
        features = self.feature_proj(features)  # [B, H, W, D]
        features = self.pos_encoding(features)  # [B, H, W, D]
        
        # エンコーダー入力の準備
        features = features.view(B, H*W, -1).permute(1, 0, 2)  # [H*W, B, D]
        
        # エンコーダー処理
        memory = self.encoder(features)  # [H*W, B, D]
        
        # デコーダーのクエリを準備
        queries, pos_queries = self.query_embed(B)  # [B, N, D], [B, N, 4]
        queries = queries.permute(1, 0, 2)  # [N, B, D]
        
        # デコーダー処理
        decoder_output = self.decoder(
            queries,
            memory
        )  # [N, B, D]
        
        # 出力の予測
        decoder_output = decoder_output.permute(1, 0, 2)  # [B, N, D]
        positions = self.position_head(decoder_output)  # [B, N, 4]
        unicode_logits = self.unicode_head(decoder_output)  # [B, N, vocab_size]
        
        # シグモイド関数で位置を[0, 1]に正規化
        positions = torch.sigmoid(positions)
        
        outputs = {
            "positions": positions,
            "unicode_logits": unicode_logits
        }
        
        return outputs
    
    @torch.no_grad()
    def decode_page(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """ページ内の文字を検出して認識"""
        outputs = self.forward(x)
        
        positions = outputs["positions"]  # [1, N, 4]
        unicode_logits = outputs["unicode_logits"]  # [1, N, vocab_size]
        
        # 確信度の計算
        position_conf = positions.mean(dim=-1)  # [1, N]
        unicode_conf = F.softmax(unicode_logits, dim=-1).max(dim=-1)[0]  # [1, N]
        confidence = (position_conf * unicode_conf).squeeze(0)  # [N]
        
        # 確信度が閾値以上の検出結果を抽出
        valid_mask = confidence > confidence_threshold
        valid_positions = positions[0, valid_mask]  # [M, 4]
        valid_unicode = unicode_logits[0, valid_mask].argmax(dim=-1)  # [M]
        valid_conf = confidence[valid_mask]  # [M]
        
        # 右から左、上から下の順にソート（日本語の縦書き）
        x_center = valid_positions[:, 0]
        y_center = valid_positions[:, 1]
        sort_idx = torch.lexsort((y_center, -x_center))
        
        results = []
        for idx in sort_idx:
            results.append({
                "position": valid_positions[idx].tolist(),
                "unicode": int(valid_unicode[idx]),
                "confidence": float(valid_conf[idx])
            })
        
        return results

def create_model(config: Dict) -> KuzushijiPageRecognizer:
    """モデルを作成"""
    return KuzushijiPageRecognizer(config)

def load_model(checkpoint_path: str, config: Dict) -> KuzushijiPageRecognizer:
    """チェックポイントからモデルを読み込み"""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
