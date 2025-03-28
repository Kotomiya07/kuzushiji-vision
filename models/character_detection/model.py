import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from transformers import ViTModel, ViTConfig


class CharacterDetectionModel(nn.Module):
    """Vision Transformerをベースにした文字検出モデル"""

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): モデルの設定
        """
        super().__init__()
        self.config = config

        # ViTの設定
        vit_config = ViTConfig(
            image_size=config["model"]["input_size"][0],  # パッチ分割の基準サイズ
            patch_size=config["model"]["patch_size"],
            num_channels=3,
            num_attention_heads=config["model"]["num_heads"],
            num_hidden_layers=config["model"]["num_layers"],
            hidden_size=config["model"]["hidden_size"],
            mlp_ratio=config["model"]["mlp_ratio"],
            hidden_dropout_prob=config["model"]["dropout"],
            attention_probs_dropout_prob=config["model"]["attention_dropout"],
            use_relative_position_bias=True,  # 相対位置エンコーディングを使用
            interpolate_pos_encoding=True,  # 位置エンコーディングを補間
        )

        # ViTモデルの初期化
        self.backbone = ViTModel(vit_config)

        # 文字検出ヘッド
        hidden_size = config["model"]["hidden_size"]
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(hidden_size, 5),  # x1, y1, x2, y2, confidence
        )

        # 文字分類ヘッド
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(hidden_size, config["model"]["num_classes"]),
        )

    def forward(self, images: torch.Tensor, targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """順伝播

        Args:
            images (torch.Tensor): 入力画像 [B, C, H, W]
            targets (Dict[str, torch.Tensor], optional):
                学習時のターゲット情報
                - boxes: バウンディングボックス [B, N, 4]
                - labels: ラベル [B, N]
                Defaults to None.

        Returns:
            Dict[str, torch.Tensor]:
                学習時: 損失の辞書
                推論時: 検出結果の辞書
        """
        # ViTによる特徴抽出（interpolate_pos_encodingを有効化）
        features = self.backbone(images, interpolate_pos_encoding=True, return_dict=True).last_hidden_state  # [B, N, D]

        # パッチごとの予測
        detection = self.detection_head(features)  # [B, N, 5]
        classification = self.classification_head(features)  # [B, N, C]

        if self.training and targets is not None:
            # 学習時の損失計算
            detection_loss = self._compute_detection_loss(detection, targets["boxes"])
            classification_loss = self._compute_classification_loss(
                classification=classification,
                detection=detection,
                targets=targets
            )

            return {
                "detection_loss": detection_loss,
                "classification_loss": classification_loss,
                "loss": detection_loss + classification_loss,
            }
        else:
            # 推論時の後処理
            boxes, scores, labels = self._post_process(detection, classification)

            return {
                "boxes": boxes,  # List[Tensor[N, 4]]
                "scores": scores,  # List[Tensor[N]]
                "labels": labels,  # List[Tensor[N]]
            }

    def _compute_detection_loss(self, predictions: torch.Tensor, targets: list) -> torch.Tensor:
        """検出の損失を計算

        Args:
            predictions (torch.Tensor): 予測値 [B, N, 5]
            targets (list): バッチ内の各画像のバウンディングボックスのリスト
                          各要素は [M, 4] のTensor

        Returns:
            torch.Tensor: 損失値
        """
        batch_size = predictions.size(0)
        device = predictions.device
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]

        total_pos_loss = 0
        total_conf_loss = 0
        valid_samples = 0

        for i in range(batch_size):
            if len(targets[i]) == 0:
                # ターゲットが存在しない場合
                conf_loss = F.binary_cross_entropy_with_logits(
                    pred_conf[i],
                    torch.zeros_like(pred_conf[i]),
                    reduction="mean"
                )
                total_conf_loss += conf_loss
                continue

            # ターゲットをTensorに変換
            target_boxes = targets[i].to(device)  # [M, 4]

            # IoUの計算
            ious = self._compute_iou(
                pred_boxes[i].unsqueeze(0),  # [1, N, 4]
                target_boxes.unsqueeze(0)     # [1, M, 4]
            ).squeeze(0)  # [N, M]

            # 各予測に対する最適なターゲットの割り当て
            max_ious, target_indices = ious.max(dim=1)  # [N]

            # 位置の損失（L1損失）
            pos_mask = max_ious > 0.5
            if pos_mask.sum() > 0:
                matched_targets = target_boxes[target_indices[pos_mask]]
                pos_loss = F.l1_loss(
                    pred_boxes[i][pos_mask],
                    matched_targets,
                    reduction="mean"
                )
                total_pos_loss += pos_loss
                valid_samples += 1

            # 信頼度の損失（バイナリクロスエントロピー）
            conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf[i],
                (max_ious > 0.5).float(),
                reduction="mean"
            )
            total_conf_loss += conf_loss

        # 最終的な損失の計算
        avg_conf_loss = total_conf_loss / batch_size
        avg_pos_loss = total_pos_loss / max(valid_samples, 1)

        return avg_pos_loss + avg_conf_loss

    def _compute_classification_loss(self, classification: torch.Tensor, detection: torch.Tensor, targets: Dict[str, list]) -> torch.Tensor:
        """分類の損失を計算

        Args:
            classification (torch.Tensor): 分類の予測値 [B, N, C]
            detection (torch.Tensor): 検出の予測値 [B, N, 5]
            targets (Dict[str, list]): ターゲット情報
                - boxes: バウンディングボックスのリスト
                - labels: ラベルのリスト

        Returns:
            torch.Tensor: 損失値
        """
        batch_size = classification.size(0)
        device = classification.device
        total_loss = 0
        valid_samples = 0

        for i in range(batch_size):
            if len(targets["boxes"][i]) == 0:
                continue

            # IoUの計算
            pred_boxes = detection[i, :, :4]  # [N, 4]
            target_boxes = targets["boxes"][i].to(device)  # [M, 4]

            ious = self._compute_iou(
                pred_boxes.unsqueeze(0),  # [1, N, 4]
                target_boxes.unsqueeze(0)  # [1, M, 4]
            ).squeeze(0)  # [N, M]

            # 各予測に対する最適なターゲットの割り当て
            max_ious, target_indices = ious.max(dim=1)  # [N]
            pos_mask = max_ious > 0.5

            if pos_mask.sum() > 0:
                # 分類の損失を計算（正例のみ）
                matched_labels = targets["labels"][i][target_indices[pos_mask]]
                pred_labels = classification[i, pos_mask]
                loss = F.cross_entropy(
                    pred_labels,
                    matched_labels,
                    reduction="mean"
                )
                total_loss += loss
                valid_samples += 1

        # 最終的な損失の計算
        return total_loss / max(valid_samples, 1)

    @staticmethod
    def _compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """IoU（Intersection over Union）を計算

        Args:
            boxes1 (torch.Tensor): バウンディングボックス1 [B, N, 4]
            boxes2 (torch.Tensor): バウンディングボックス2 [B, M, 4]

        Returns:
            torch.Tensor: IoU [B, N, M]
        """
        # 座標の展開
        x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(2).chunk(4, dim=-1)  # [B, N, 1, 1]
        x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(1).chunk(4, dim=-1)  # [B, 1, M, 1]

        # 共通部分の座標を計算
        xA = torch.max(x1_1, x1_2)
        yA = torch.max(y1_1, y1_2)
        xB = torch.min(x2_1, x2_2)
        yB = torch.min(y2_1, y2_2)

        # 共通部分の面積を計算
        intersection = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        # それぞれの面積を計算
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # IoUを計算
        iou = intersection / (area1 + area2 - intersection + 1e-6)

        return iou.squeeze(-1)  # [B, N, M]

    def _post_process(self, detection: torch.Tensor, classification: torch.Tensor) -> Tuple[list, list, list]:
        """推論時の後処理

        Args:
            detection (torch.Tensor): 検出結果 [B, N, 5]
            classification (torch.Tensor): 分類結果 [B, N, C]

        Returns:
            Tuple[list, list, list]:
                - boxes: バウンディングボックスのリスト
                - scores: スコアのリスト
                - labels: ラベルのリスト
        """
        batch_size = detection.size(0)
        boxes = []
        scores = []
        labels = []

        for i in range(batch_size):
            # 信頼度によるフィルタリング
            conf = torch.sigmoid(detection[i, :, 4])
            mask = conf > self.config["model"]["confidence_threshold"]

            if mask.sum() == 0:
                # 検出なしの場合
                boxes.append(torch.zeros((0, 4), device=detection.device))
                scores.append(torch.zeros(0, device=detection.device))
                labels.append(torch.zeros(0, dtype=torch.long, device=detection.device))
                continue

            # 検出結果の抽出
            filtered_boxes = detection[i, mask, :4]
            filtered_scores = conf[mask]

            # クラス予測
            filtered_labels = classification[i, mask].argmax(dim=1)

            # NMSの適用
            keep = self._nms(filtered_boxes, filtered_scores, self.config["model"]["nms_threshold"])

            boxes.append(filtered_boxes[keep])
            scores.append(filtered_scores[keep])
            labels.append(filtered_labels[keep])

        return boxes, scores, labels

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """Non-Maximum Suppression

        Args:
            boxes (torch.Tensor): バウンディングボックス [N, 4]
            scores (torch.Tensor): スコア [N]
            threshold (float): IoUのしきい値

        Returns:
            torch.Tensor: 選択されたインデックス
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        # 面積の計算
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # スコアでソート
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0]
            keep.append(i)

            # 残りのボックスとのIoUを計算
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # しきい値以下のものを残す
            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        return torch.tensor(keep, dtype=torch.long)
