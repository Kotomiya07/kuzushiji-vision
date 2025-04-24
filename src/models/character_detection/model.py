import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel


class CharacterDetectionModel(nn.Module):
    """Vision Transformerをベースにした文字検出モデル"""

    def __init__(self, config: dict):
        """
        Args:
            config (dict): モデルの設定
        """
        super().__init__()
        self.config = config
        self.current_epoch = 0  # エポック数を追跡

        # 事前学習済みViTモデルの設定
        model_config = ViTConfig(
            image_size=config.model.input_size[0],
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_layers,
            num_attention_heads=config.model.num_heads,
            intermediate_size=config.model.hidden_size * 4,  # mlp_ratioの代わり
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.attention_dropout,
            num_channels=3,
        )

        # 事前学習済みViTモデルの読み込み
        self.backbone = ViTModel.from_pretrained(
            config.model.backbone,
            config=model_config,
            ignore_mismatched_sizes=True,
        )

        # バックボーンの最初の数層を固定（事前学習の特徴を保持）
        num_layers_to_freeze = 6  # 最初の6層を固定
        # ViTModelの構造に合わせて修正 (embeddings + 最初のN層)
        modules_to_freeze = [self.backbone.embeddings] + list(self.backbone.encoder.layer[:num_layers_to_freeze])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        # 文字検出ヘッド
        hidden_size = config.model.hidden_size
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_size, 5),  # x1, y1, x2, y2, confidence
        )

        # 文字分類ヘッド
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_size, config.model.num_classes),
        )

    def forward(self, images: torch.Tensor, targets: dict[str, list[torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        """順伝播

        Args:
            images (torch.Tensor): 入力画像 [B, C, H, W]
            targets (dict[str, list[torch.Tensor]], optional):
                学習時のターゲット情報
                - boxes: バウンディングボックス [B, N, 4]
                - labels: ラベル [B, N]
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]:
                学習時: 損失の辞書
                推論時: 検出結果の辞書
        """
        # ViTによる特徴抽出（interpolate_pos_encodingを有効化）
        features = self.backbone(images, interpolate_pos_encoding=True, return_dict=True).last_hidden_state  # [B, N, D]

        # CLSトークンを除外 (通常はインデックス0)
        patch_features = features[:, 1:, :]  # [B, N_patches, D]

        # パッチごとの予測
        detection = self.detection_head(patch_features)  # [B, N_patches, 5]
        classification = self.classification_head(patch_features)  # [B, N_patches, C]

        if self.training and targets is not None:
            # 学習時の損失計算
            detection_loss = self._compute_detection_loss(detection, targets["boxes"])
            classification_loss = self._compute_classification_loss(
                classification=classification, detection=detection, targets=targets
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

    def set_epoch(self, epoch: int):
        """現在のエポック数を設定

        Args:
            epoch (int): 現在のエポック数
        """
        self.current_epoch = epoch

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
        batch_size = predictions.size(0)
        device = predictions.device
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]

        total_pos_loss = 0
        total_conf_loss = 0
        total_giou_loss = 0  # GIoU損失を追加
        valid_samples = 0

        # 設定に基づいてIoUしきい値を決定
        if self.config.model.dynamic_iou_threshold:
            params = self.config.model.dynamic_iou_params
            progress = min(1.0, self.current_epoch / params.epochs)
            iou_threshold = params.start + (params.end - params.start) * progress
            iou_threshold = max(params.start, min(params.end, iou_threshold))  # 範囲内に収める
        else:
            iou_threshold = self.config.model.fixed_iou_threshold

        for i in range(batch_size):
            if len(targets[i]) == 0:
                # ターゲットが存在しない場合
                conf_loss = F.binary_cross_entropy_with_logits(pred_conf[i], torch.zeros_like(pred_conf[i]), reduction="mean")
                total_conf_loss += conf_loss
                continue

            # ターゲットをTensorに変換
            target_boxes = targets[i].to(device)  # [M, 4]

            # IoUの計算
            ious = self._compute_iou(
                pred_boxes[i].unsqueeze(0),  # [1, N, 4]
                target_boxes.unsqueeze(0),  # [1, M, 4]
            ).squeeze(0)  # [N, M]

            # 各予測に対する最適なターゲットの割り当て
            max_ious, target_indices = ious.max(dim=1)  # [N]
            # 各予測に対する最適なターゲットの割り当て
            max_ious, target_indices = ious.max(dim=1)  # [N]

            # 位置の損失（L1損失とGIoU損失の組み合わせ）
            pos_mask = max_ious > iou_threshold
            if pos_mask.sum() > 0:
                matched_targets = target_boxes[target_indices[pos_mask]]

                # L1損失
                pos_loss = F.l1_loss(pred_boxes[i][pos_mask], matched_targets, reduction="mean")

                # GIoU損失
                giou_loss = 1 - self._compute_giou(pred_boxes[i][pos_mask], matched_targets).mean()

                total_pos_loss += pos_loss
                total_giou_loss += giou_loss
                valid_samples += 1

            # 信頼度の損失（フォーカル損失を使用）
            target_conf = (max_ious > iou_threshold).float()
            pred_prob = torch.sigmoid(pred_conf[i])

            # フォーカル損失の計算
            bce_loss = F.binary_cross_entropy_with_logits(pred_conf[i], target_conf, reduction="none")

            p_t = torch.where(target_conf > 0.5, pred_prob, 1 - pred_prob)
            alpha = 0.25
            gamma = 2.0
            alpha_t = torch.where(
                target_conf > 0.5, torch.tensor(alpha, device=device), torch.tensor(1 - alpha, device=device)
            )
            focal_weight = (1 - p_t) ** gamma

            # 最終的な損失（重みは勾配から切り離す）
            conf_loss = (focal_weight.detach() * alpha_t * bce_loss).mean()
            total_conf_loss += conf_loss

        # 最終的な損失の計算（重み付き合計）
        avg_pos_loss = total_pos_loss / max(valid_samples, 1)
        avg_giou_loss = total_giou_loss / max(valid_samples, 1)
        avg_conf_loss = total_conf_loss / batch_size

        return 0.5 * avg_pos_loss + 0.5 * avg_giou_loss + avg_conf_loss

    def _compute_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """GIoU（Generalized Intersection over Union）を計算

        Args:
            boxes1 (torch.Tensor): 予測ボックス [N, 4]
            boxes2 (torch.Tensor): 正解ボックス [N, 4]

        Returns:
            torch.Tensor: GIoU [N]
        """
        # ボックスの座標を取得
        x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(-1)
        x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(-1)

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
        union = area1 + area2 - intersection

        # 最小包含矩形の座標を計算
        xC = torch.min(x1_1, x1_2)
        yC = torch.min(y1_1, y1_2)
        xD = torch.max(x2_1, x2_2)
        yD = torch.max(y2_1, y2_2)

        # 最小包含矩形の面積を計算
        enclosing_area = (xD - xC) * (yD - yC)

        # GIoUを計算
        iou = intersection / (union + 1e-6)
        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)

        return giou

    def _compute_classification_loss(
        self, classification: torch.Tensor, detection: torch.Tensor, targets: dict[str, list[torch.Tensor]]
    ) -> torch.Tensor:
        """分類の損失を計算

        Args:
            classification (torch.Tensor): 分類の予測値 [B, N, C]
            detection (torch.Tensor): 検出の予測値 [B, N, 5]
            targets (dict[str, list[torch.Tensor]]): ターゲット情報
                - boxes: バウンディングボックスのリスト
                - labels: ラベルのリスト

        Returns:
            torch.Tensor: 損失値
        """
        batch_size = classification.size(0)
        device = classification.device
        # Initialize total_loss as a zero tensor on the correct device
        total_loss = torch.tensor(0.0, device=device, dtype=classification.dtype)
        valid_samples = 0

        # 設定に基づいてIoUしきい値を決定 (detection_lossと同じロジック)
        if self.config.model.dynamic_iou_threshold:
            params = self.config.model.dynamic_iou_params
            progress = min(1.0, self.current_epoch / params.epochs)
            iou_threshold = params.start + (params.end - params.start) * progress
            iou_threshold = max(params.start, min(params.end, iou_threshold))  # 範囲内に収める
        else:
            iou_threshold = self.config.model.fixed_iou_threshold

        for i in range(batch_size):
            if len(targets["boxes"][i]) == 0:
                continue

            # IoUの計算
            pred_boxes = detection[i, :, :4]  # [N, 4]
            target_boxes = targets["boxes"][i].to(device)  # [M, 4]

            ious = self._compute_iou(
                pred_boxes.unsqueeze(0),  # [1, N, 4]
                target_boxes.unsqueeze(0),  # [1, M, 4]
            ).squeeze(0)  # [N, M]

            # 各予測に対する最適なターゲットの割り当て
            max_ious, target_indices = ious.max(dim=1)  # [N]
            # Use the same IoU threshold as detection loss
            pos_mask = max_ious > iou_threshold

            if pos_mask.sum() > 0:
                # 分類の損失を計算（正例のみ）
                matched_labels = targets["labels"][i][target_indices[pos_mask]]
                pred_labels = classification[i, pos_mask]
                loss = F.cross_entropy(pred_labels, matched_labels, reduction="mean")
                total_loss += loss
                valid_samples += 1

        # 最終的な損失の計算 (ensure tensor output even if valid_samples is 0)
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            # Return the initial zero tensor if no valid samples
            # Ensure it requires grad if it's part of the computation graph leading to the final loss
            # If total_loss is already a tensor initialized with requires_grad=True, this is fine.
            # Alternatively, return total_loss directly as it's already a zero tensor.
            return total_loss  # Return the zero tensor

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

    def _post_process(self, detection: torch.Tensor, classification: torch.Tensor) -> tuple[list, list, list]:
        """推論時の後処理

        Args:
            detection (torch.Tensor): 検出結果 [B, N, 5]
            classification (torch.Tensor): 分類結果 [B, N, C]

        Returns:
            tuple[list, list, list]:
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
            mask = conf > self.config.model.confidence_threshold

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
            keep = self._nms(filtered_boxes, filtered_scores, self.config.model.nms_threshold)

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
