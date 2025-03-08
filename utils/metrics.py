import torch
import numpy as np
from typing import List, Dict, Tuple, Union
from collections import defaultdict

def compute_iou(
    box1: Union[torch.Tensor, np.ndarray],
    box2: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """IoU（Intersection over Union）を計算

    Args:
        box1 (Union[torch.Tensor, np.ndarray]): バウンディングボックス1 [N, 4]
        box2 (Union[torch.Tensor, np.ndarray]): バウンディングボックス2 [M, 4]

    Returns:
        Union[torch.Tensor, np.ndarray]: IoU [N, M]
    """
    if isinstance(box1, torch.Tensor):
        return _compute_iou_torch(box1, box2)
    else:
        return _compute_iou_numpy(box1, box2)

def _compute_iou_torch(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """PyTorch版のIoU計算"""
    N = box1.size(0)
    M = box2.size(0)
    
    # 座標の展開
    box1 = box1.unsqueeze(1).expand(-1, M, -1)  # [N, M, 4]
    box2 = box2.unsqueeze(0).expand(N, -1, -1)  # [N, M, 4]
    
    # 共通部分の座標を計算
    xA = torch.max(box1[..., 0], box2[..., 0])
    yA = torch.max(box1[..., 1], box2[..., 1])
    xB = torch.min(box1[..., 2], box2[..., 2])
    yB = torch.min(box1[..., 3], box2[..., 3])
    
    # 共通部分の面積を計算
    intersection = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    
    # それぞれの面積を計算
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # IoUを計算
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    
    return iou

def _compute_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """NumPy版のIoU計算"""
    # N = box1.shape[0]
    # M = box2.shape[0]
    
    # 座標の展開
    box1 = np.expand_dims(box1, axis=1)  # [N, 1, 4]
    box2 = np.expand_dims(box2, axis=0)  # [1, M, 4]
    
    # 共通部分の座標を計算
    xA = np.maximum(box1[..., 0], box2[..., 0])
    yA = np.maximum(box1[..., 1], box2[..., 1])
    xB = np.minimum(box1[..., 2], box2[..., 2])
    yB = np.minimum(box1[..., 3], box2[..., 3])
    
    # 共通部分の面積を計算
    intersection = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    
    # それぞれの面積を計算
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # IoUを計算
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    
    return iou

def compute_map(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[int, float]]:
    """mAP（mean Average Precision）を計算

    Args:
        pred_boxes (List[torch.Tensor]): 予測バウンディングボックスのリスト
        pred_scores (List[torch.Tensor]): 予測スコアのリスト
        pred_labels (List[torch.Tensor]): 予測ラベルのリスト
        gt_boxes (List[torch.Tensor]): 正解バウンディングボックスのリスト
        gt_labels (List[torch.Tensor]): 正解ラベルのリスト
        iou_threshold (float, optional): IoUのしきい値. Defaults to 0.5.

    Returns:
        Tuple[float, Dict[int, float]]: 
            - mAP
            - クラスごとのAP
    """
    # クラスごとの予測と正解を収集
    class_preds = defaultdict(list)
    class_gts = defaultdict(list)
    
    # 各画像の予測と正解を処理
    for pred_box, pred_score, pred_label, gt_box, gt_label in zip(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    ):
        # クラスごとに予測を収集
        for box, score, label in zip(pred_box, pred_score, pred_label):
            class_preds[label.item()].append({
                'box': box,
                'score': score
            })
        
        # クラスごとに正解を収集
        for box, label in zip(gt_box, gt_label):
            class_gts[label.item()].append({
                'box': box,
                'matched': False
            })
    
    # クラスごとのAPを計算
    aps = {}
    for class_id in class_gts.keys():
        ap = compute_ap(
            class_preds[class_id],
            class_gts[class_id],
            iou_threshold
        )
        aps[class_id] = ap
    
    # mAPを計算
    mAP = sum(aps.values()) / len(aps) if aps else 0.0
    
    return mAP, aps

def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float
) -> float:
    """クラスごとのAP（Average Precision）を計算

    Args:
        predictions (List[Dict]): 予測のリスト
        ground_truths (List[Dict]): 正解のリスト
        iou_threshold (float): IoUのしきい値

    Returns:
        float: AP
    """
    # 予測をスコアでソート
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # 適合率と再現率を計算
    precisions = []
    recalls = []
    num_positives = len(ground_truths)
    num_correct = 0
    
    for i, pred in enumerate(predictions, 1):
        # 最も高いIoUを持つ未マッチの正解を探す
        max_iou = 0
        max_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if not gt['matched']:
                iou = compute_iou(
                    pred['box'].unsqueeze(0),
                    gt['box'].unsqueeze(0)
                )[0, 0]
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
        
        # IoUがしきい値を超える場合は正解とみなす
        if max_iou >= iou_threshold:
            num_correct += 1
            ground_truths[max_idx]['matched'] = True
        
        # 適合率と再現率を計算
        precision = num_correct / i
        recall = num_correct / num_positives if num_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 空の場合の処理
    if not precisions:
        return 0.0
    
    # 11点補間によるAP計算
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[np.where(recalls >= t)[0]])
        ap += p / 11.0
    
    return ap

def compute_character_accuracy(
    pred_labels: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    pred_boxes: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[int, float]]:
    """文字認識の精度を計算

    Args:
        pred_labels (List[torch.Tensor]): 予測ラベルのリスト
        gt_labels (List[torch.Tensor]): 正解ラベルのリスト
        pred_boxes (List[torch.Tensor]): 予測バウンディングボックスのリスト
        gt_boxes (List[torch.Tensor]): 正解バウンディングボックスのリスト
        iou_threshold (float, optional): IoUのしきい値. Defaults to 0.5.

    Returns:
        Tuple[float, Dict[int, float]]:
            - 全体の文字認識精度
            - クラスごとの文字認識精度
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred_label, gt_label, pred_box, gt_box in zip(
        pred_labels, gt_labels, pred_boxes, gt_boxes
    ):
        # IoUの計算
        ious = compute_iou(pred_box, gt_box)
        
        # IoUがしきい値を超えるペアを探す
        max_ious, matched_idx = ious.max(dim=1)
        valid_mask = max_ious >= iou_threshold
        
        # 正解とのマッチング
        for i, (pred, gt_idx, is_valid) in enumerate(zip(
            pred_label, matched_idx, valid_mask
        )):
            if is_valid:
                gt = gt_label[gt_idx]
                class_total[gt.item()] += 1
                if pred == gt:
                    class_correct[gt.item()] += 1
    
    # クラスごとの精度を計算
    class_accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for class_id in class_total.keys():
        accuracy = class_correct[class_id] / class_total[class_id]
        class_accuracies[class_id] = accuracy
        total_correct += class_correct[class_id]
        total_samples += class_total[class_id]
    
    # 全体の精度を計算
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return overall_accuracy, class_accuracies 