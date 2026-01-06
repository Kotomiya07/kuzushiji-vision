# test_character_detection.py
"""
学習済みYOLO文字検出モデルのテストスクリプト

テストデータセットで推論を行い、精度が低い10件について
Ground Truth と Prediction を左右に並べて可視化・保存する
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """YOLOフォーマットのラベルファイルを読み込む

    Args:
        label_path: ラベルファイルのパス

    Returns:
        shape (N, 5) の配列 [class_id, x_center, y_center, width, height]
    """
    if not label_path.exists():
        return np.array([]).reshape(0, 5)

    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])

    return np.array(labels) if labels else np.array([]).reshape(0, 5)


def yolo_to_xyxy(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """YOLO形式 (cx, cy, w, h) normalized を (x1, y1, x2, y2) pixel 座標に変換

    Args:
        boxes: shape (N, 4) [cx, cy, w, h] normalized
        img_width: 画像幅
        img_height: 画像高さ

    Returns:
        shape (N, 4) [x1, y1, x2, y2] pixel座標
    """
    if len(boxes) == 0:
        return np.array([]).reshape(0, 4)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height

    return np.stack([x1, y1, x2, y2], axis=1)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """2つのbox集合間のIoU行列を計算

    Args:
        boxes1: shape (N, 4) [x1, y1, x2, y2]
        boxes2: shape (M, 4) [x1, y1, x2, y2]

    Returns:
        shape (N, M) のIoU行列
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.array([]).reshape(len(boxes1), len(boxes2))

    # 交差領域の計算
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 各ボックスの面積
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-6)


def compute_image_score(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray, iou_threshold: float = 0.5
) -> dict:
    """画像単位のスコアを計算

    Args:
        gt_boxes: Ground Truth boxes (N, 4) [x1, y1, x2, y2]
        pred_boxes: Prediction boxes (M, 4) [x1, y1, x2, y2]
        pred_scores: Prediction confidence scores (M,)
        iou_threshold: IoU閾値

    Returns:
        各種メトリクスを含む辞書
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0 and n_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_gt": 0, "n_pred": 0, "tp": 0, "fp": 0, "fn": 0}

    if n_gt == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "n_gt": 0, "n_pred": n_pred, "tp": 0, "fp": n_pred, "fn": 0}

    if n_pred == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "n_gt": n_gt, "n_pred": 0, "tp": 0, "fp": 0, "fn": n_gt}

    # IoU行列を計算
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # True Positiveのカウント（貪欲マッチング）
    tp = 0
    matched_gt = set()

    # 高いconfidenceから順に処理
    sorted_indices = np.argsort(-pred_scores)
    for pred_idx in sorted_indices:
        if len(matched_gt) == n_gt:
            break

        # 未マッチのGTの中で最大IoUを探す
        best_iou = 0
        best_gt_idx = -1
        for gt_idx in range(n_gt):
            if gt_idx not in matched_gt:
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / n_pred if n_pred > 0 else 1.0
    recall = tp / n_gt if n_gt > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "n_gt": n_gt, "n_pred": n_pred, "tp": tp, "fp": fp, "fn": fn}


def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray, color: tuple, label: str) -> np.ndarray:
    """画像にバウンディングボックスを描画

    Args:
        image: 入力画像
        boxes: バウンディングボックス (N, 4) [x1, y1, x2, y2]
        color: BGR色
        label: ボックスのラベル

    Returns:
        描画済み画像
    """
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # ラベルを左上に追加
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return img


def create_comparison_image(image_path: Path, gt_boxes: np.ndarray, pred_boxes: np.ndarray, scores: dict) -> np.ndarray:
    """Ground TruthとPredictionの比較画像を作成

    Args:
        image_path: 画像ファイルパス
        gt_boxes: Ground Truth boxes
        pred_boxes: Prediction boxes
        scores: スコア情報

    Returns:
        左右に並べた比較画像
    """
    # 画像読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Ground Truth画像（緑）
    gt_image = draw_boxes_on_image(image, gt_boxes, (0, 255, 0), f"GT (n={scores['n_gt']})")

    # Prediction画像（赤）
    pred_image = draw_boxes_on_image(image, pred_boxes, (0, 0, 255), f"Pred (n={scores['n_pred']})")

    # 左右に結合
    combined = np.hstack([gt_image, pred_image])

    # スコア情報を下部に追加
    h, w = combined.shape[:2]
    info_height = 60
    info_area = np.ones((info_height, w, 3), dtype=np.uint8) * 255

    info_text = (
        f"F1: {scores['f1']:.3f} | "
        f"Precision: {scores['precision']:.3f} | "
        f"Recall: {scores['recall']:.3f} | "
        f"TP: {scores['tp']} FP: {scores['fp']} FN: {scores['fn']}"
    )

    cv2.putText(info_area, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return np.vstack([combined, info_area])


def main():
    parser = argparse.ArgumentParser(description="YOLOモデルのテストスクリプト")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/character_detection/yolo12x_14split_new/weights/best.pt",
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="datasets/yolo_dataset_character_position_detection/test",
        help="テストデータセットのディレクトリ",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/character_detection", help="出力ディレクトリ")
    parser.add_argument("--conf", type=float, default=0.25, help="信頼度閾値")
    parser.add_argument("--iou", type=float, default=0.7, help="NMSのIoU閾値")
    parser.add_argument("--max-det", type=int, default=1000, help="最大検出数")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU閾値")
    parser.add_argument("--n-worst", type=int, default=10, help="表示する精度が低い件数")
    parser.add_argument("--device", type=str, default="0", help="使用するデバイス")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ（高速化のため）")
    parser.add_argument("--workers", type=int, default=8, help="並列読み込みのワーカー数")
    args = parser.parse_args()

    # パスの設定
    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    print(f"モデル: {model_path}")
    print(f"テストディレクトリ: {test_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    # モデルのロード
    print("\nモデルをロード中...")
    model = YOLO(str(model_path))

    # テスト画像の取得
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"テスト画像数: {len(image_files)}")

    # 各画像の評価結果を格納
    results = []

    # 画像とラベルを読み込む関数（並列処理用）
    def load_image_and_label(image_path: Path) -> dict | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        img_height, img_width = image.shape[:2]

        # Ground Truthの読み込み
        label_path = labels_dir / (image_path.stem + ".txt")
        gt_labels = load_yolo_labels(label_path)

        # GT boxesをxyxy形式に変換
        if len(gt_labels) > 0:
            gt_boxes = yolo_to_xyxy(gt_labels[:, 1:], img_width, img_height)
        else:
            gt_boxes = np.array([]).reshape(0, 4)

        return {
            "image_path": image_path,
            "image": image,
            "img_width": img_width,
            "img_height": img_height,
            "gt_boxes": gt_boxes,
        }

    # 画像情報を並列で事前読み込み
    print("\n画像を並列読み込み中...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        loaded_results = list(executor.map(load_image_and_label, image_files))

    # Noneを除外
    image_info_list = [r for r in loaded_results if r is not None]

    print(f"有効な画像数: {len(image_info_list)}")

    # バッチ推論を実行
    print(f"\nバッチ推論を実行中... (batch_size={args.batch_size})")
    batch_size = args.batch_size
    n_batches = (len(image_info_list) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_info_list))
        batch_info = image_info_list[start_idx:end_idx]

        # バッチの画像データを取得（メモリから）
        batch_images = [info["image"] for info in batch_info]

        # バッチ推論（画像配列を直接渡す）
        batch_results = model.predict(
            batch_images,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            batch=batch_size,
        )

        # 各画像の結果を処理
        for info, pred_result in zip(batch_info, batch_results, strict=True):
            # 予測結果の取得
            if len(pred_result.boxes) > 0:
                pred_boxes = pred_result.boxes.xyxy.cpu().numpy()
                pred_scores = pred_result.boxes.conf.cpu().numpy()
            else:
                pred_boxes = np.array([]).reshape(0, 4)
                pred_scores = np.array([])

            # スコアの計算
            scores = compute_image_score(info["gt_boxes"], pred_boxes, pred_scores, args.iou_threshold)
            scores["image_path"] = info["image_path"]
            scores["gt_boxes"] = info["gt_boxes"]
            scores["pred_boxes"] = pred_boxes

            results.append(scores)

        # 進捗表示
        print(f"  バッチ {batch_idx + 1}/{n_batches} 完了 ({end_idx}/{len(image_info_list)} 画像)")

    # 全体のメトリクス計算
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_gt = sum(r["n_gt"] for r in results)
    total_pred = sum(r["n_pred"] for r in results)

    overall_precision = total_tp / total_pred if total_pred > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    print("\n=== 全体結果 ===")
    print(f"画像数: {len(results)}")
    print(f"GT総数: {total_gt}")
    print(f"Prediction総数: {total_pred}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1: {overall_f1:.4f}")

    # F1スコアでソート（低い順）
    results_sorted = sorted(results, key=lambda x: x["f1"])

    # 精度が低い上位N件を保存
    print(f"\n=== 精度が低い{args.n_worst}件を保存中 ===")
    for i, result in enumerate(results_sorted[: args.n_worst]):
        image_path = result["image_path"]
        gt_boxes = result["gt_boxes"]
        pred_boxes = result["pred_boxes"]

        print(f"  {i + 1}. {image_path.name} - F1: {result['f1']:.3f} (GT: {result['n_gt']}, Pred: {result['n_pred']})")

        # 比較画像を作成
        comparison_img = create_comparison_image(image_path, gt_boxes, pred_boxes, result)

        # 保存
        output_path = output_dir / f"worst_{i + 1:02d}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), comparison_img)

    print(f"\n結果を {output_dir} に保存しました")


if __name__ == "__main__":
    main()
