import argparse
import csv
from pathlib import Path

import cv2
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.data_loader import (
    build_char_to_int_map,
    get_data_loader,
)
from scripts.ocr_model import OCRModel


class OCRTester:
    def __init__(self, model_path, data_dir, output_dir="test_results", device="auto"):
        """
        OCRモデルのテスト・可視化クラス

        Args:
            model_path (str): 学習済みモデルのチェックポイントパス
            data_dir (str): テストデータディレクトリ
            output_dir (str): 結果出力ディレクトリ
            device (str): 使用デバイス ("auto", "cpu", "gpu")
        """
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # デバイス設定
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"使用デバイス: {self.device}")

        # 文字マッピング構築
        self.char_to_int, self.int_to_char = build_char_to_int_map(str(self.data_dir))
        self.num_chars = len(self.char_to_int)
        print(f"語彙サイズ: {self.num_chars} 文字")

        # モデル読み込み
        self.model = self._load_model()
        self.model.eval()

        # 結果保存用
        self.results = []

    def _load_model(self):
        """学習済みモデルを読み込む"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")

        print(f"モデルを読み込み中: {self.model_path}")
        model = OCRModel.load_from_checkpoint(
            self.model_path, num_chars=self.num_chars, int_to_char=self.int_to_char, map_location=self.device
        )
        model.to(self.device)
        return model

    def _create_test_loader(self, batch_size=1, num_workers=0):
        """テストデータローダーを作成"""
        return get_data_loader(
            data_dir=str(self.data_dir),
            split="test",
            batch_size=batch_size,
            num_workers=num_workers,
            char_to_int=self.char_to_int,
            max_label_length=self.model.hparams.max_label_length,
            target_height=self.model.hparams.input_height,
            target_width=self.model.hparams.input_width,
            shuffle=False,
        )

    def _decode_predictions(self, char_logits):
        """CTC予測をデコード"""
        # Greedy decoding
        predictions = torch.argmax(char_logits, dim=2)  # (batch_size, seq_len)
        decoded_texts = []

        for pred_seq in predictions:
            # CTC blank removal and consecutive duplicate removal
            decoded_chars = []
            prev_char = None

            for char_idx in pred_seq:
                char_idx = char_idx.item()
                if char_idx != 0 and char_idx != prev_char:  # 0 is blank token
                    if char_idx in self.int_to_char:
                        decoded_chars.append(self.int_to_char[char_idx])
                prev_char = char_idx

            decoded_texts.append("".join(decoded_chars))

        return decoded_texts

    def _calculate_cer(self, pred_text, gt_text):
        """Character Error Rate計算"""
        import editdistance

        if len(gt_text) == 0:
            return 1.0 if len(pred_text) > 0 else 0.0
        return editdistance.eval(pred_text, gt_text) / len(gt_text)

    def _calculate_iou(self, box1, box2):
        """IoU計算"""
        # box format: [x_min, y_min, x_max, y_max]
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 交差領域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 各ボックスの面積
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # IoU
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _visualize_text_results(self, image, pred_text, gt_text, cer, sample_idx):
        """文字列結果の可視化"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 画像表示
        if isinstance(image, torch.Tensor):
            # テンソルをnumpy配列に変換
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.array(image)

        ax.imshow(image_np)
        ax.set_title(f"サンプル {sample_idx}: 文字認識結果", fontsize=14, fontweight="bold")
        ax.axis("off")

        # テキスト情報を画像下部に表示
        text_info = f"予測: {pred_text}\n正解: {gt_text}\nCER: {cer:.3f}"
        ax.text(
            0.02,
            0.02,
            text_info,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=12,
            verticalalignment="bottom",
        )

        plt.tight_layout()

        # 保存
        output_path = self.output_dir / f"text_result_{sample_idx:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def _visualize_bbox_results(self, image, pred_boxes, gt_boxes, label_length, sample_idx):
        """バウンディングボックス結果の可視化"""
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.array(image)

        # 2つの画像を作成（予測用と正解用）
        pred_image = image_np.copy()
        gt_image = image_np.copy()

        # 有効な文字数分のボックスのみ処理
        valid_length = min(label_length, len(pred_boxes), len(gt_boxes))

        ious = []

        for i in range(valid_length):
            pred_box = pred_boxes[i]
            gt_box = gt_boxes[i]

            # IoU計算
            iou = self._calculate_iou(pred_box, gt_box)
            ious.append(iou)

            # 予測ボックス描画（赤）
            cv2.rectangle(
                pred_image, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (255, 0, 0), 2
            )

            # 正解ボックス描画（緑）
            cv2.rectangle(gt_image, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 255, 0), 2)

        # 横並び表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(pred_image)
        ax1.set_title("予測バウンディングボックス", fontsize=14, fontweight="bold", color="red")
        ax1.axis("off")

        ax2.imshow(gt_image)
        ax2.set_title("正解バウンディングボックス", fontsize=14, fontweight="bold", color="green")
        ax2.axis("off")

        # IoU情報表示
        mean_iou = np.mean(ious) if ious else 0.0
        fig.suptitle(
            f"サンプル {sample_idx}: バウンディングボックス比較 (平均IoU: {mean_iou:.3f})", fontsize=16, fontweight="bold"
        )

        plt.tight_layout()

        # 保存
        output_path = self.output_dir / f"bbox_result_{sample_idx:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path, mean_iou

    def test_and_visualize(self, max_samples=None, batch_size=1):
        """テスト実行と可視化"""
        print("テスト開始...")

        test_loader = self._create_test_loader(batch_size=batch_size)

        total_cer = 0.0
        total_iou = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if max_samples and batch_idx >= max_samples:
                    break

                images = batch["image"].to(self.device)
                labels = batch["label"]
                bboxes_gt = batch["bounding_boxes"]
                label_lengths = batch["label_lengths"]

                # モデル推論
                outputs = self.model(images)
                char_logits = outputs["char_logits"]
                bbox_preds = outputs["bbox_preds"]

                # バッチ内の各サンプルを処理
                for i in range(images.size(0)):
                    sample_idx = batch_idx * batch_size + i

                    # 予測テキストデコード
                    pred_texts = self._decode_predictions(char_logits[i : i + 1])
                    pred_text = pred_texts[0]

                    # 正解テキスト取得
                    label_length = label_lengths[i].item()
                    gt_label = labels[i][:label_length]
                    gt_text = "".join([self.int_to_char.get(idx.item(), "") for idx in gt_label if idx.item() != 0])

                    # CER計算
                    cer = self._calculate_cer(pred_text, gt_text)
                    total_cer += cer

                    # バウンディングボックス取得
                    pred_boxes = bbox_preds[i][:label_length].cpu().numpy()
                    gt_boxes = bboxes_gt[i][:label_length].cpu().numpy()

                    # 可視化
                    text_path = self._visualize_text_results(images[i].cpu(), pred_text, gt_text, cer, sample_idx)

                    bbox_path, mean_iou = self._visualize_bbox_results(
                        images[i].cpu(), pred_boxes, gt_boxes, label_length, sample_idx
                    )

                    total_iou += mean_iou
                    sample_count += 1

                    # 結果保存
                    self.results.append(
                        {
                            "sample_idx": sample_idx,
                            "pred_text": pred_text,
                            "gt_text": gt_text,
                            "cer": cer,
                            "mean_iou": mean_iou,
                            "text_viz_path": str(text_path),
                            "bbox_viz_path": str(bbox_path),
                        }
                    )

                    print(f"サンプル {sample_idx}: CER={cer:.3f}, IoU={mean_iou:.3f}")

        # 全体統計
        avg_cer = total_cer / sample_count if sample_count > 0 else 0.0
        avg_iou = total_iou / sample_count if sample_count > 0 else 0.0

        print("\n=== テスト結果 ===")
        print(f"処理サンプル数: {sample_count}")
        print(f"平均CER: {avg_cer:.3f}")
        print(f"平均IoU: {avg_iou:.3f}")

        # 結果をCSVに保存
        self._save_results_csv()
        self._save_summary_report(avg_cer, avg_iou, sample_count)

        return avg_cer, avg_iou

    def _save_results_csv(self):
        """結果をCSVファイルに保存"""
        csv_path = self.output_dir / "test_results.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sample_idx", "pred_text", "gt_text", "cer", "mean_iou", "text_viz_path", "bbox_viz_path"]
            )
            writer.writeheader()
            writer.writerows(self.results)

        print(f"詳細結果をCSVに保存: {csv_path}")

    def _save_summary_report(self, avg_cer, avg_iou, sample_count):
        """サマリーレポートを保存"""
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("OCRモデル テスト結果サマリー\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"モデルパス: {self.model_path}\n")
            f.write(f"データディレクトリ: {self.data_dir}\n")
            f.write(f"出力ディレクトリ: {self.output_dir}\n\n")
            f.write(f"処理サンプル数: {sample_count}\n")
            f.write(f"平均CER: {avg_cer:.3f}\n")
            f.write(f"平均IoU: {avg_iou:.3f}\n\n")
            f.write("詳細結果は test_results.csv を参照してください。\n")

        print(f"サマリーレポートを保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="OCRモデルのテスト・可視化")

    parser.add_argument("--model_path", type=str, required=True, help="学習済みモデルのチェックポイントパス")
    parser.add_argument("--data_dir", type=str, required=True, help="テストデータディレクトリ")
    parser.add_argument("--output_dir", type=str, default="test_results", help="結果出力ディレクトリ")
    parser.add_argument("--max_samples", type=int, default=None, help="処理する最大サンプル数（None=全て）")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="使用デバイス")

    args = parser.parse_args()

    # テスター初期化
    tester = OCRTester(model_path=args.model_path, data_dir=args.data_dir, output_dir=args.output_dir, device=args.device)

    # テスト実行
    avg_cer, avg_iou = tester.test_and_visualize(max_samples=args.max_samples, batch_size=args.batch_size)

    print(f"\nテスト完了！結果は {args.output_dir} に保存されました。")


if __name__ == "__main__":
    main()

