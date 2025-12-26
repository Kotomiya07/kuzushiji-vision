#!/usr/bin/env python3
"""
学習済みモデルをテストするスクリプト
文を入力して、[MASK]トークンを使って対話的にテストできます。
"""

import argparse
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class KuzushijiModelTester:
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        学習済みモデルとトークナイザーを読み込み

        Args:
            model_path: 学習済みモデルのパス
            tokenizer_path: トークナイザーのパス
        """
        print(f"モデルを読み込み中: {model_path}")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

        print(f"トークナイザーを読み込み中: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"デバイス: {self.device}")
        print(f"語彙サイズ: {len(self.tokenizer)}")
        print(f"マスクトークン: {self.tokenizer.mask_token}")

    def predict_masked_tokens(self, text: str, top_k: int = 5) -> dict[str, Any]:
        """
        マスクされたトークンを予測

        Args:
            text: [MASK]を含む入力文
            top_k: 上位何位まで予測結果を取得するか

        Returns:
            予測結果の辞書
        """
        # [MASK]をトークナイザーのマスクトークンに変換
        processed_text = text.replace("[MASK]", self.tokenizer.mask_token)

        # トークナイズ
        inputs = self.tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # マスクトークンの位置を取得
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = torch.where(inputs["input_ids"] == mask_token_id)[1]

        if len(mask_positions) == 0:
            return {
                "error": "マスクトークンが見つかりませんでした。[MASK]を含む文を入力してください。",
                "original_text": text,
                "processed_text": processed_text,
            }

        # 予測実行
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        results = []

        # 各マスク位置について予測
        for _i, mask_pos in enumerate(mask_positions):
            mask_logits = predictions[0, mask_pos, :]

            # Top-k予測を取得
            top_k_values, top_k_indices = torch.topk(mask_logits, top_k)

            # 確率に変換
            probabilities = torch.softmax(mask_logits, dim=-1)
            top_k_probs = probabilities[top_k_indices]

            # トークンをテキストに変換
            top_k_tokens = []
            for idx, prob in zip(top_k_indices, top_k_probs, strict=True):
                token = self.tokenizer.decode([idx], skip_special_tokens=True)
                top_k_tokens.append({"token": token, "probability": prob.item(), "token_id": idx.item()})

            results.append({"mask_position": mask_pos.item(), "top_predictions": top_k_tokens})

        # 各予測で文を復元
        restored_texts = []
        for i in range(top_k):
            restored_tokens = inputs["input_ids"].clone()
            for j, mask_pos in enumerate(mask_positions):
                if j < len(results) and i < len(results[j]["top_predictions"]):
                    predicted_token_id = results[j]["top_predictions"][i]["token_id"]
                    restored_tokens[0, mask_pos] = predicted_token_id

            restored_text = self.tokenizer.decode(restored_tokens[0], skip_special_tokens=True)
            restored_texts.append(restored_text)

        return {
            "original_text": text,
            "processed_text": processed_text,
            "mask_count": len(mask_positions),
            "predictions": results,
            "restored_texts": restored_texts,
        }

    def print_prediction_results(self, results: dict[str, Any], top_display: int = 5):
        """
        予測結果を見やすく表示

        Args:
            results: predict_masked_tokens の結果
            top_display: 表示する上位予測数
        """
        if "error" in results:
            print(f"エラー: {results['error']}")
            return

        print("\n" + "=" * 80)
        print("マスクトークン予測結果")
        print("=" * 80)

        print(f"入力文:     {results['original_text']}")
        print(f"処理後文:   {results['processed_text']}")
        print(f"マスク数:   {results['mask_count']}")

        print("\n" + "-" * 60)
        print("各マスク位置の予測:")
        print("-" * 60)

        for i, mask_result in enumerate(results["predictions"]):
            print(f"\nマスク位置 {i + 1} (トークン位置: {mask_result['mask_position']}):")
            for j, pred in enumerate(mask_result["top_predictions"][:top_display]):
                print(f"  {j + 1}. '{pred['token']}' (確率: {pred['probability']:.4f})")

        print("\n" + "-" * 60)
        print("復元された文:")
        print("-" * 60)

        for i, restored_text in enumerate(results["restored_texts"][:top_display]):
            print(f"Top{i + 1}: {restored_text}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="学習済みモデルをテストするスクリプト")
    parser.add_argument("--model_path", type=str, required=True, help="学習済みモデルのパス")
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="トークナイザーのパス（指定しない場合はmodel_pathと同じ）"
    )
    parser.add_argument("--top_k", type=int, default=5, help="予測結果の上位何位まで表示するか")
    parser.add_argument("--batch_mode", action="store_true", help="バッチモード（複数の文を一度に処理）")

    args = parser.parse_args()

    # トークナイザーのパスが指定されていない場合はモデルパスと同じにする
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path

    # モデルテスターを初期化
    tester = KuzushijiModelTester(args.model_path, tokenizer_path)

    print("\n" + "=" * 80)
    print("くずし字言語モデル テスター")
    print("=" * 80)
    print("使用方法:")
    print("  - [MASK]を含む文を入力してください")
    print("  - 例: 今日は[MASK]い天気です")
    print("  - 終了するには 'quit' または 'exit' を入力してください")
    print("=" * 80)

    if args.batch_mode:
        # バッチモード
        print("\nバッチモード: 複数の文を入力してください（空行で終了）")
        texts = []
        while True:
            try:
                line = input("文を入力 (空行で終了): ").strip()
                if not line:
                    break
                texts.append(line)
            except KeyboardInterrupt:
                print("\n処理を中断しました。")
                return

        for i, text in enumerate(texts):
            print(f"\n{'=' * 20} 文 {i + 1} {'=' * 20}")
            results = tester.predict_masked_tokens(text, top_k=args.top_k)
            tester.print_prediction_results(results, top_display=args.top_k)

    else:
        # 対話モード
        print("\n対話モード: 文を一つずつ入力してください")
        while True:
            try:
                text = input("\n文を入力: ").strip()

                if text.lower() in ["quit", "exit", "q"]:
                    print("テストを終了します。")
                    break

                if not text:
                    print("文を入力してください。")
                    continue

                if "[MASK]" not in text:
                    print("警告: [MASK]トークンが含まれていません。そのまま処理しますか？ (y/n)")
                    confirm = input().strip().lower()
                    if confirm not in ["y", "yes"]:
                        continue

                # 予測実行
                results = tester.predict_masked_tokens(text, top_k=args.top_k)
                tester.print_prediction_results(results, top_display=args.top_k)

            except KeyboardInterrupt:
                print("\n\nテストを終了します。")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()


"""
使用例:

# 基本的な使用方法
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model

# トークナイザーのパスを別途指定
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --tokenizer_path experiments/kuzushiji_tokenizer_one_char

# 上位10位まで表示
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --top_k 10

# バッチモードで複数の文を一度に処理
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --batch_mode

# 対話例:
# 文を入力: 今日は[MASK]い天気です
# 文を入力: この[MASK]は美しい
# 文を入力: [MASK]が好きです
# 文を入力: 昔[MASK]あるところに
"""
