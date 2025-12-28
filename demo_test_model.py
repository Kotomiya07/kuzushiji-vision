#!/usr/bin/env python3
"""
学習済みモデルのテスト用デモスクリプト
いくつかの例文を用意して、テストを実行します。
"""

import os
import sys

from test_trained_model import KuzushijiModelTester


def demo_test_sentences():
    """テスト用の例文を返す"""
    return [
        "今日は[MASK]い天気です",
        "この[MASK]は美しい",
        "[MASK]が好きです",
        "昔[MASK]あるところに",
        "春の[MASK]が咲いている",
        "彼は[MASK]を読んでいる",
        "山の[MASK]に住んでいる",
        "美しい[MASK]を見た",
        "古い[MASK]を発見した",
        "この[MASK]は素晴らしい",
        "雨の[MASK]を歩く",
        "風が[MASK]を吹く",
        "花の[MASK]が香る",
        "鳥が[MASK]を歌う",
        "水が[MASK]を流れる",
        # 複数マスクのテスト
        "[MASK]は[MASK]です",
        "この[MASK]は[MASK]している",
        "[MASK]が[MASK]を[MASK]",
        # より複雑な文
        "源氏物語の[MASK]は美しい",
        "平安時代の[MASK]について",
        "和歌の[MASK]を学ぶ",
        "古典文学の[MASK]を読む",
        "くずし字の[MASK]は難しい",
    ]


def run_demo(model_path: str, tokenizer_path: str = None, top_k: int = 5):
    """
    デモを実行する

    Args:
        model_path: モデルのパス
        tokenizer_path: トークナイザーのパス
        top_k: 予測結果の上位何位まで表示するか
    """
    if tokenizer_path is None:
        tokenizer_path = model_path

    # モデルテスターを初期化
    try:
        tester = KuzushijiModelTester(model_path, tokenizer_path)
    except Exception as e:
        print(f"モデルまたはトークナイザーの読み込みに失敗しました: {e}")
        return

    # テスト用の例文を取得
    test_sentences = demo_test_sentences()

    print("\n" + "=" * 80)
    print("くずし字言語モデル デモテスト")
    print("=" * 80)
    print(f"モデルパス: {model_path}")
    print(f"トークナイザーパス: {tokenizer_path}")
    print(f"テスト文数: {len(test_sentences)}")
    print(f"表示する予測数: {top_k}")
    print("=" * 80)

    # 各例文をテスト
    for i, sentence in enumerate(test_sentences):
        print(f"\n{'=' * 20} テスト {i + 1:2d}/{len(test_sentences)} {'=' * 20}")

        try:
            # 予測実行
            results = tester.predict_masked_tokens(sentence, top_k=top_k)

            # 結果を表示
            tester.print_prediction_results(results, top_display=min(3, top_k))

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

        # 途中でユーザーが中断できるように
        if i < len(test_sentences) - 1:
            try:
                input("\nEnterキーを押して次のテストに進むか、Ctrl+Cで終了...")
            except KeyboardInterrupt:
                print("\n\nデモを中断しました。")
                break

    print("\n" + "=" * 80)
    print("デモテスト完了")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="学習済みモデルのデモテスト")
    parser.add_argument("--model_path", type=str, required=True, help="学習済みモデルのパス")
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="トークナイザーのパス（指定しない場合はmodel_pathと同じ）"
    )
    parser.add_argument("--top_k", type=int, default=5, help="予測結果の上位何位まで表示するか")

    args = parser.parse_args()

    # モデルパスが存在するかチェック
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルパス '{args.model_path}' が存在しません。")
        sys.exit(1)

    # トークナイザーパスのチェック
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    if not os.path.exists(tokenizer_path):
        print(f"エラー: トークナイザーパス '{tokenizer_path}' が存在しません。")
        sys.exit(1)

    # デモを実行
    run_demo(args.model_path, tokenizer_path, args.top_k)


if __name__ == "__main__":
    main()


"""
使用例:

# 基本的な使用方法
python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model

# トークナイザーのパスを別途指定
python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --tokenizer_path experiments/kuzushiji_tokenizer_one_char

# 上位3位まで表示
python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --top_k 3
"""
