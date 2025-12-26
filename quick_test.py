#!/usr/bin/env python3
"""
簡単な使用例とクイックテスト
"""

import argparse
import os


def run_quick_test():
    """簡単なテストを実行"""
    # 実際のモデルパスを探す
    base_path = "experiments/pretrain_language_model"

    if not os.path.exists(base_path):
        print(f"エラー: {base_path} が存在しません。")
        print("先にモデルを学習してください。")
        return

    # 最新のモデルを探す
    model_dirs = []
    for root, dirs, _files in os.walk(base_path):
        if "final_model" in dirs:
            model_dirs.append(os.path.join(root, "final_model"))

    if not model_dirs:
        print("学習済みモデルが見つかりません。")
        print("先にtrain_language_model.pyでモデルを学習してください。")
        return

    # 最新のモデルを使用
    model_path = sorted(model_dirs)[-1]
    print(f"使用するモデル: {model_path}")

    # test_trained_model.pyを実行
    os.system(
        f"python test_trained_model.py --model_path {model_path} --tokenizer_path experiments/kuzushiji_tokenizer_one_char"
    )


def show_usage():
    """使用方法を表示"""
    print("=" * 80)
    print("くずし字言語モデル テスト用スクリプト")
    print("=" * 80)
    print()
    print("1. 対話式テスト:")
    print("   python test_trained_model.py --model_path [モデルパス]")
    print()
    print("2. デモテスト:")
    print("   python demo_test_model.py --model_path [モデルパス]")
    print()
    print("3. クイックテスト:")
    print("   python quick_test.py")
    print()
    print("使用例:")
    print(
        "  python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model"
    )
    print()
    print(
        "  python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model"
    )
    print()
    print("マスクトークンの使用方法:")
    print("  - [MASK]を含む文を入力")
    print("  - 例: 今日は[MASK]い天気です")
    print("  - 例: この[MASK]は美しい")
    print("  - 例: [MASK]が好きです")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="クイックテスト")
    parser.add_argument("--usage", action="store_true", help="使用方法を表示")

    args = parser.parse_args()

    if args.usage:
        show_usage()
    else:
        run_quick_test()


if __name__ == "__main__":
    main()
