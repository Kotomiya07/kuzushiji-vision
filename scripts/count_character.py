#!/usr/bin/env python3
"""
文字コード出現回数カウントスクリプト

data/processed_v2/column_info.csvのunicode_ids列から文字コードの出現回数を数え、
5回以上と5回未満の文字コードを別々のJSON形式ファイルに出力します。

出力形式:
{
    "summary": {
        "total_characters": int,
        "characters_in_category": int,
        "threshold": int,
        "category": str
    },
    "characters": [
        {
            "index": int,
            "char": str,
            "unicode_id": str,
            "count": int
        },
        ...
    ]
}
"""

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def convert_unicode_to_char(unicode_str: str) -> str:
    """
    unicode_strを文字に変換する

    Args:
        unicode_str: "U+306E" 形式の文字列

    Returns:
        unicode_strを文字に変換した文字列
    """
    return chr(int(unicode_str.lstrip("U+"), 16))


def parse_unicode_list(unicode_str: str) -> list[str]:
    """
    文字列として格納されたリスト形式のunicode_idsをパースする

    Args:
        unicode_str: "['U+56DB', 'U+4FEE', ...]" 形式の文字列

    Returns:
        unicode文字コードのリスト
    """
    try:
        # ast.literal_evalを使って安全にリストをパース
        unicode_list = ast.literal_eval(unicode_str)
        return unicode_list if isinstance(unicode_list, list) else []
    except (ValueError, SyntaxError):
        # パースに失敗した場合は空リストを返す
        print(f"Warning: Failed to parse unicode string: {unicode_str}")
        return []


def count_character_frequencies(csv_path: str, threshold: int = 5) -> tuple[dict, dict]:
    """
    CSVファイルから文字コードの出現回数を数える

    Args:
        csv_path: CSVファイルのパス
        threshold: 出現回数の閾値

    Returns:
        (above_threshold_result, below_threshold_result)のタプル
        それぞれ結果を格納した辞書
    """
    # CSVファイル読み込み
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # 全文字コードを収集
    all_unicode_codes = []
    parse_errors = 0

    print("Parsing unicode_ids column...")
    for idx, unicode_str in enumerate(df["unicode_ids"]):
        if pd.isna(unicode_str):
            continue

        unicode_list = parse_unicode_list(unicode_str)
        if not unicode_list:
            parse_errors += 1
        else:
            all_unicode_codes.extend(unicode_list)

        # 進捗表示
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")

    print(f"Parse errors: {parse_errors}")
    print(f"Total unicode codes collected: {len(all_unicode_codes)}")

    # 文字コードの出現回数をカウント
    print("Counting character frequencies...")
    char_counter = Counter(all_unicode_codes)

    # 閾値以上と閾値未満の文字コードを分ける
    above_threshold_chars = [(unicode_id, count) for unicode_id, count in char_counter.items() if count >= threshold]
    below_threshold_chars = [(unicode_id, count) for unicode_id, count in char_counter.items() if count < threshold]

    # 出現回数の降順でソート
    above_threshold_chars.sort(key=lambda x: x[1], reverse=True)
    below_threshold_chars.sort(key=lambda x: x[1], reverse=True)

    # 閾値以上の結果を辞書形式で整理
    above_threshold_result = {
        "summary": {
            "total_characters": len(char_counter),
            "characters_in_category": len(above_threshold_chars),
            "threshold": threshold,
            "category": "above_threshold",
        },
        "characters": [
            {"index": idx + 1, "char": convert_unicode_to_char(unicode_id), "unicode_id": unicode_id, "count": count}
            for idx, (unicode_id, count) in enumerate(above_threshold_chars)
        ],
    }

    # 閾値未満の結果を辞書形式で整理
    below_threshold_result = {
        "summary": {
            "total_characters": len(char_counter),
            "characters_in_category": len(below_threshold_chars),
            "threshold": threshold,
            "category": "below_threshold",
        },
        "characters": [
            {"index": idx + 1, "char": convert_unicode_to_char(unicode_id), "unicode_id": unicode_id, "count": count}
            for idx, (unicode_id, count) in enumerate(below_threshold_chars)
        ],
    }

    return above_threshold_result, below_threshold_result


def main():
    """メイン関数"""
    # パス設定
    csv_path = "data/processed_v2/column_info.csv"
    above_threshold_output_path = "character_count_above_threshold.json"
    below_threshold_output_path = "character_count_below_threshold.json"
    threshold = 5

    # CSVファイルの存在確認
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print("=" * 50)
    print("文字コード出現回数カウントスクリプト")
    print("=" * 50)
    print(f"入力ファイル: {csv_path}")
    print(f"出力ファイル（閾値以上）: {above_threshold_output_path}")
    print(f"出力ファイル（閾値未満）: {below_threshold_output_path}")
    print(f"閾値: {threshold}回")
    print("=" * 50)

    try:
        # 文字コード出現回数をカウント
        above_result, below_result = count_character_frequencies(csv_path, threshold)

        # 結果をJSONファイルに出力
        print("\nSaving results...")

        # 閾値以上の結果を保存
        print(f"Saving above threshold results to: {above_threshold_output_path}")
        with open(above_threshold_output_path, "w", encoding="utf-8") as f:
            json.dump(above_result, f, ensure_ascii=False, indent=2)

        # 閾値未満の結果を保存
        print(f"Saving below threshold results to: {below_threshold_output_path}")
        with open(below_threshold_output_path, "w", encoding="utf-8") as f:
            json.dump(below_result, f, ensure_ascii=False, indent=2)

        # サマリーを表示
        print("\n" + "=" * 50)
        print("実行結果サマリー")
        print("=" * 50)
        print(f"総文字種数: {above_result['summary']['total_characters']}")
        print(f"閾値{threshold}回以上の文字種数: {above_result['summary']['characters_in_category']}")
        print(f"閾値{threshold}回未満の文字種数: {below_result['summary']['characters_in_category']}")

        # 上位10文字を表示（閾値以上）
        print(f"\n閾値{threshold}回以上の上位10文字:")
        for i, char_info in enumerate(above_result["characters"][:10]):
            print(f"  {i + 1:2d}. {char_info['char']}: {char_info['count']:>6d}回")

        # 閾値未満の上位10文字を表示
        print(f"\n閾値{threshold}回未満の上位10文字:")
        for i, char_info in enumerate(below_result["characters"][:10]):
            print(f"  {i + 1:2d}. {char_info['char']}: {char_info['count']:>6d}回")

        print("\n詳細結果は以下のファイルに保存されました:")
        print(f"  - 閾値以上: {above_threshold_output_path}")
        print(f"  - 閾値未満: {below_threshold_output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
