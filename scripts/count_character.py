#!/usr/bin/env python3
"""
文字コード出現回数カウントスクリプト

data/processed_v2/column_info.csvのunicode_ids列から文字コードの出現回数を数え、
5回以上出現する文字コードをJSON形式で出力します。

出力形式:
{
    "summary": {
        "total_characters": int,
        "characters_above_threshold": int,
        "threshold": int
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


def count_character_frequencies(csv_path: str, threshold: int = 5) -> dict:
    """
    CSVファイルから文字コードの出現回数を数える

    Args:
        csv_path: CSVファイルのパス
        threshold: 出現回数の閾値（この値以上のものを出力）

    Returns:
        結果を格納した辞書
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

    # 閾値以上の文字コードを抽出
    frequent_chars = [(unicode_id, count) for unicode_id, count in char_counter.items() if count >= threshold]
    frequent_chars.sort(key=lambda x: x[1], reverse=True)  # 出現回数の降順でソート

    # 結果を辞書形式で整理
    result = {
        "summary": {
            "total_characters": len(char_counter),
            "characters_above_threshold": len(frequent_chars),
            "threshold": threshold,
        },
        "characters": [
            {"index": idx + 1, "char": convert_unicode_to_char(unicode_id), "unicode_id": unicode_id, "count": count}
            for idx, (unicode_id, count) in enumerate(frequent_chars)
        ],
    }

    return result


def main():
    """メイン関数"""
    # パス設定
    csv_path = "data/processed_v2/column_info.csv"
    output_path = "character_count_results.json"
    threshold = 5

    # CSVファイルの存在確認
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print("=" * 50)
    print("文字コード出現回数カウントスクリプト")
    print("=" * 50)
    print(f"入力ファイル: {csv_path}")
    print(f"出力ファイル: {output_path}")
    print(f"閾値: {threshold}回以上")
    print("=" * 50)

    try:
        # 文字コード出現回数をカウント
        result = count_character_frequencies(csv_path, threshold)

        # 結果をJSONファイルに出力
        print(f"\nSaving results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # サマリーを表示
        print("\n" + "=" * 50)
        print("実行結果サマリー")
        print("=" * 50)
        print(f"総文字種数: {result['summary']['total_characters']}")
        print(f"閾値{threshold}回以上の文字種数: {result['summary']['characters_above_threshold']}")

        # 上位10文字を表示
        print("\n上位10文字:")
        for i, char_info in enumerate(result["characters"][:10]):
            print(f"  {i + 1:2d}. {char_info['char']}: {char_info['count']:>6d}回")

        print(f"\n詳細結果は {output_path} に保存されました。")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
