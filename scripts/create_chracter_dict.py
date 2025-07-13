#!/usr/bin/env python3
"""
Unicode文字辞書作成スクリプト

data/processed_v2/column_info.csvのunicode_ids列から全ての文字コードを取得し、
Unicode順に並べてunicodeとidの対応辞書を作成します。

出力形式:
{
    "summary": {
        "total_characters": int,
        "description": str
    },
    "unicode_to_id": {
        "U+xxxx": int,
        ...
    },
    "id_to_char": {
        "0": str,
        "1": str,
        ...
    },
    "characters": [
        {
            "id": int,
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


def create_unicode_dict(csv_path: str) -> dict:
    """
    CSVファイルから全文字をUnicode順に並べた辞書を作成する

    Args:
        csv_path: CSVファイルのパス

    Returns:
        Unicode辞書を格納した辞書
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

    # ユニークな文字コードを取得してUnicode順（昇順）にソート
    print("Sorting characters by Unicode order...")
    unique_unicode_codes = list(char_counter.keys())

    # Unicode値でソート（U+の部分を除いて16進数として比較）
    unique_unicode_codes.sort(key=lambda x: int(x.lstrip("U+"), 16))

    # unicode_to_id 辞書を作成
    unicode_to_id = {unicode_code: idx for idx, unicode_code in enumerate(unique_unicode_codes)}

    # id_to_char 辞書を作成
    id_to_char = {str(idx): convert_unicode_to_char(unicode_code) for idx, unicode_code in enumerate(unique_unicode_codes)}

    # characters リストを作成
    characters = [
        {
            "id": idx,
            "char": convert_unicode_to_char(unicode_code),
            "unicode_id": unicode_code,
            "count": char_counter[unicode_code],
        }
        for idx, unicode_code in enumerate(unique_unicode_codes)
    ]

    # 結果辞書を作成
    result = {
        "summary": {"total_characters": len(unique_unicode_codes), "description": "Unicode順に並べた文字辞書"},
        "unicode_to_id": unicode_to_id,
        "id_to_char": id_to_char,
        "characters": characters,
    }

    return result


def main():
    """メイン関数"""
    # パス設定
    csv_path = "data/processed_v2/column_info.csv"
    output_path = "unicode_character_dict.json"

    # CSVファイルの存在確認
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print("=" * 50)
    print("Unicode文字辞書作成スクリプト")
    print("=" * 50)
    print(f"入力ファイル: {csv_path}")
    print(f"出力ファイル: {output_path}")
    print("=" * 50)

    try:
        # Unicode辞書を作成
        result = create_unicode_dict(csv_path)

        # 結果をJSONファイルに出力
        print("\nSaving results...")
        print(f"Saving Unicode dictionary to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # サマリーを表示
        print("\n" + "=" * 50)
        print("実行結果サマリー")
        print("=" * 50)
        print(f"総文字種数: {result['summary']['total_characters']}")

        # 最初の10文字を表示
        print("\nUnicode順 最初の10文字:")
        for i, char_info in enumerate(result["characters"][:10]):
            print(f"  ID {char_info['id']:3d}: {char_info['char']} ({char_info['unicode_id']}) - {char_info['count']:>4d}回")

        # 最後の10文字を表示
        print("\nUnicode順 最後の10文字:")
        for char_info in result["characters"][-10:]:
            print(f"  ID {char_info['id']:3d}: {char_info['char']} ({char_info['unicode_id']}) - {char_info['count']:>4d}回")

        print("\n詳細結果は以下のファイルに保存されました:")
        print(f"  - {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
