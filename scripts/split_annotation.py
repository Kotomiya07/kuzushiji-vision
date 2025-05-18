"""
アノテーションCSVファイルを読み込み、書籍IDごとに分割して保存するスクリプト
このスクリプトは、指定されたCSVファイルを読み込み、各行の書籍IDに基づいて新しいCSVファイルを作成します。
各書籍IDごとに1つのCSVファイルが作成され、元のCSVファイルのヘッダー行も含まれます。
書籍IDは、column_image列のパスから抽出されます。
"""

import csv
import sys
from pathlib import Path

# --- 設定 ---
# 入力CSVファイルのパス (スクリプトからの相対パスまたは絶対パス)
# スクリプトと同じディレクトリにある場合: INPUT_CSV_PATH = Path("column_info.csv")
# 指定されたパスの場合:
INPUT_CSV_PATH = Path("data/processed/column_info.csv")

# 出力先ディレクトリ (存在しない場合は作成されます)
OUTPUT_DIR = Path("data/processed/book_annotations")

# column_image 列のインデックス (0から始まる)
COLUMN_IMAGE_INDEX = 0
# パスから書籍IDを抽出する際の基準となるプレフィックス
# "processed/column_images/" の後に続く部分をIDとする
PATH_PREFIX_PARTS = ["processed", "column_images"]
# --- 設定ここまで ---


def split_annotations_by_book(input_csv_path: Path, output_dir: Path):
    """
    アノテーションCSVファイルを書籍IDごとに分割する関数

    Args:
        input_csv_path (Path): 入力CSVファイルのパスオブジェクト
        output_dir (Path): 出力先ディレクトリのパスオブジェクト
    """
    if not input_csv_path.is_file():
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリを作成 (存在してもエラーにならない)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力先ディレクトリ: {output_dir.resolve()}")

    output_files = {}  # {book_id: {'file': file_handle, 'writer': csv_writer}}
    header = None
    processed_rows = 0
    skipped_rows = 0
    book_ids_found = set()

    try:
        print(f"入力ファイルを開いています: {input_csv_path}")
        with open(input_csv_path, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)

            # ヘッダーを読み込む
            try:
                header = next(reader)
                print(f"ヘッダーを読み込みました: {header}")
                if len(header) <= COLUMN_IMAGE_INDEX:
                    print(
                        f"エラー: ヘッダーに十分な列がありません。column_image列 (インデックス {COLUMN_IMAGE_INDEX}) が存在しません。",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except StopIteration:
                print("エラー: 入力ファイルが空です。", file=sys.stderr)
                sys.exit(1)

            # 各行を処理
            for i, row in enumerate(reader):
                if not row:  # 空行はスキップ
                    skipped_rows += 1
                    continue

                try:
                    # column_image のパスを取得
                    image_path_str = row[COLUMN_IMAGE_INDEX]
                    image_path = Path(image_path_str)
                    parts = image_path.parts

                    # 書籍IDを抽出
                    book_id = None
                    # partsが ('processed', 'column_images', 'BOOK_ID', ...) の形式かチェック
                    if len(parts) > len(PATH_PREFIX_PARTS) and all(
                        parts[j] == PATH_PREFIX_PARTS[j] for j in range(len(PATH_PREFIX_PARTS))
                    ):
                        book_id = parts[len(PATH_PREFIX_PARTS)]
                    else:
                        print(
                            f"警告: 行 {i + 2}: 予期しないパス形式のため書籍IDを抽出できませんでした: {image_path_str}",
                            file=sys.stderr,
                        )
                        skipped_rows += 1
                        continue

                    if book_id not in output_files:
                        # 新しい書籍IDの場合、新しいファイルを開く
                        output_filename = output_dir / f"{book_id}_annotations.csv"
                        print(f"新しい書籍IDが見つかりました: {book_id} -> {output_filename}")
                        outfile = open(output_filename, "w", newline="", encoding="utf-8")
                        writer = csv.writer(outfile)
                        writer.writerow(header)  # ヘッダーを書き込む
                        output_files[book_id] = {"file": outfile, "writer": writer}
                        book_ids_found.add(book_id)

                    # 対応するファイルに行を書き込む
                    output_files[book_id]["writer"].writerow(row)
                    processed_rows += 1

                except IndexError:
                    print(f"警告: 行 {i + 2}: 列数が足りません。スキップします。データ: {row}", file=sys.stderr)
                    skipped_rows += 1
                except Exception as e:
                    print(f"警告: 行 {i + 2} の処理中にエラーが発生しました: {e}。データ: {row}", file=sys.stderr)
                    skipped_rows += 1

    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # すべての出力ファイルを閉じる
        closed_count = 0
        for book_id, data in output_files.items():
            try:
                if data["file"] and not data["file"].closed:
                    data["file"].close()
                    closed_count += 1
            except Exception as e:
                print(f"警告: ファイル '{book_id}_annotations.csv' のクローズ中にエラー: {e}", file=sys.stderr)
        print(f"\n処理が完了しました。{closed_count} 個のファイルを閉じました。")

    print("\n--- 結果 ---")
    print(f"処理された行数: {processed_rows}")
    print(f"スキップされた行数: {skipped_rows}")
    print(f"見つかった書籍IDの数: {len(book_ids_found)}")
    if book_ids_found:
        print(f"書籍IDの例: {list(book_ids_found)[:5]}...")  # 最初の5つを表示
    print(f"出力ファイルはディレクトリ '{output_dir.resolve()}' に保存されました。")


if __name__ == "__main__":
    # スクリプトを実行
    split_annotations_by_book(INPUT_CSV_PATH, OUTPUT_DIR)
