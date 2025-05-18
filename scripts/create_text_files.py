"""
このスクリプトは、CSVファイルからテキストファイルを作成するためのものです。
CSVファイルは、「kokubunken_repo/csv」ディレクトリにあります。
出力ファイルは、「kokubunken_repo/text」ディレクトリに作成されます。

CSVファイルのカラム名は、「file_column_mapping」変数で指定します。
CSVファイルのカラム名が上記マッピングにない場合は、「default_column_name」変数で指定したカラム名が使用されます。
"""

import csv
import os
import re  # 正規表現モジュールをインポート

from pykakasi import kakasi  # pykakasiをインポート

# 対象ディレクトリと出力ディレクトリ
target_dir = "kokubunken_repo/csv"
output_base_dir = "kokubunken_repo/text"  # ベースとなる出力ディレクトリ

# ディレクトリが存在しない場合は作成 (通常は存在すると仮定)
# if not os.path.exists(output_dir):
# os.makedirs(output_dir)
# ↑ この部分は各CSVファイル処理時に個別ディレクトリ作成に変更

# ファイル名と抽出するカラム名のマッピング
file_column_mapping = {"二十一代集.csv": "詞書原表記"}
# 上記マッピングにないCSVファイルの場合のデフォルトカラム名
default_column_name = "本文領域（該当行）"


def process_text(text: str) -> str:
    """
    指定された記号の処理とスペースの正規化を行う

    全領域に使われる記号
    ＋＋	ヲドリ字（二字以上を繰り返す記号）
    ／＼	ヲドリ字（二字以上を繰り返す記号）
    ／″＼	ヲドリ字（二字以上を繰り返し、濁点の付く文字を含む記号）
    △	原本のスペース
    〈　〉	割書き

    標準領域に使われる記号
    ＄	漢字に濁点が付されている場合、当該漢字の後
    ＝	音読みを意味する連辞符（字間中央にあるもの）
    －	音読みを意味する連辞符（字間左側にあるもの）
    ☆	文字でない記号・系図部分の線など
    ｛／｝	訓点を再現するために用いた記号、｛送り仮名／返り点｝

    """
    # ヲドリ字の処理 (特殊なケースを先に処理)
    text = text.replace("＋＋＄", "ゞ")  # 「＋＋」の後に「＄」が続く場合
    text = text.replace("＋＋", "ゝ")
    text = text.replace("／″＼", "ヽ")  # 濁点付きの繰り返し
    text = text.replace("／＼", "ヾ")

    # 原本のスペースを削除 (ユーザーによる変更)
    text = text.replace("△", "")
    # 割書きの記号を削除
    text = text.replace("〈", "")
    text = text.replace("〉", "")
    # 標準領域の記号の削除
    text = text.replace("＄", "")
    text = text.replace("＝", "")
    text = text.replace("－", "")
    text = text.replace("☆", "")

    # ｛ と ｝ の間にある文字も削除 (正規表現を使用)
    text = re.sub(r"｛.*?｝", "", text)

    # 複数のスペースが連続している場合、一つにまとめる
    # strip() で前後の空白も除去しておく
    text = " ".join(text.split())
    return text


# ディレクトリ内のCSVファイルを取得
try:
    all_files = os.listdir(target_dir)
    csv_files = [f for f in all_files if f.endswith(".csv")]
except FileNotFoundError:
    print(f"エラー: ディレクトリ '{target_dir}' が見つかりません。")
    exit()

if not csv_files:
    print(f"ディレクトリ '{target_dir}' にCSVファイルが見つかりません。")
    exit()

print(f"処理対象のCSVファイル: {csv_files}")

for csv_file_name in csv_files:
    input_csv_path = os.path.join(target_dir, csv_file_name)

    # CSVファイル名（拡張子なし）を取得
    base_name_without_ext = os.path.splitext(csv_file_name)[0]

    # pykakasiのインスタンスを作成
    kks = kakasi()
    # 変換モードを設定（ひらがな -> ローマ字, カタカナ -> ローマ字, 漢字 -> ローマ字）
    kks.setMode("H", "a")  # Hiragana to ascii
    kks.setMode("K", "a")  # Katakana to ascii
    kks.setMode("J", "a")  # Japanese to ascii (kanji)
    conv = kks.getConverter()

    # ファイル名をローマ字に変換
    romanized_name = conv.do(base_name_without_ext)
    # スペースや特殊文字をアンダースコアに置換（オプション、必要に応じて調整）
    output_subdir_name = re.sub(r"\W+", "_", romanized_name).lower()

    # 出力サブディレクトリのフルパス
    output_dir_for_book = os.path.join(output_base_dir, output_subdir_name)

    # 出力サブディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir_for_book):
        os.makedirs(output_dir_for_book)
        print(f"ディレクトリ作成: {output_dir_for_book}")

    column_to_extract = file_column_mapping.get(csv_file_name, default_column_name)
    extracted_texts = []
    file_counter = 1  # 各本（CSVファイル）ごとにカウンターをリセット

    print(f"処理中: {csv_file_name} (カラム: {column_to_extract}) -> 出力先: {output_dir_for_book}")

    try:
        # まずUTF-8で試行
        try:
            with open(input_csv_path, encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                if column_to_extract not in reader.fieldnames:
                    print(
                        f"警告 (UTF-8): ファイル '{csv_file_name}' にカラム '{column_to_extract}' が見つかりません。スキップします。"
                    )
                    continue  # continue from the outer loop

                temp_extracted_texts = []
                for _i, row in enumerate(reader):
                    text = row.get(column_to_extract)
                    if text and text.strip():
                        processed_text = process_text(text.strip())
                        if processed_text:  # 処理後も空文字列でない場合
                            temp_extracted_texts.append(processed_text)
                extracted_texts = temp_extracted_texts
        except UnicodeDecodeError:
            print(f"情報: ファイル '{csv_file_name}' はUTF-8デコードに失敗しました。CP932で再試行します。")
            try:
                with open(input_csv_path, encoding="cp932") as infile:
                    reader = csv.DictReader(infile)
                    if column_to_extract not in reader.fieldnames:
                        print(
                            f"警告 (CP932): ファイル '{csv_file_name}' にカラム '{column_to_extract}' が見つかりません。スキップします。"
                        )
                        continue  # continue from the outer loop

                    temp_extracted_texts = []
                    for _i, row in enumerate(reader):
                        text = row.get(column_to_extract)
                        if text and text.strip():
                            processed_text = process_text(text.strip())
                            if processed_text:  # 処理後も空文字列でない場合
                                temp_extracted_texts.append(processed_text)
                    extracted_texts = temp_extracted_texts
            except UnicodeDecodeError:
                print(f"エラー: ファイル '{csv_file_name}' はUTF-8およびCP932でのデコードに失敗しました。スキップします。")
                continue  # continue from the outer loop
            except Exception as e_cp932:
                print(f"ファイル '{csv_file_name}' のCP932処理中に予期せぬエラーが発生しました: {e_cp932}")
                continue  # continue from the outer loop
        except Exception as e_utf8:
            print(f"ファイル '{csv_file_name}' のUTF-8処理中に予期せぬエラーが発生しました: {e_utf8}")
            continue  # continue from the outer loop

        if extracted_texts:
            for _i, text_content in enumerate(extracted_texts):
                # 出力ファイル名 (例: 00001.txt)
                output_txt_file_name = f"{file_counter:05d}.txt"
                output_txt_path = os.path.join(output_dir_for_book, output_txt_file_name)
                try:
                    with open(output_txt_path, "w", encoding="utf-8") as outfile:
                        outfile.write(text_content + "\n")
                    # print(f"作成完了: {output_txt_path}") # 個別ファイル作成のログは冗長なためコメントアウトも検討
                    file_counter += 1
                except Exception as e_write:
                    print(f"エラー: ファイル '{output_txt_path}' の書き込み中にエラーが発生しました: {e_write}")
            print(
                f"完了: {csv_file_name} から {len(extracted_texts)}行のテキストを {output_dir_for_book} に個別ファイルとして保存しました。"
            )
        else:
            print(
                f"警告: ファイル '{csv_file_name}' のカラム '{column_to_extract}' から抽出できる有効なデータがありませんでした。{output_dir_for_book} にファイルは作成されませんでした。"
            )

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_csv_path}' が見つかりません。")
    except Exception as e:
        print(f"ファイル '{csv_file_name}' の処理中に予期せぬエラーが発生しました: {e}")

print("全ての処理が完了しました。")
