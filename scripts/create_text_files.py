"""
このスクリプトは、CSVファイルからテキストファイルを作成するためのものです。
CSVファイルは、「kokubunken_repo/csv」ディレクトリにあります。
出力ファイルは、「kokubunken_repo/text」ディレクトリに作成されます。

CSVファイルのカラム名は、「file_column_mapping」変数で指定します。
CSVファイルのカラム名が上記マッピングにない場合は、「default_column_name」変数で指定したカラム名が使用されます。
"""

import csv
import hashlib  # ハッシュ化のために追加
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
file_column_mapping = {"二十一代集.csv": ["和歌原表記", "詞書原表記"]}
# 上記マッピングにないCSVファイルの場合のデフォルトカラム名
default_column_name = "本文領域（該当行）"

# pykakasiの共通インスタンスを作成
kks = kakasi()
kks.setMode("H", "a")  # Hiragana to ascii
kks.setMode("K", "a")  # Katakana to ascii
kks.setMode("J", "a")  # Japanese to ascii (kanji)
conv = kks.getConverter()


def romanize_name(name: str) -> str:
    """文字列をローマ字に変換し、ファイル名/ディレクトリ名として使えるように整形する"""
    romanized = conv.do(name)
    return re.sub(r"\\W+", "_", romanized).lower()


def process_text(text: str) -> str:
    """
    指定された記号の処理とスペースの正規化を行う

    全領域に使われる記号
    ＋＋\tヲドリ字（二字以上を繰り返す記号）
    ／＼\tヲドリ字（二字以上を繰り返す記号）
    ／″＼\tヲドリ字（二字以上を繰り返し、濁点の付く文字を含む記号）
    △\t原本のスペース
    〈　〉\t割書き

    標準領域に使われる記号
    ＄\t漢字に濁点が付されている場合、当該漢字の後
    ＝\t音読みを意味する連辞符（字間中央にあるもの）
    －\t音読みを意味する連辞符（字間左側にあるもの）
    ☆\t文字でない記号・系図部分の線など
    ｛／｝\t訓点を再現するために用いた記号、｛送り仮名／返り点｝

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

    # 「／」記号を削除 (ユーザー指示による追加)
    text = text.replace("／", "")

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

generated_text_files = []  # 生成された全テキストファイルのパスを保存するリスト

for csv_file_name in csv_files:
    input_csv_path = os.path.join(target_dir, csv_file_name)
    base_name_without_ext = os.path.splitext(csv_file_name)[0]
    romanized_book_name = romanize_name(base_name_without_ext)

    columns_to_process = file_column_mapping.get(csv_file_name)
    if not columns_to_process:
        columns_to_process = [default_column_name]  # デフォルトカラム名をリストとして扱う
    elif isinstance(columns_to_process, str):  # 後方互換性のため、もし文字列で定義されていてもリストに変換
        columns_to_process = [columns_to_process]

    for column_to_extract in columns_to_process:
        romanized_column_name = romanize_name(column_to_extract)
        output_dir_for_column = os.path.join(output_base_dir, romanized_book_name, romanized_column_name)

        if not os.path.exists(output_dir_for_column):
            os.makedirs(output_dir_for_column)
            print(f"ディレクトリ作成: {output_dir_for_column}")

        extracted_texts = []
        file_counter = 1

        print(f"処理中: {csv_file_name} (カラム: {column_to_extract}) -> 出力先: {output_dir_for_column}")

        try:
            # まずUTF-8で試行
            try:
                with open(input_csv_path, encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    if column_to_extract not in reader.fieldnames:
                        print(
                            f"警告 (UTF-8): ファイル '{csv_file_name}' のカラム '{column_to_extract}' が見つかりません。このカラムはスキップします。"
                        )
                        continue  # continue to the next column in columns_to_process

                    temp_extracted_texts = []
                    for _i, row in enumerate(reader):
                        text_original = row.get(column_to_extract)
                        if text_original and text_original.strip():
                            if column_to_extract == "和歌原表記":
                                # 「／」を含む元のテキストの長さを確認
                                if len(text_original) <= 50:
                                    # 50文字以内なら、「／」を削除してそのまま1つのテキストとして処理
                                    processed_text = process_text(text_original.strip())
                                    if processed_text:
                                        temp_extracted_texts.append(processed_text)
                                else:
                                    # 50文字を超える場合は、「／」で分割し、結合ロジックを適用
                                    segments = text_original.split("／")
                                    current_chunk_segments = []
                                    current_length = 0
                                    for segment in segments:
                                        segment = segment.strip()
                                        if not segment:  # 空のセグメントはスキップ
                                            continue

                                        # 現在のチャンクにこのセグメントを追加した場合の長さを予測
                                        # 連結する「／」の分も考慮（ただし最初以外）
                                        predicted_length = current_length + len(segment)
                                        if current_chunk_segments:  # 既に要素があれば「／」の1文字分追加
                                            predicted_length += 1

                                        if predicted_length <= 50:
                                            current_chunk_segments.append(segment)
                                            current_length = predicted_length
                                        else:
                                            # 現在のチャンクを処理
                                            if current_chunk_segments:
                                                chunk_to_process = "／".join(current_chunk_segments)
                                                processed_text = process_text(chunk_to_process)
                                                if processed_text:
                                                    temp_extracted_texts.append(processed_text)
                                            # 新しいチャンクを開始
                                            current_chunk_segments = [segment]
                                            current_length = len(segment)

                                    # ループ後に残っているチャンクを処理
                                    if current_chunk_segments:
                                        chunk_to_process = "／".join(current_chunk_segments)
                                        processed_text = process_text(chunk_to_process)
                                        if processed_text:
                                            temp_extracted_texts.append(processed_text)
                            elif column_to_extract == "詞書原表記":
                                # 「／」を含む元のテキストの長さを確認
                                if len(text_original) <= 50:
                                    # 50文字以内なら、「、」を削除してそのまま1つのテキストとして処理
                                    processed_text = process_text(text_original.strip())
                                    if processed_text:
                                        temp_extracted_texts.append(processed_text)
                                else:
                                    # 50文字を超える場合は、「、」で分割し、結合ロジックを適用
                                    segments = text_original.split("、")
                                    current_chunk_segments = []
                                    current_length = 0
                                    for segment in segments:
                                        segment = segment.strip()
                                        if not segment:  # 空のセグメントはスキップ
                                            continue

                                        # 現在のチャンクにこのセグメントを追加した場合の長さを予測
                                        # 連結する「、」の分も考慮（ただし最初以外）
                                        predicted_length = current_length + len(segment)
                                        if current_chunk_segments:  # 既に要素があれば「、」の1文字分追加
                                            predicted_length += 1

                                        if predicted_length <= 50:
                                            current_chunk_segments.append(segment)
                                            current_length = predicted_length
                                        else:
                                            # 現在のチャンクを処理
                                            if current_chunk_segments:
                                                chunk_to_process = "、".join(current_chunk_segments)
                                                processed_text = process_text(chunk_to_process)
                                                if processed_text:
                                                    temp_extracted_texts.append(processed_text)
                                            # 新しいチャンクを開始
                                            current_chunk_segments = [segment]
                                            current_length = len(segment)

                                    # ループ後に残っているチャンクを処理
                                    if current_chunk_segments:
                                        chunk_to_process = "、".join(current_chunk_segments)
                                        processed_text = process_text(chunk_to_process)
                                        if processed_text:
                                            temp_extracted_texts.append(processed_text)
                            else:  # 「和歌原表記」以外のカラム
                                processed_text = process_text(text_original.strip())
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
                                f"警告 (CP932): ファイル '{csv_file_name}' のカラム '{column_to_extract}' が見つかりません。このカラムはスキップします。"
                            )
                            continue  # continue to the next column in columns_to_process

                        temp_extracted_texts = []
                        for _i, row in enumerate(reader):
                            text_original = row.get(column_to_extract)
                            if text_original and text_original.strip():
                                if column_to_extract == "和歌原表記":
                                    if len(text_original) <= 50:
                                        processed_text = process_text(text_original.strip())
                                        if processed_text:
                                            temp_extracted_texts.append(processed_text)
                                    else:
                                        segments = text_original.split("／")
                                        current_chunk_segments = []
                                        current_length = 0
                                        for segment in segments:
                                            segment = segment.strip()
                                            if not segment:
                                                continue

                                            predicted_length = current_length + len(segment)
                                            if current_chunk_segments:
                                                predicted_length += 1

                                            if predicted_length <= 50:
                                                current_chunk_segments.append(segment)
                                                current_length = predicted_length
                                            else:
                                                if current_chunk_segments:
                                                    chunk_to_process = "／".join(current_chunk_segments)
                                                    processed_text = process_text(chunk_to_process)
                                                    if processed_text:
                                                        temp_extracted_texts.append(processed_text)
                                                current_chunk_segments = [segment]
                                                current_length = len(segment)

                                    # ループ後に残っているチャンクを処理 (CP932)
                                    if current_chunk_segments:
                                        chunk_to_process = "／".join(current_chunk_segments)
                                        processed_text = process_text(chunk_to_process)
                                        if processed_text:
                                            temp_extracted_texts.append(processed_text)

                                else:  # 「和歌原表記」以外のカラム
                                    processed_text = process_text(text_original.strip())
                                    if processed_text:
                                        temp_extracted_texts.append(processed_text)
                        extracted_texts = temp_extracted_texts
                except UnicodeDecodeError:
                    print(
                        f"エラー: ファイル '{csv_file_name}' はUTF-8およびCP932でのデコードに失敗しました。このカラムはスキップします。"
                    )
                    continue  # continue to the next column in columns_to_process
                except Exception as e_cp932:
                    print(
                        f"ファイル '{csv_file_name}' (カラム: {column_to_extract}) のCP932処理中に予期せぬエラーが発生しました: {e_cp932}"
                    )
                    continue  # continue to the next column in columns_to_process
            except Exception as e_utf8:
                print(
                    f"ファイル '{csv_file_name}' (カラム: {column_to_extract}) のUTF-8処理中に予期せぬエラーが発生しました: {e_utf8}"
                )
                continue  # continue to the next column in columns_to_process

            if extracted_texts:
                for _i, text_content in enumerate(extracted_texts):
                    output_txt_file_name = f"{file_counter:05d}.txt"
                    output_txt_path = os.path.join(output_dir_for_column, output_txt_file_name)
                    try:
                        with open(output_txt_path, "w", encoding="utf-8") as outfile:
                            outfile.write(text_content + "\n")
                        generated_text_files.append(output_txt_path)  # 生成されたファイルパスをリストに追加
                        file_counter += 1
                    except Exception as e_write:
                        print(f"エラー: ファイル '{output_txt_path}' の書き込み中にエラーが発生しました: {e_write}")
                print(
                    f"完了: {csv_file_name} (カラム: {column_to_extract}) から {len(extracted_texts)}行のテキストを {output_dir_for_column} に個別ファイルとして保存しました。"
                )
            else:
                print(
                    f"警告: ファイル '{csv_file_name}' のカラム '{column_to_extract}' から抽出できる有効なデータがありませんでした。{output_dir_for_column} にファイルは作成されませんでした。"
                )

        except FileNotFoundError:
            print(f"エラー: ファイル '{input_csv_path}' が見つかりません。このCSVファイルはスキップします。")
            continue  # to the next csv_file_name
        except Exception as e:
            print(f"ファイル '{csv_file_name}' (カラム: {column_to_extract}) の処理中に予期せぬエラーが発生しました: {e}")
            # ここではカラム処理のループを継続するか、CSVファイル処理のループを継続するか検討。
            # カラム固有のエラーなら continue で次のカラムへ、そうでなければCSVの次のファイルへ。
            # 現状は次のカラムへ進むようになっているが、より広範なエラーなら break や外側の continue が適切かもしれない。
            # 今回は column_to_extract を含むエラーメッセージが出ているので、カラムごとのスキップでよいと判断。
            continue  # to the next column in columns_to_process

# --- 重複ファイルの削除処理 ---
print("\n重複ファイルの削除処理を開始します...")
hashes = {}
duplicates_found = 0
for filepath in generated_text_files:
    if not os.path.exists(filepath):  # 書き込み失敗などでファイルが存在しない場合を考慮
        continue
    try:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in hashes:
            hashes[file_hash].append(filepath)
        else:
            hashes[file_hash] = [filepath]
    except Exception as e:
        print(f"エラー: ファイル '{filepath}' のハッシュ計算中にエラーが発生しました: {e}")

for file_hash, filepaths in hashes.items():
    if len(filepaths) > 1:
        print(f"重複検出: ハッシュ {file_hash} に対応するファイルが {len(filepaths)} 個見つかりました。")
        # 最初のファイル以外を削除
        files_to_keep = filepaths[0]
        print(f"  保持: {files_to_keep}")
        for filepath_to_delete in filepaths[1:]:
            try:
                os.remove(filepath_to_delete)
                print(f"  削除: {filepath_to_delete}")
                duplicates_found += 1
            except Exception as e:
                print(f"エラー: ファイル '{filepath_to_delete}' の削除中にエラーが発生しました: {e}")
if duplicates_found > 0:
    print(f"{duplicates_found} 個の重複ファイルを削除しました。")
else:
    print("重複ファイルは見つかりませんでした。")

print("全ての処理が完了しました。")
