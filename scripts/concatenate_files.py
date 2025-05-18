import glob
import os


def get_all_text_files(data_dirs):
    """指定されたディレクトリ内のすべての.txtファイルへのパスのリストを取得します。"""
    all_files = []
    for data_dir in data_dirs:
        files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        print(f"ディレクトリ: {data_dir}, ファイル数: {len(files)}")
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"指定されたディレクトリにテキストファイルが見つかりませんでした: {data_dirs}")
    return all_files


def concatenate_files(file_list, output_dir):
    """指定されたファイルリストの内容を1つの一時ファイルに結合します。"""
    try:
        with open(os.path.join(output_dir, "honkoku.txt"), "w", encoding="utf-8") as outfile:
            for filepath in file_list:
                try:
                    with open(filepath, encoding="utf-8") as infile:
                        for line in infile:
                            outfile.write(line)
                        outfile.write("\n")  # 各ファイルの内容を書き込んだ後に改行を追加
                except Exception as e:
                    print(f"ファイル {filepath} の読み込み中にエラー: {e}")
    except Exception as e:
        print(f"一時ファイルの作成中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    DATA_DIRS = [
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
        "honkoku_yatanavi/honkoku_oneline",
        "data/oneline",
        "kokubunken_repo/text",
    ]
    file_list = get_all_text_files(DATA_DIRS)
    output_dir = "data/honkoku"
    os.makedirs(output_dir, exist_ok=True)
    concatenate_files(file_list, output_dir)
