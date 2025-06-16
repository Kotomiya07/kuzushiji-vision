import os

import pandas as pd


def ensure_dir(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        #print(f"ディレクトリを作成しました: {directory}")


def unicode_to_char(unicode_str):
    """Unicode ID（例: 'U+601D'）から文字に変換する"""
    # 'U+'を削除して16進数に変換
    try:
        code_point = int(unicode_str.replace("U+", ""), 16)
        return chr(code_point)
    except:
        print(f"変換エラー: {unicode_str}")
        return ""


def process_dataset(dataset_type):
    """指定されたデータセット（train/val/test）のラベルを処理する"""
    base_dir = f"data/column_dataset_padded/{dataset_type}"
    csv_path = f"{base_dir}/{dataset_type}_column_info.csv"
    labels_dir = f"{base_dir}/labels"

    # ラベルディレクトリを作成
    ensure_dir(labels_dir)

    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    print(f"{dataset_type}データセットの処理を開始: {len(df)}行")

    for idx, row in df.iterrows():
        # 画像ファイル名を取得（拡張子を除く）
        image_path = row["column_image"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # book_id を image_path から抽出
        book_id = image_path.split("/")[2]

        # book_id を含む新しいラベル出力ディレクトリパスを作成
        book_labels_dir = os.path.join(labels_dir, book_id)
        ensure_dir(book_labels_dir)

        # Unicode IDのリストを取得して文字に変換
        unicode_ids = eval(row["unicode_ids"])  # 文字列リストを実際のリストに変換
        characters = [unicode_to_char(uid) for uid in unicode_ids]
        text = "".join(characters)

        # ラベルファイルに保存
        label_path = os.path.join(book_labels_dir, f"{image_name}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(text)

        # 進捗表示
        if (idx + 1) % 1000 == 0:
            print(f"{idx + 1}/{len(df)} 処理完了")

    #print(f"{dataset_type}データセットの処理が完了しました")


def main():
    """メイン処理"""
    print("ラベル作成処理を開始します")

    # 各データセットを処理
    for dataset_type in ["train", "val", "test"]:
        # 対応するCSVファイルが存在するか確認
        csv_path = f"data/column_dataset_padded/{dataset_type}/{dataset_type}_column_info.csv"
        if os.path.exists(csv_path):
            process_dataset(dataset_type)
        else:
            print(f"{csv_path} が見つかりません。スキップします。")

    print("すべての処理が完了しました")


if __name__ == "__main__":
    main()
