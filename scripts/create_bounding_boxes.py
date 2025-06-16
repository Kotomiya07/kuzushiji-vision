import ast
import json  # JSONを扱うために追加
import os

import pandas as pd


def ensure_dir(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        #print(f"ディレクトリを作成しました: {directory}")


def process_dataset(dataset_type):
    """指定されたデータセット（train/val/test）のバウンディングボックス情報を処理する"""
    base_dir = f"data/column_dataset_padded/{dataset_type}"
    csv_path = f"data/column_dataset/{dataset_type}/{dataset_type}_column_info.csv"
    # 出力ディレクトリ名を bounding_boxes に変更
    output_dir = f"{base_dir}/bounding_boxes"

    # 出力ディレクトリを作成
    ensure_dir(output_dir)

    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    print(f"{dataset_type}データセットのバウンディングボックス情報処理を開始: {len(df)}行")

    for idx, row in df.iterrows():
        # 画像ファイル名を取得（拡張子を除く）
        image_path = row["column_image"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # book_id を image_path から抽出
        book_id = image_path.split("/")[2]

        # book_id を含む新しい出力ディレクトリパスを作成
        book_output_dir = os.path.join(output_dir, book_id)
        ensure_dir(book_output_dir)

        # char_boxes_in_column カラムからバウンディングボックス情報を取得
        try:
            bounding_boxes = ast.literal_eval(row["char_boxes_in_column"])
        except Exception as e:
            print(f"エラー: {image_name} の char_boxes_in_column のパースに失敗しました。スキップします。エラー詳細: {e}")
            continue

        # バウンディングボックス情報をJSONファイルに保存
        output_json_path = os.path.join(book_output_dir, f"{image_name}.json")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(bounding_boxes, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"エラー: {output_json_path} への書き込みに失敗しました。エラー詳細: {e}")
            continue

        # 進捗表示
        if (idx + 1) % 1000 == 0:
            print(f"{idx + 1}/{len(df)} 処理完了")

    #print(f"{dataset_type}データセットのバウンディングボックス情報処理が完了しました")


def main():
    """メイン処理"""
    print("バウンディングボックス情報作成処理を開始します")

    # 各データセットを処理
    for dataset_type in ["train", "val", "test"]:
        # 対応するCSVファイルが存在するか確認
        csv_path = f"data/column_dataset/{dataset_type}/{dataset_type}_column_info.csv"
        if os.path.exists(csv_path):
            process_dataset(dataset_type)
        else:
            print(f"{csv_path} が見つかりません。スキップします。")

    print("すべてのバウンディングボックス情報作成処理が完了しました")


if __name__ == "__main__":
    main()
