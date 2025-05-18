import os
from PIL import Image
import concurrent.futures
import argparse

def check_image(image_path):
    """
    画像ファイルが破損しているか確認する。
    破損していればファイルパスを、そうでなければNoneを返す。
    """
    try:
        img = Image.open(image_path)
        img.verify()  # verify() はヘッダー情報をチェックし、破損していれば例外を発生させる
        # verify() の後、再度openする必要がある場合がある
        img_reopen = Image.open(image_path)
        img_reopen.load() # load() は画像データを実際に読み込み、破損していれば例外を発生させる
        return None
    except (IOError, SyntaxError, Image.DecompressionBombError) as e:
        return image_path

def find_jpg_files(directory):
    """
    指定されたディレクトリ以下のすべてのjpgファイルを再帰的に検索する。
    """
    jpg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def main():
    parser = argparse.ArgumentParser(description="指定されたディレクトリ内のJPG画像が破損していないかチェックします。")
    parser.add_argument("target_directory", type=str, help="画像ファイルを検索するディレクトリのパス。")
    parser.add_argument("--workers", type=int, default=None, help="並列処理に使用するワーカーの数。デフォルトはCPUコア数。")
    args = parser.parse_args()

    image_dir = args.target_directory
    max_workers = args.workers

    if not os.path.isdir(image_dir):
        print(f"エラー: 指定されたディレクトリ '{image_dir}' が存在しません。")
        return

    print(f"'{image_dir}' 内のJPG画像を検索しています...")
    image_paths = find_jpg_files(image_dir)

    if not image_paths:
        print("JPG画像が見つかりませんでした。")
        return

    print(f"{len(image_paths)} 件のJPG画像が見つかりました。破損チェックを開始します...")

    corrupted_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # future_to_path = {executor.submit(check_image, path): path for path in image_paths}
        # for future in concurrent.futures.as_completed(future_to_path):
        #     result = future.result()
        #     if result:
        #         corrupted_files.append(result)
        results = executor.map(check_image, image_paths)
        for result in results:
            if result:
                corrupted_files.append(result)


    if corrupted_files:
        print("\n破損している可能性のある画像ファイル:")
        for file_path in corrupted_files:
            print(file_path)
    else:
        print("\nすべてのJPG画像は正常に開けました。")

if __name__ == "__main__":
    main() 
