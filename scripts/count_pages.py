# scripts/count_pages.py
import argparse
from pathlib import Path

def count_pages(dataset_dir: Path):
    """
    指定されたデータセットディレクトリ内の各書籍のページ数をカウントし、
    ページ数の昇順で表示します。

    Args:
        dataset_dir (Path): データセットのルートディレクトリ (例: data/raw/dataset)
    """
    if not dataset_dir.is_dir():
        print(f"エラー: ディレクトリが見つかりません: {dataset_dir}")
        return

    print(f"データセットディレクトリ: {dataset_dir}")
    print("-" * 30)

    book_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if not book_dirs:
        print("書籍ディレクトリが見つかりません。")
        return

    book_page_counts = []
    for book_dir in book_dirs:
        images_dir = book_dir / "images"
        page_count = 0
        if images_dir.is_dir():
            # .jpg ファイルのみをカウント（隠しファイルなどを除外）
            page_count = len(list(images_dir.glob("*.jpg")))
        else:
            # images ディレクトリがない場合も考慮
            print(f"警告: {book_dir.name} に images ディレクトリが見つかりません。")

        book_page_counts.append((book_dir.name, page_count))

    # ページ数で昇順にソート
    book_page_counts.sort(key=lambda item: item[1])

    # ソート結果を表示
    for book_name, count in book_page_counts:
        print(f"書籍ID: {book_name}, ページ数: {count}")

    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセット内の各書籍のページ数をカウントし、ページ数で昇順に表示します。")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/raw/dataset",
        help="データセットのルートディレクトリのパス",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    count_pages(dataset_path)
