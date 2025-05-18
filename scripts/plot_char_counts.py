# %%
from pathlib import Path

import matplotlib.pyplot as plt

import japanize_matplotlib


def count_images_in_directories(base_dir):
    char_counts = {}
    base_path = Path(base_dir)

    # 各Unicodeディレクトリを処理
    for char_dir in base_path.iterdir():
        if char_dir.is_dir():
            # ディレクトリ内の画像ファイルをカウント
            image_count = len([f for f in char_dir.glob("*") if f.is_file()])
            char_counts[char_dir.name] = image_count

    return char_counts


def plot_char_counts(char_counts):
    # プロットの設定
    plt.figure(figsize=(15, 6))

    # データをソート
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    chars = [x[0] for x in sorted_chars]
    counts = [x[1] for x in sorted_chars]

    # 棒グラフを作成
    plt.bar(range(len(chars)), counts, color="skyblue")
    plt.xticks(range(len(chars)), chars, rotation=90)

    # グラフの設定
    plt.title("各Unicode文字ごとの画像数")
    plt.xlabel("Unicode文字")
    plt.ylabel("画像数")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # グラフを保存
    plt.savefig("char_counts.png")
    plt.close()


def main():
    base_dir = "../data/onechannel"
    char_counts = count_images_in_directories(base_dir)
    plot_char_counts(char_counts)

    # 結果を表示
    print("各Unicode文字ごとの画像数:")
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{char}: {count}枚")

    # 文字の総数
    print(f"文字の総数: {len(char_counts)}")
    # 1枚しかない文字の総数
    print(f"1枚しかない文字の総数: {sum(1 for count in char_counts.values() if count == 1)}")
    # 文字の総数: 4328
    # 1枚しかない文字の総数: 790

if __name__ == "__main__":
    main()
# %%
