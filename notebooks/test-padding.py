# %%
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def list_images_with_pathlib(root_dir):
    p = Path(root_dir)
    # iterdir() や glob() を使えますが、再帰的には rglob を使うのが便利
    return [str(f) for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]


# 使用例
# images = list_images_with_pathlib('data/processed/column_images/100241706/100241706_00005_1')
# for p in images:
#     print(p)

# 画像ファイルが格納されているディレクトリのパス
# '@/' は通常プロジェクトルートを示すため、ここではカレントディレクトリからの相対パスを使用します
# 必要に応じて絶対パスや適切な相対パスに変更してください
# image_dir = "../data/processed/column_images/100241706/100241706_00004_2"  # 例: './data/processed' や '/path/to/your/images'
image_dir = "../data/raw/dataset/200021869/images/"

# 対応する画像拡張子 (必要に応じて追加・変更)
extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

image_paths = list_images_with_pathlib(image_dir)

# 取得した画像パスの数を確認
print(f"Found {len(image_paths)} images in '{image_dir}'.")

# 最初のいくつかのパスを表示（確認用）
print("First few image paths:")
for p in image_paths[:5]:
    print(p)


def calculate_average_color(image_path):
    """
    画像ファイルのパスを受け取り、全ピクセルの平均色 (BGR) を計算する関数。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        tuple: 平均色 (B, G, R)。画像が読み込めない場合は None。
    """
    try:
        # 画像をカラーで読み込む
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # 画像の高さと幅の軸に沿って平均値を計算 (チャンネルごとに平均)
        # 結果は [平均B, 平均G, 平均R] の NumPy 配列になる
        average_color_bgr = np.mean(img, axis=(0, 1))

        # NumPy 配列を整数のタプルに変換して返す
        return tuple(map(int, average_color_bgr))

    except Exception as e:
        print(f"Error calculating average color for {image_path}: {e}")
        return None


def calculate_average_color_exclude_black(image_path, black_threshold=30):
    """
    黒い色（BGRすべてがblack_threshold以下）を除外し、残りピクセルの平均色（BGR）を返す関数。

    Args:
        image_path (str): 画像ファイルのパス。
        black_threshold (int): 黒色とみなす閾値（B,G,Rすべてがこの値以下なら黒と判定）。

    Returns:
        tuple: 平均色 (B, G, R)。該当ピクセルがなければNone。
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
        # 黒以外のピクセルのマスクを作成
        mask = ~((img[:, :, 0] <= black_threshold) & (img[:, :, 1] <= black_threshold) & (img[:, :, 2] <= black_threshold))
        mask = mask.astype(bool)
        # 黒以外のピクセルを抽出
        non_black_pixels = img[mask]
        if non_black_pixels.size == 0:
            print(f"Warning: No non-black pixels found in {image_path}")
            return None
        average_color_bgr = np.mean(non_black_pixels, axis=0)
        return tuple(map(int, average_color_bgr))
    except Exception as e:
        print(f"Error calculating average color (exclude black) for {image_path}: {e}")
        return None


def calculate_average_background_color_otsu(image_path):
    """
    大津の二値化で背景・前景を分離し、背景ピクセルのみの平均色（BGR）を返す関数。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        tuple: 背景ピクセルの平均色 (B, G, R)。背景がなければNone。
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 大津の二値化
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 背景は255（白）側と仮定（和紙や紙の画像なら多くは白背景）
        background_mask = mask == 255
        if np.sum(background_mask) == 0:
            print(f"Warning: No background pixels found in {image_path}")
            return None
        background_pixels = img[background_mask]
        average_color_bgr = np.mean(background_pixels, axis=0)
        return tuple(map(int, average_color_bgr))
    except Exception as e:
        print(f"Error calculating average background color (Otsu) for {image_path}: {e}")
        return None


# %%
# 関数のテスト (最初の画像で試す)
if image_paths:
    avg_color = calculate_average_color(image_paths[0])
    if avg_color is not None:
        print(f"Average color (BGR) for {image_paths[0]}: {avg_color}")
        avg_color_patch = np.full((50, 50, 3), avg_color, dtype=np.uint8)
        plt.imshow(cv2.cvtColor(avg_color_patch, cv2.COLOR_BGR2RGB))
        plt.title("Average Color")
        plt.axis("off")
        plt.show()
    # 新関数のテスト
    avg_color_ex_black = calculate_average_color_exclude_black(image_paths[0], black_threshold=150)
    if avg_color_ex_black is not None:
        print(f"Average color (BGR, exclude black) for {image_paths[0]}: {avg_color_ex_black}")
        avg_color_patch_ex_black = np.full((50, 50, 3), avg_color_ex_black, dtype=np.uint8)
        plt.imshow(cv2.cvtColor(avg_color_patch_ex_black, cv2.COLOR_BGR2RGB))
        plt.title("Average Color (Exclude Black)")
        plt.axis("off")
        plt.show()
    # 大津アルゴリズムによる背景平均色のテスト
    avg_bg_color_otsu = calculate_average_background_color_otsu(image_paths[0])
    if avg_bg_color_otsu is not None:
        print(f"Average background color (BGR, Otsu) for {image_paths[0]}: {avg_bg_color_otsu}")
        avg_bg_color_patch_otsu = np.full((50, 50, 3), avg_bg_color_otsu, dtype=np.uint8)
        plt.imshow(cv2.cvtColor(avg_bg_color_patch_otsu, cv2.COLOR_BGR2RGB))
        plt.title("Average Background Color (Otsu)")
        plt.axis("off")
        plt.show()
else:
    print("No images found to test the average color function.")
# %%

# --- 大津の二値化による分離の可視化 ---
if image_paths:
    img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Otsu Binarization\n(Background/Foreground)")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Could not read image for Otsu visualization: {image_paths[0]}")

if image_paths:
    img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(gray, cmap="gray")
        axs[1].set_title("Gray Image")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Could not read image for Otsu visualization: {image_paths[0]}")
# %%


def resize_and_pad(image_path, target_size=(224, 224), bg_color=(255, 255, 255)):
    """
    画像を読み込み、アスペクト比を維持してリサイズし、指定された背景色でパディングする関数。

    Args:
        image_path (str): 画像ファイルのパス。
        target_size (tuple): 目標の (幅, 高さ)。
        bg_color (tuple): 背景色 (B, G, R)。OpenCVはBGR順で色を扱うため注意。

    Returns:
        numpy.ndarray: リサイズおよびパディングされた画像。エラー時は None。
    """
    try:
        # 画像を読み込む (カラーで読み込む)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None

        h, w = img.shape[:2]
        target_w, target_h = target_size

        # アスペクト比を計算
        aspect_ratio = w / h

        # 目標サイズに合わせてリサイズ後のサイズを計算
        if aspect_ratio > (target_w / target_h):
            # 幅が基準
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            # 高さが基準
            new_h = target_h
            new_w = int(new_h * aspect_ratio)

        # 画像をリサイズ (INTER_AREA は縮小に適している)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 背景画像を作成 (目標サイズで、指定された背景色)
        # np.full は (高さ, 幅, チャンネル数) の順で形状を指定
        background = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

        # リサイズされた画像を背景の中央に配置するためのオフセットを計算
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2

        # 背景画像にリサイズされた画像を貼り付け
        background[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized_img

        return background

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# %%
# 関数のテスト (最初の画像で試す)
if image_paths:
    target_w, target_h = 256, 256  # 目標サイズ
    background_color_bgr = (128, 128, 128)  # 背景色 (灰色)

    processed_image = resize_and_pad(image_paths[0], target_size=(target_w, target_h), bg_color=background_color_bgr)

    if processed_image is not None:
        print(f"Successfully processed {image_paths[0]}. Output shape: {processed_image.shape}")
        # 表示のためにBGRからRGBに変換
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title("Test Processed Image")
        plt.axis("off")
        plt.show()
else:
    print("No images found to test the function.")

# %%
# 処理パラメータ
target_width = 100
target_height = 500
bg_color_bgr = (128, 128, 128)  # 背景色 (BGR形式)

# 表示する画像の数
num_display = min(5, len(image_paths))  # 最大5枚、または存在する画像の数

if num_display > 0:
    plt.figure(figsize=(10, 5 * num_display))

    for i in range(num_display):
        img_path = image_paths[i]

        bg_color_bgr = calculate_average_background_color_otsu(img_path)

        # 元の画像を読み込み (表示用)
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Skipping display for {img_path} as it couldn't be read.")
            continue

        # リサイズとパディングを実行
        processed_img = resize_and_pad(img_path, target_size=(target_width, target_height), bg_color=bg_color_bgr)

        if processed_img is None:
            print(f"Skipping display for {img_path} due to processing error.")
            continue

        # 元の画像を表示 (BGR -> RGB)
        plt.subplot(num_display, 2, 2 * i + 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {os.path.basename(img_path)}\nShape: {original_img.shape[:2]}")
        plt.axis("off")

        # 処理後の画像を表示 (BGR -> RGB)
        plt.subplot(num_display, 2, 2 * i + 2)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.title(
            f"Processed\nTarget: ({target_width},{target_height}), Shape: {processed_img.shape[:2]}, Avg Color: {background_color_bgr}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No images found to process and display.")

# --- オプション: 処理結果を保存する場合 ---
# save_dir = './data/resized_padded'
# os.makedirs(save_dir, exist_ok=True)
#
# print(f"\nProcessing and saving all images to '{save_dir}'...")
# for img_path in image_paths:
#     processed_img = resize_and_pad(img_path, target_size=(target_width, target_height), bg_color=bg_color_bgr)
#     if processed_img is not None:
#         base_name = os.path.basename(img_path)
#         save_path = os.path.join(save_dir, base_name)
#         try:
#             cv2.imwrite(save_path, processed_img)
#             # print(f"Saved: {save_path}")
#         except Exception as e:
#             print(f"Error saving {save_path}: {e}")
# print("Finished processing and saving.")
# --- ここまでオプション ---
# %%
