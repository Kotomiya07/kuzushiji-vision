# %%
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt  # matplotlib.pyplotをインポート
from PIL import Image, ImageDraw, ImageFont
import japanize_matplotlib

# コマンドライン引数を設定
parser = argparse.ArgumentParser(description="Visualize annotations on images.")
parser.add_argument("--num_images", type=int, default=10, help="Number of images to display per dataset type.")
args, _ = parser.parse_known_args()

# 表示する画像の件数を取得
num_images_to_display = args.num_images

# ルートディレクトリパスを設定
base_data_dir = "data/column_dataset_padded"

# 日本語フォントファイルパスを設定（環境に合わせて変更してください）
# 例: font_path = '/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf'
font_path = "assets/font/fonts-japanese-gothic.ttf"  # サンプルパス。適切なパスに置き換えてください。
font_size = 20  # フォントサイズ

# フォントをロード
try:
    font = ImageFont.truetype(font_path, font_size)
except OSError:
    print(f"エラー: フォントファイルが見つかりませんまたは読み込めません: {font_path}")
    print("適切な日本語フォントファイルのパスを設定してください。")
    font = ImageFont.load_default()  # デフォルトフォントを使用（日本語表示できない可能性あり）

# データセットの種類
dataset_types = ["train", "val", "test"]

print(f"指定された表示件数: {num_images_to_display}")

for dataset_type in dataset_types:
    print(f"\nデータセット: {dataset_type} の処理を開始します...")

    images_dir = os.path.join(base_data_dir, dataset_type, "images")
    bboxes_dir = os.path.join(base_data_dir, dataset_type, "bounding_boxes")
    labels_dir = os.path.join(base_data_dir, dataset_type, "labels")

    # ディレクトリが存在するか確認
    if not os.path.exists(images_dir):
        print(f"エラー: 画像ディレクトリが見つかりません: {images_dir}")
        continue
    if not os.path.exists(bboxes_dir):
        print(f"エラー: バウンディングボックスディレクトリが見つかりません: {bboxes_dir}")
        continue
    if not os.path.exists(labels_dir):
        print(f"エラー: ラベルディレクトリが見つかりません: {labels_dir}")
        continue

    # 画像ファイルリストを再帰的に取得
    image_paths = []
    try:
        # **/* でimages_dir以下の全てのファイルとディレクトリを取得し、画像拡張子でフィルタ
        all_files_and_dirs = glob.glob(os.path.join(images_dir, "**", "*"), recursive=True)
        image_extensions = (".png", ".jpg", ".jpeg")
        image_paths = [f for f in all_files_and_dirs if os.path.isfile(f) and f.lower().endswith(image_extensions)]

        if not image_paths:
            print(f"警告: {images_dir} 内に画像ファイルが見つかりません。")
            continue
    except Exception as e:
        print(f"エラー: {images_dir} 内の画像ファイルリスト取得に失敗しました: {e}")
        continue

    # 表示する画像をn件選択（または全件）
    images_to_process = image_paths[:num_images_to_display]
    print(f"{len(images_to_process)} 件の画像を処理対象とします。")

    # ここから個別の画像処理ループを実装
    for image_path in images_to_process:
        print(f"  処理中: {image_path}")

        # images_dirからの相対パスを取得し、拡張子を除去
        relative_image_path = os.path.relpath(image_path, images_dir)
        file_base_name_with_subdir = os.path.splitext(relative_image_path)[0]

        # 対応するJSONファイルとTXTファイルのパスを生成
        # バウンディングボックスとラベルのディレクトリ構造がimagesと同じであることを前提
        bbox_file = os.path.join(bboxes_dir, file_base_name_with_subdir + ".json")
        label_file = os.path.join(labels_dir, file_base_name_with_subdir + ".txt")

        # 画像を読み込み
        try:
            img = Image.open(image_path).convert("RGB")  # RGBに変換して透過をなくす
            draw = ImageDraw.Draw(img)
        except FileNotFoundError:
            # globで取得しているので発生しにくいが念のため
            print(f"    警告: 画像ファイルが見つかりません: {image_path}")
            continue
        except Exception as e:
            print(f"    エラー: 画像ファイルの読み込みに失敗しました {image_path}: {e}")
            continue

        # バウンディングボックスを読み込み
        bboxes = []
        try:
            with open(bbox_file) as f:
                bboxes = json.load(f)
        except FileNotFoundError:
            print(f"    警告: バウンディングボックスファイルが見つかりません: {bbox_file}")
            # バウンディングボックスがない場合は次のファイルへ
            continue
        except json.JSONDecodeError:
            print(f"    エラー: バウンディングボックスファイルのJSON形式が不正です: {bbox_file}")
            continue
        except Exception as e:
            print(f"    エラー: バウンディングボックスファイルの読み込みに失敗しました {bbox_file}: {e}")
            continue

        # ラベルテキストを読み込み
        label_text = ""
        try:
            with open(label_file, encoding="utf-8") as f:
                label_text = f.read().strip()
        except FileNotFoundError:
            print(f"    警告: ラベルファイルが見つかりません: {label_file}")
            # ラベルがない場合はバウンディングボックスのみ描画
        except Exception as e:
            print(f"    エラー: ラベルファイルの読み込みに失敗しました {label_file}: {e}")
            # ラベルがない場合はバウンディングボックスのみ描画

        # バウンディングボックスとラベルを描画
        # バウンディングボックスの数とテキストの文字数が一致すると仮定
        if len(bboxes) != len(label_text):
            print(
                f"    警告: バウンディングボックスの数 ({len(bboxes)}) とラベル文字数 ({len(label_text)}) が一致しません。ファイル: {image_path}"
            )
            # 数が一致しない場合は、一致する数だけ描画するなどの対応が必要になるが、ここでは一旦警告のみ
            min_count = min(len(bboxes), len(label_text))
        else:
            min_count = len(bboxes)

        for i in range(min_count):
            box = bboxes[i]
            char = label_text[i] if i < len(label_text) else ""  # ラベルが足りない場合のFallback

            # バウンディングボックスを描画 [x1, y1, x2, y2]
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)

            # ラベル文字を描画 (バウンディングボックスの右側に配置)
            # 文字の描画位置を調整
            text_x = box[2] + 5  # バウンディングボックスの右端から少し右
            text_y = box[1]  # バウンディングボックスの上端と同じ高さに合わせる

            if char:  # 文字が存在する場合のみ描画
                try:
                    draw.text((text_x, text_y), char, fill="blue", font=font)
                except Exception as e:
                    print(f"    エラー: 文字 '{char}' の描画に失敗しました: {e}")

        # 描画結果をMatplotlibで表示
        try:
            plt.figure(figsize=(10, 10))  # 画像サイズに合わせて調整
            plt.imshow(img)
            plt.axis("off")  # 軸を非表示
            plt.title(os.path.basename(image_path))  # ファイル名をタイトルに
            plt.show()
        except Exception as e:
            print(f"    エラー: Matplotlibでの画像表示に失敗しました {image_path}: {e}")

    print(f"データセット: {dataset_type} の処理が完了しました。")

print("\nスクリプトの実行が完了しました。")
# %%
