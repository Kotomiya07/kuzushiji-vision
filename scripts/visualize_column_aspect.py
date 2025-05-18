"""
指定されたディレクトリ内の画像ファイルを再帰的に探索し、各画像のアスペクト比（幅/高さ）を分析します。
設定された閾値に基づいて「極端なアスペクト比」を持つ画像を特定し、そのパスをリストアップします。
また、収集した全画像のアスペクト比、幅、高さのデータを用いて、分布を示すヒストグラムや散布図を生成・表示する機能も備えています。
これにより、データセット内の画像のサイズ傾向や特異な形状を持つ画像を視覚的に把握することができます。
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- 設定 ---
TARGET_DIR = "data/processed/column_images"  # 探索を開始するルートディレクトリ
HIGH_THRESHOLD = 5.0  # この値より大きいアスペクト比を「極端な横長」とする
LOW_THRESHOLD = 0.04  # この値より小さいアスペクト比を「極端な縦長」とする
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")  # 処理対象の拡張子
SHOW_DISTRIBUTION_PLOT = True  # アスペクト比の分布グラフを表示するかどうか
PLOT_BINS = 50  # ヒストグラムのビン（区間）の数
PLOT_LOG_SCALE = False  # X軸（アスペクト比）を対数スケールで表示するかどうか (True/False)
PLOT_RANGE = None  # ヒストグラムの表示範囲をタプルで指定 (例: (0, 1.5))。Noneの場合は自動調整。
# -------------


def find_images_and_ratios_recursive(directory, high_thresh, low_thresh, extensions):
    """
    指定されたディレクトリとそのサブディレクトリを再帰的に探索し、
    極端なアスペクト比を持つ画像のパスと、すべての画像のアスペクト比リストを返す。
    （この関数は前回のコードと同じです）
    """
    extreme_images = []
    all_aspect_ratios = []
    all_widths = []
    all_heights = []
    print(f"ディレクトリ '{directory}' およびそのサブディレクトリを再帰的に検索中...")
    print(f"極端なアスペクト比の閾値: > {high_thresh} または < {low_thresh}")
    print("-" * 30)

    if not os.path.isdir(directory):
        print(f"エラー: ディレクトリが見つかりません: {directory}")
        return [], []

    found_count = 0
    processed_count = 0
    error_count = 0

    for root, _, files in os.walk(directory):
        # print(f"探索中: {root}") # 詳細表示が必要な場合はコメント解除
        for filename in files:
            if filename.lower().endswith(extensions):
                file_path = os.path.join(root, filename)
                processed_count += 1
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        all_widths.append(width)
                        all_heights.append(height)

                        # 幅が64未満の画像のpathを表示
                        # if width < 64:
                        #    print(f"警告: 幅が64未満の画像を検出しました: {file_path} (サイズ: {width}x{height})")

                        # 高さが4000以上の画像のpathを表示
                        if height > 4000:
                            print(f"警告: 高さが4000以上の画像を検出しました: {file_path} (サイズ: {width}x{height})")

                        if height == 0:
                            print(f"警告: 画像の高さが0です。スキップします: {file_path}")
                            continue

                        aspect_ratio = width / height
                        all_aspect_ratios.append(aspect_ratio)  # 全てのアスペクト比を記録

                        # 極端なアスペクト比かチェック
                        if aspect_ratio > high_thresh or aspect_ratio < low_thresh:
                            extreme_images.append(file_path)
                            found_count += 1
                            # print(f"  極端な比率検出: {file_path} (サイズ: {width}x{height}, 比率: {aspect_ratio:.2f})")

                except UnidentifiedImageError:
                    print(f"警告: 画像ファイルとして認識できませんでした: {file_path}")
                    error_count += 1
                except FileNotFoundError:
                    print(f"警告: ファイルが見つかりませんでした（処理中に削除された可能性）: {file_path}")
                    error_count += 1
                except Exception as e:
                    print(f"エラー: 画像処理中にエラーが発生しました ({file_path}): {e}")
                    error_count += 1

    print("-" * 30)
    print("処理完了。")
    print(f"処理した画像ファイル数: {processed_count}")
    print(f"極端なアスペクト比の画像数: {found_count}")
    print(f"平均の幅: {np.mean(all_widths) if all_widths else 0:.2f}")
    print(f"平均の高さ: {np.mean(all_heights) if all_heights else 0:.2f}")
    if error_count > 0:
        print(f"処理中にエラーが発生したファイル数: {error_count}")

    return extreme_images, all_aspect_ratios, all_widths, all_heights


def plot_aspect_ratio_distribution_below_mean(all_aspect_ratios, bins=50, use_log_scale=False, plot_range=None):
    """
    アスペクト比のリストから、全体の平均値以下のデータのみでヒストグラムを作成して表示する。

    Args:
        all_aspect_ratios (list): 全てのアスペクト比の数値のリスト。
        bins (int): ヒストグラムのビンの数。
        use_log_scale (bool): X軸を対数スケールにするか。
        plot_range (tuple or None): ヒストグラムのX軸の表示範囲 (min, max)。Noneなら自動。
    """
    if not all_aspect_ratios:
        print("アスペクト比データがないため、分布グラフは表示できません。")
        return

    # numpy配列に変換し、有効な値のみを抽出
    ratios_np = np.array(all_aspect_ratios)
    ratios_np = ratios_np[np.isfinite(ratios_np)]  # 無限大やNaNを除外

    if len(ratios_np) == 0:
        print("有効なアスペクト比データがないため、分布グラフは表示できません。")
        return

    # 全データの平均値を計算
    overall_mean_ratio = np.mean(ratios_np)
    print(f"\n全データのアスペクト比 平均値: {overall_mean_ratio:.3f}")

    # 平均値以下のデータをフィルタリング
    filtered_ratios = ratios_np[ratios_np <= overall_mean_ratio]

    if len(filtered_ratios) == 0:
        print(f"平均値 ({overall_mean_ratio:.3f}) 以下のデータが見つかりませんでした。グラフは表示しません。")
        return

    print(f"平均値以下のデータ数: {len(filtered_ratios)} / {len(ratios_np)}")

    plt.figure(figsize=(10, 6))

    # フィルタリングされたデータでヒストグラムをプロット
    # plot_range が指定されていても、filtered_ratiosの範囲外なら無視される
    n, bin_edges, patches = plt.hist(filtered_ratios, bins=bins, range=plot_range, edgecolor="black")

    # 実際にプロットされた範囲に基づいてタイトルや軸を設定
    actual_min = bin_edges[0]
    actual_max = bin_edges[-1]

    plt.title(f"画像アスペクト比の分布 (全データ平均 {overall_mean_ratio:.2f} 以下, N={len(filtered_ratios)})")
    plt.xlabel(f"アスペクト比{' (対数スケール)' if use_log_scale else ''}")
    plt.ylabel("画像数")

    # 対数スケールを設定
    if use_log_scale:
        plt.xscale("log")
        # 対数スケールの場合、表示範囲の最小値が0以下にならないように調整
        current_xlim = plt.xlim()
        min_lim = current_xlim[0]
        if min_lim <= 0:
            positive_min = np.min(filtered_ratios[filtered_ratios > 0]) if np.any(filtered_ratios > 0) else 0.01
            plt.xlim(left=max(positive_min * 0.9, 1e-3))  # 最小値より少し小さい正の値から開始

    # 表示範囲を設定 (rangeで指定してもxlimで上書き/調整可能)
    if plot_range:
        # plot_rangeが指定されている場合は、その範囲を優先する (ただし対数スケール時の0以下は除く)
        xlim_min = plot_range[0]
        if use_log_scale and xlim_min <= 0:
            xlim_min = plt.xlim()[0]  # 対数スケールで設定された最小値を使う
        plt.xlim(xlim_min, plot_range[1])
    else:
        # 自動調整の場合、少しマージンを持たせる（任意）
        margin = (actual_max - actual_min) * 0.05
        if not use_log_scale:  # 通常スケールなら左も調整
            plt.xlim(max(0, actual_min - margin), actual_max + margin)
        else:  # 対数スケールなら右のみ調整
            plt.xlim(plt.xlim()[0], actual_max + margin)

    # フィルタリング後のデータの統計量を計算して表示
    mean_filtered = np.mean(filtered_ratios)
    median_filtered = np.median(filtered_ratios)
    plt.axvline(mean_filtered, color="red", linestyle="dashed", linewidth=1, label=f"平均 (表示データ): {mean_filtered:.2f}")
    plt.axvline(
        median_filtered, color="green", linestyle="dashed", linewidth=1, label=f"中央値 (表示データ): {median_filtered:.2f}"
    )

    # 全体の平均値も参考線として引く (フィルタリング境界を示す)
    if overall_mean_ratio >= plt.xlim()[0] and overall_mean_ratio <= plt.xlim()[1]:  # グラフ範囲内なら表示
        plt.axvline(
            overall_mean_ratio,
            color="purple",
            linestyle="dotted",
            linewidth=1.5,
            label=f"全データ平均: {overall_mean_ratio:.2f}",
        )

    # 1:1の比率を示す線 (グラフ範囲内なら表示)
    if 1.0 >= plt.xlim()[0] and 1.0 <= plt.xlim()[1]:
        plt.axvline(1.0, color="blue", linestyle="dotted", linewidth=1, label="1:1 (正方形)")

    plt.legend()  # 凡例を表示
    plt.grid(axis="y", alpha=0.75)  # グリッド線を表示
    plt.tight_layout()  # レイアウトを調整
    plt.show()  # グラフを表示


# --- スクリプトの実行 ---
if __name__ == "__main__":
    extreme_image_paths, all_ratios, all_widths, all_heights = find_images_and_ratios_recursive(
        TARGET_DIR, HIGH_THRESHOLD, LOW_THRESHOLD, SUPPORTED_EXTENSIONS
    )

    # 極端な画像のパスを表示（必要な場合）
    if extreme_image_paths:
        print(f"\n--- 極端なアスペクト比 ({len(extreme_image_paths)}件) ---")
        # for path in extreme_image_paths:
        #    print(path) # 検出時に表示済み
    elif not all_ratios:
        print("\n処理対象の画像が見つかりませんでした。")
    else:
        print("\n指定された条件に合う極端なアスペクト比の画像は見つかりませんでした。")

    # アスペクト比の分布グラフ（平均以下）を表示
    if SHOW_DISTRIBUTION_PLOT and all_ratios:
        print("\n全データ平均以下のアスペクト比の分布をプロットします...")
        plot_aspect_ratio_distribution_below_mean(  # 関数名を変更
            all_ratios, bins=PLOT_BINS, use_log_scale=PLOT_LOG_SCALE, plot_range=PLOT_RANGE
        )
    elif SHOW_DISTRIBUTION_PLOT:
        print("\nグラフ表示が有効ですが、表示するアスペクト比データがありません。")

    # 高さと幅の分布グラフを表示
    if SHOW_DISTRIBUTION_PLOT and all_widths and all_heights:
        print("\n幅と高さの分布をプロットします...")
        plt.figure(figsize=(12, 6))

        # 幅のヒストグラム
        plt.subplot(1, 2, 1)
        plt.hist(all_widths, bins=PLOT_BINS, edgecolor="black")
        plt.title("画像幅の分布")
        plt.xlabel("幅 (ピクセル)")
        plt.ylabel("画像数")

        # 高さのヒストグラム
        plt.subplot(1, 2, 2)
        plt.hist(all_heights, bins=PLOT_BINS, edgecolor="black")
        plt.title("画像高さの分布")
        plt.xlabel("高さ (ピクセル)")
        plt.ylabel("画像数")

        plt.tight_layout()
        plt.show()

    # 縦軸が高さ、横軸が幅の散布図を表示
    if SHOW_DISTRIBUTION_PLOT and all_widths and all_heights:
        print("\n幅と高さの散布図をプロットします...")
        plt.figure(figsize=(8, 8))
        plt.scatter(all_widths, all_heights, alpha=0.5)
        plt.title("画像の幅と高さの散布図")
        plt.xlabel("幅 (ピクセル)")
        plt.ylabel("高さ (ピクセル)")
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850])
        plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
        # plt.xscale('log')
        # plt.yscale('log')
        # 幅に対して高さが16倍になる線を引く
        plt.plot([0, 400], [0, 6400], color="red", linestyle="dashed", linewidth=1, label="高さ = 16 * 幅")
        plt.grid(True)
        plt.show()

    # 高さがh_minからh_maxで幅がw_minからw_maxまでの画像が全体の何%かを計算
    h_min = 100
    h_max = 3500
    w_min = 100
    w_max = 700
    count_in_range = sum(
        1 for w, h in zip(all_widths, all_heights, strict=False) if w_min <= w <= w_max and h_min <= h <= h_max
    )
    total_count = len(all_widths)
    if total_count > 0:
        percentage_in_range = (count_in_range / total_count) * 100
        print(
            f"\n幅が{w_min}から{w_max}ピクセル、高さが{h_min}から{h_max}ピクセルの画像は\n全体の {percentage_in_range:.2f}% です。"
        )
    else:
        print("\n画像データがありません。範囲内の割合を計算できません。")

# %%
