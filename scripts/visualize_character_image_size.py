# %%
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

# Pandasが利用可能か試す
try:
    import pandas as pd

    use_pandas = True
except ImportError:
    use_pandas = False

# Matplotlib と Numpy が利用可能か試す
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker  # 軸のフォーマット用
    import numpy as np  # スケーリングのために追加

    use_matplotlib = True
except ImportError:
    use_matplotlib = False
    print("警告: Matplotlib または Numpy が見つかりません。グラフ表示機能は無効になります。", file=sys.stderr)
    print("インストールするには 'pip install matplotlib numpy' を実行してください。", file=sys.stderr)


def analyze_image_sizes(root_dir_path: Path):
    """
    指定されたディレクトリ内の画像ファイルのサイズ分布を分析する関数。
    (内容は変更なし)
    """
    target_pattern = "*/characters/*/*jpg"
    image_files = list(root_dir_path.glob(target_pattern))

    if not image_files:
        print(
            f"エラー: パターン '{root_dir_path / target_pattern}' に一致するファイルが見つかりませんでした。", file=sys.stderr
        )
        print(f"指定されたディレクトリ '{root_dir_path.resolve()}' の構造を確認してください。", file=sys.stderr)
        return None

    print(f"対象ファイルを探索中... パターン: {root_dir_path / target_pattern}")
    print(f"候補ファイル数: {len(image_files)} 個")

    size_counter = Counter()
    processed_count = 0
    error_count = 0
    skipped_dirs = 0
    total_files = len(image_files)
    update_interval = max(1, total_files // 100)

    for i, image_path in enumerate(image_files):
        if (i + 1) % update_interval == 0 or (i + 1) == total_files:
            progress = (i + 1) / total_files * 100
            print(f"\r処理中: {i + 1}/{total_files} ({progress:.1f}%)", end="")

        if image_path.is_file():
            try:
                with Image.open(image_path) as img:
                    size = img.size
                    size_counter[size] += 1
                processed_count += 1
            except Exception:
                error_count += 1
        elif image_path.is_dir():
            skipped_dirs += 1
        else:
            error_count += 1

    print("\n処理完了。")
    print("-" * 30)
    print(f"処理した画像ファイル数: {processed_count}")
    if skipped_dirs > 0:
        print(f"スキップしたディレクトリ数: {skipped_dirs}")
    if error_count > 0:
        print(f"読み込みエラー/スキップしたファイル数: {error_count}")
    print(f"ユニークな画像サイズの数: {len(size_counter)}")
    print("-" * 30)

    return size_counter


def display_results_table(size_counter: Counter):
    """
    分析結果を表形式で表示する関数。
    (内容は変更なし)
    """
    if not size_counter:
        print("表示するデータがありません。")
        return

    sorted_sizes = sorted(size_counter.items(), key=lambda item: item[1], reverse=True)
    print("\n--- 画像サイズ分布 (表形式 - カウント数順) ---")

    if use_pandas:
        df_data = [{"Width": w, "Height": h, "Count": count} for (w, h), count in sorted_sizes]
        df = pd.DataFrame(df_data)
        df["Size (WxH)"] = df.apply(lambda row: f"{row['Width']}x{row['Height']}", axis=1)
        df_display = df[["Size (WxH)", "Width", "Height", "Count"]].set_index("Size (WxH)")

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df_display)
        total_count = df["Count"].sum()
        print("-" * (len(df_display.columns[0]) + 30))  # 区切り線調整
        print(f"合計画像数: {total_count}")
    else:
        # (Pandasなしの表形式表示コード - 前回のものと同様)
        if not sorted_sizes:
            print("データがありません。")
            return
        max_size_len = max(len(f"{w}x{h}") for (w, h), count in sorted_sizes) if sorted_sizes else 10
        max_count_len = max(len(str(count)) for size, count in sorted_sizes) if sorted_sizes else 5
        header_size = "サイズ (WxH)".ljust(max_size_len)
        header_count = "カウント".rjust(max_count_len)
        separator = f"+-{'-' * max_size_len}-+-{'-' * max_count_len}-+"
        print(separator)
        print(f"| {header_size} | {header_count} |")
        print(separator)
        total_count = 0
        for (width, height), count in sorted_sizes:
            size_str = f"{width}x{height}".ljust(max_size_len)
            count_str = str(count).rjust(max_count_len)
            print(f"| {size_str} | {count_str} |")
            total_count += count
        print(separator)
        print(f"合計画像数: {total_count}")


def plot_size_scatter(size_counter: Counter, save_path: str = None, size_scale_factor: float = 5.0):
    """
    分析結果を (幅, 高さ) の散布図で表示する関数。
    マーカーのサイズと色でカウント数を表現する。

    Args:
        size_counter (collections.Counter): analyze_image_sizesの結果。
        save_path (str, optional): グラフを保存するファイルパス。指定しない場合は表示のみ。
        size_scale_factor (float): マーカーサイズの調整係数。大きいほどマーカーが大きくなる。
    """
    if not use_matplotlib:
        print("Matplotlib/Numpyが利用できないため、グラフは表示されません。")
        return

    if not size_counter:
        print("グラフ表示するデータがありません。")
        return

    # データを準備: (width, height), count のペアをリストに
    size_data = list(size_counter.items())

    if not size_data:
        print("プロットするデータがありません。")
        return

    widths = [size[0][0] for size in size_data]
    heights = [size[0][1] for size in size_data]
    counts = np.array([size[1] for size in size_data])  # カウント数をNumpy配列に

    # マーカーサイズをスケーリング
    # カウント数の平方根に比例させ、係数で調整する。
    # plt.scatter の s はポイントの半径の二乗 (area) に比例するような値。
    # sqrtを使うことで、カウント数の差が大きい場合にサイズ差が緩和される。
    scaled_marker_sizes = np.sqrt(counts) * size_scale_factor

    print("\n--- 画像サイズ分布 (散布図: 幅 vs 高さ) ---")
    print(f"マーカーサイズ係数: {size_scale_factor} (調整は --marker_scale オプションで)")

    # グラフの描画
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 8))  # プロットエリアのサイズ

    # 日本語フォント設定
    try:
        if sys.platform == "darwin":
            plt.rcParams["font.family"] = "Hiragino Sans"
        elif os.name == "nt":
            plt.rcParams["font.family"] = "Meiryo"  # または 'MS Gothic' など
        else:
            # Linuxの場合、インストールされている日本語フォントを指定
            # 例: plt.rcParams['font.family'] = 'IPAPGothic'
            plt.rcParams["font.family"] = "../assets/font/fonts-japanese-gothic.ttf"  # フォールバック
            print("警告: 日本語フォントが正しく設定されていない可能性があります。", file=sys.stderr)
    except Exception as e:
        print(f"警告: 日本語フォントの設定中にエラー: {e}", file=sys.stderr)
        plt.rcParams["font.family"] = "assets/font/fonts-japanese-gothic.ttf"

    # 散布図を作成
    # x軸: 幅, y軸: 高さ
    # c: マーカーの色をカウント数で変化させる
    # s: マーカーのサイズをスケーリングしたカウント数で変化させる
    # cmap: 色の変化のパターン (カラーマップ)
    # alpha: マーカーの透明度 (点が重なった場合に見やすくするため)
    # edgecolors: マーカーの枠線の色
    scatter = ax.scatter(
        widths, heights, s=scaled_marker_sizes, c=counts, cmap="viridis", alpha=0.7, edgecolors="k", linewidth=0.5
    )

    # カラーバーを追加 (マーカーの色がどのカウント数に対応するかを示す凡例)
    fig.colorbar(scatter, label="カウント数")

    # 軸ラベル、タイトル設定
    ax.set_xlabel("幅 (Width)", fontsize=12)
    ax.set_ylabel("高さ (Height)", fontsize=12)
    ax.set_title("画像サイズの分布 (幅 vs 高さ)", fontsize=14)

    # グリッド表示
    ax.grid(True, linestyle="--", alpha=0.6)

    # 軸の範囲をデータに基づいて自動調整し、少し余裕を持たせる
    if widths and heights:
        # x軸、y軸ともに0から開始すると分かりやすい場合が多い
        ax.set_xlim(left=0, right=max(widths) * 1.1 if widths else 10)
        ax.set_ylim(bottom=0, top=max(heights) * 1.1 if heights else 10)

    # 軸の目盛りを整数にする
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=5))

    # レイアウトが重ならないように調整
    plt.tight_layout()

    # グラフの保存または表示
    if save_path:
        try:
            # bbox_inches='tight' でラベルなどがはみ出ないように調整
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"グラフを {save_path} に保存しました。")
        except Exception as e:
            print(f"エラー: グラフの保存に失敗しました: {e}", file=sys.stderr)
        plt.close(fig)  # 保存したら表示ウィンドウは閉じる
    else:
        print("グラフを表示します... (ウィンドウを閉じて続行)")
        plt.show()  # 画面に表示


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたディレクトリ構造内の画像ファイルのサイズ分布を集計し、表形式およびグラフで出力します。\n"
        "ディレクトリ構造: ROOT_DIR/{BookId}/characters/{Unicode}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../data/raw/dataset",  # デフォルト値を適切に設定してください
        help="分析対象のルートディレクトリ (デフォルト: data/raw/dataset)",
    )
    # 棒グラフ用の --top_n は削除
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="グラフを指定したファイルパスに画像として保存します (例: size_scatter.png)。指定しない場合は画面に表示します。",
    )
    parser.add_argument("--no_plot", action="store_true", help="グラフを表示または保存しません。")
    parser.add_argument(
        "--marker_scale",  # マーカーサイズ調整用引数を追加
        type=float,
        default=5.0,  # この値はデータによって調整が必要
        help="散布図のマーカーサイズのスケール係数。大きいほどマーカーが大きくなります (デフォルト: 5.0)",
    )

    # Jupyter環境などでの未知の引数を無視する
    args, unknown = parser.parse_known_args()

    root_dir_path = Path(args.root_dir)

    if not root_dir_path.is_dir():
        print(f"エラー: 指定されたルートディレクトリが見つかりません: {root_dir_path.resolve()}", file=sys.stderr)
        print("正しいパスを --root_dir で指定するか、スクリプト内のデフォルト値を変更してください。", file=sys.stderr)
        sys.exit(1)  # エラー終了

    print(f"分析を開始します。ルートディレクトリ: {root_dir_path.resolve()}")
    results = analyze_image_sizes(root_dir_path)

    if results is not None:
        # 表形式で結果を表示
        display_results_table(results)

        # 散布図で結果を表示 (matplotlibが利用可能で --no_plot が指定されていない場合)
        if use_matplotlib and not args.no_plot:
            # plot_size_scatter 関数を呼び出すように変更
            plot_size_scatter(results, save_path=args.save_plot, size_scale_factor=args.marker_scale)
        elif args.no_plot:
            print("\n--no_plot オプションが指定されたため、グラフは生成されません。")
        # Matplotlibがない場合は、plot_size_scatter 内で警告が表示される

    else:
        print("分析中にエラーが発生したか、対象ファイルが見つかりませんでした。", file=sys.stderr)
        # sys.exit(1) # Jupyter環境などではカーネルが停止するためコメントアウト推奨

    print("\nスクリプトの実行が完了しました。")
# %%
