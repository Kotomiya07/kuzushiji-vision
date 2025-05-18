from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- 設定項目 ---
INPUT_ROOT_DIR: Path = Path("data/raw/dataset")
OUTPUT_ROOT_DIR: Path = Path("data/onechannel")

# 最終的な出力画像の目標サイズ
FINAL_TARGET_SIZE: tuple[int, int] = (128, 128)

# 2値化処理を行う前の一時的なキャンバスのサイズ
TEMP_CANVAS_SIZE: tuple[int, int] = (192, 192)

# 元画像を一時キャンバスに配置する際の、元画像のコンテンツ部分の目標サイズ
# (このサイズにアスペクト比を維持してリサイズされ、TEMP_CANVAS_SIZEの中央に配置される)
CONTENT_AREA_IN_TEMP_SIZE: tuple[int, int] = (120, 120)

# 元画像の平均色を背景として使用する際に、明るくする度合い (BGR各値に加算)
# 0を指定すると明るさ調整なし。マイナスも可能だが、通常は正の値を想定。
BACKGROUND_BRIGHTNESS_ADJUSTMENT: int = 20  # 少し明るくする
# ----------------


def process_image(
    input_path: Path,
    output_path: Path,
    final_target_size: tuple[int, int],
    temp_canvas_size: tuple[int, int],
    content_area_in_temp_size: tuple[int, int],
    bg_brightness_adjustment: int,
) -> str:
    """
    個々の画像を処理し、指定された形式で保存する関数。
    一時的な大きなキャンバスと調整された背景色を使用して2値化を行う。

    Args:
        input_path (Path): 入力画像のパス。
        output_path (Path): 出力画像のパス。
        final_target_size (tuple): 最終的な出力画像の目標サイズ (width, height)。
        temp_canvas_size (tuple): 2値化処理を行う一時的なキャンバスのサイズ (width, height)。
        content_area_in_temp_size (tuple): 元画像を一時キャンバスに配置する際のコンテンツ目標サイズ (width, height)。
        bg_brightness_adjustment (int): 背景色の明るさ調整値。

    Returns:
        str: 処理結果またはエラーメッセージ。
    """
    try:
        # 画像を読み込み
        pic1_original: np.ndarray | None = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if pic1_original is None:
            return f"Error: Could not read image {input_path}. File may be missing, corrupted, or of an unsupported format."

        # 1. 元画像の平均色を計算し、明るさを調整
        mean_val = cv2.mean(pic1_original)
        # mean_val は (B, G, R, Alpha) のタプル。Alphaがない場合は0。
        adjusted_mean_b: int = np.clip(int(mean_val[0]) + bg_brightness_adjustment, 0, 255)
        adjusted_mean_g: int = np.clip(int(mean_val[1]) + bg_brightness_adjustment, 0, 255)
        adjusted_mean_r: int = np.clip(int(mean_val[2]) + bg_brightness_adjustment, 0, 255)
        adjusted_background_color: tuple[int, int, int] = (adjusted_mean_b, adjusted_mean_g, adjusted_mean_r)

        # 2. 元画像を content_area_in_temp_size にアスペクト比を維持してリサイズ
        h_orig, w_orig = pic1_original.shape[:2]
        target_content_w, target_content_h = content_area_in_temp_size

        if target_content_w <= 0 or target_content_h <= 0:
            return f"Error: content_area_in_temp_size {content_area_in_temp_size} must have positive dimensions."

        scale_h: float = target_content_h / h_orig
        scale_w: float = target_content_w / w_orig
        scale: float = min(scale_h, scale_w)

        resized_content_w: int = int(w_orig * scale)
        resized_content_h: int = int(h_orig * scale)

        if resized_content_w == 0 or resized_content_h == 0:
            return (
                f"Error: Resized content dimension became zero for {input_path}. "
                f"Original: ({w_orig},{h_orig}), TargetContent: {content_area_in_temp_size}, Scale: {scale:.4f}"
            )

        pic1_resized_content: np.ndarray = cv2.resize(
            pic1_original, (resized_content_w, resized_content_h), interpolation=cv2.INTER_AREA
        )

        # 3. 一時的な大きなキャンバスを調整後の背景色で作成
        temp_canvas_w, temp_canvas_h = temp_canvas_size
        if temp_canvas_w < resized_content_w or temp_canvas_h < resized_content_h:
            return (
                f"Error: temp_canvas_size {temp_canvas_size} is smaller than resized content "
                f"({resized_content_w},{resized_content_h}) for {input_path}."
            )

        temp_canvas: np.ndarray = np.full((temp_canvas_h, temp_canvas_w, 3), adjusted_background_color, dtype=np.uint8)

        # 4. リサイズされたコンテンツを一時キャンバスの中央に配置
        paste_x_start: int = (temp_canvas_w - resized_content_w) // 2
        paste_y_start: int = (temp_canvas_h - resized_content_h) // 2

        temp_canvas[
            paste_y_start : paste_y_start + resized_content_h, paste_x_start : paste_x_start + resized_content_w, :
        ] = pic1_resized_content

        # 5. 一時キャンバスをグレースケールに変換
        temp_canvas_gray: np.ndarray = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY)

        # 6. 大津の2値化を一時キャンバスグレースケール画像に対して行う
        _, img_otsu_on_temp = cv2.threshold(temp_canvas_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. 平均輝度値が127.5より大きい場合、画像を反転 (文字を黒、背景を白に近づけるため)
        if cv2.mean(img_otsu_on_temp)[0] > 127.5:
            img_otsu_on_temp = cv2.bitwise_not(img_otsu_on_temp)

        # 8. 2値化された一時キャンバスを最終的な目標サイズにリサイズ
        # 2値画像なので、補間はニアレストネイバーが良い
        # final_image_binary: np.ndarray = cv2.resize(img_otsu_on_temp, final_target_size, interpolation=cv2.INTER_NEAREST)
        # 中央を切り出す
        h_final, w_final = final_target_size
        h_start = (temp_canvas_h - h_final) // 2
        w_start = (temp_canvas_w - w_final) // 2
        final_image_binary = img_otsu_on_temp[h_start : h_start + h_final, w_start : w_start + w_final]

        # 出力ディレクトリが存在しない場合は作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), final_image_binary)
        return f"Processed: {input_path} -> {output_path}"

    except cv2.error as e:
        return f"OpenCV Error processing {input_path}: {e.err} (func: {e.func}, line: {e.lineno})"
    except Exception as e:
        return f"Error processing {input_path}: {type(e).__name__} - {str(e)}"


def find_image_files_and_prepare_args(
    input_root: Path,
    output_root: Path,
    final_target_size: tuple[int, int],
    temp_canvas_size: tuple[int, int],
    content_area_in_temp_size: tuple[int, int],
    bg_brightness_adjustment: int,
) -> list[tuple[Path, Path, tuple[int, int], tuple[int, int], tuple[int, int], int]]:
    """
    処理対象の画像ファイルを見つけ、process_image関数用の引数リストを作成する。
    """
    args_list: list[tuple[Path, Path, tuple[int, int], tuple[int, int], tuple[int, int], int]] = []

    for book_dir in input_root.iterdir():
        if not book_dir.is_dir():
            continue
        book_id: str = book_dir.name
        characters_dir: Path = book_dir / "characters"
        if not characters_dir.is_dir():
            continue

        for unicode_dir in characters_dir.iterdir():
            if not unicode_dir.is_dir():
                continue
            unicode_val: str = unicode_dir.name

            for img_file_path in unicode_dir.glob("*.jpg"):
                original_filename_stem: str = img_file_path.stem
                output_unicode_dir: Path = output_root / unicode_val
                output_filename: str = f"{book_id}-{unicode_val}-{original_filename_stem}.jpg"
                output_path: Path = output_unicode_dir / output_filename
                args_list.append(
                    (
                        img_file_path,
                        output_path,
                        final_target_size,
                        temp_canvas_size,
                        content_area_in_temp_size,
                        bg_brightness_adjustment,
                    )
                )
    return args_list


def main() -> None:
    """
    メイン処理関数。
    """
    print("データ前処理を開始します...")
    print(f"入力ルートディレクトリ: {INPUT_ROOT_DIR}")
    print(f"出力ルートディレクトリ: {OUTPUT_ROOT_DIR}")
    print(f"最終目標画像サイズ: {FINAL_TARGET_SIZE}")
    print(f"一時キャンバスサイズ: {TEMP_CANVAS_SIZE}")
    print(f"一時キャンバス内コンテンツ目標サイズ: {CONTENT_AREA_IN_TEMP_SIZE}")
    print(f"背景輝度調整値: {BACKGROUND_BRIGHTNESS_ADJUSTMENT}")

    tasks_args = find_image_files_and_prepare_args(
        INPUT_ROOT_DIR,
        OUTPUT_ROOT_DIR,
        FINAL_TARGET_SIZE,
        TEMP_CANVAS_SIZE,
        CONTENT_AREA_IN_TEMP_SIZE,
        BACKGROUND_BRIGHTNESS_ADJUSTMENT,
    )

    if not tasks_args:
        print("処理対象の画像が見つかりませんでした。入力ディレクトリ構造を確認してください。")
        print(f"期待する構造: {INPUT_ROOT_DIR}/{{BookId}}/characters/{{Unicode}}/*.jpg")
        return

    print(f"合計 {len(tasks_args)} 個の画像を処理します。")

    results: list[str] = []
    for args in tqdm(tasks_args, desc="画像処理中"):
        result: str = process_image(*args)
        results.append(result)

    processed_count: int = sum(1 for r in results if "Processed" in r)
    error_count: int = sum(1 for r in results if "Error" in r)

    print("\nデータ前処理が完了しました。")
    print(f"処理成功: {processed_count} 件")
    if error_count > 0:
        print(f"エラー: {error_count} 件")
        print("エラーが発生したファイルとエラー内容:")
        for r in results:
            if "Error" in r:
                print(f"  - {r}")
    print(f"処理済み画像は {OUTPUT_ROOT_DIR} に保存されています。")


if __name__ == "__main__":
    main()
