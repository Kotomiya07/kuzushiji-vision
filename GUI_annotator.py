import ast  # 文字列で保存されたリストを評価するため
import os
import re
import shutil
import tkinter as tk
import traceback  # エラー追跡用に追加
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageTk

# from skimage.transform import resize  # Pillowだけだと高品質なリサイズが少し面倒なため (不要になったためコメントアウト)


def unicode_to_char(unicode_str):
    """
    'U+XXXX' 形式のUnicodeコードポイント文字列を文字に変換します。

    Args:
      unicode_str: 'U+XXXX' 形式の文字列 (例: 'U+4E00')。

    Returns:
      対応する文字。変換できない場合は None を返します。
    """
    # 'U+'で始まり、その後に16進数が続く形式かチェック
    if not isinstance(unicode_str, str) or not re.match(r"^U\+[0-9A-Fa-f]+$", unicode_str):
        # print(f"エラー: 不正な形式です。'U+XXXX' の形式で入力してください。入力値: {unicode_str}") # GUI上はメッセージボックスで通知
        return None

    try:
        # 'U+' プレフィックスを除去
        hex_code = unicode_str[2:]
        # 16進数文字列を整数（コードポイント）に変換
        code_point = int(hex_code, 16)
        # コードポイントを文字に変換
        return chr(code_point)
    except ValueError:
        # int() や chr() でエラーが発生した場合 (例: 無効な16進数、範囲外のコードポイント)
        # print(f"エラー: コードポイントの変換に失敗しました。入力値: {unicode_str}")
        return None
    except Exception as e:
        # その他の予期せぬエラー
        print(f"予期せぬエラーが発生しました: {e}")
        return None


# 追加: 文字からU+XXXX形式への変換
def char_to_unicode_str(char):
    """
    文字を 'U+XXXX' 形式のUnicodeコードポイント文字列に変換します。

    Args:
      char: 変換する単一の文字。

    Returns:
      'U+XXXX' 形式の文字列。入力が無効な場合は None を返します。
    """
    if isinstance(char, str) and len(char) == 1:
        return f"U+{ord(char):04X}"
    else:
        return None


class ImageCanvas(tk.Canvas):
    """画像表示とインタラクション用Canvas"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.scale = 1.0
        self.original_pil_image = None  # ズーム前のオリジナル画像
        self.boxes = []  # 表示するボックスの情報 (tag, x1, y1, x2, y2, color, text)
        self.selected_box_tags = set()
        self.box_id_map = {}  # canvas_id -> box_tag

        # ズーム/パン用
        self.bind("<ButtonPress-2>", self.on_button_press)  # 中ボタンクリック
        self.bind("<B2-Motion>", self.on_move_press)  # 中ボタンドラッグ
        self.bind("<MouseWheel>", self.on_mouse_wheel)  # ホイールでズーム
        self.bind("<ButtonPress-1>", self.on_left_click)  # 左クリックで選択

    def load_image(self, image_path):
        # print(f"DEBUG: ImageCanvas.load_image called with path: {image_path}")  # DEBUG
        img = None  # Initialize img to None
        try:
            self.image_path = Path(image_path)
            # print(f"DEBUG: Opening image: {self.image_path}")  # DEBUG
            img = Image.open(self.image_path)
            # print(f"DEBUG: Image opened successfully. Type: {type(img)}")  # DEBUG
            # print(f"DEBUG: Converting image to RGBA...")  # DEBUG
            self.original_pil_image = img.convert("RGBA")  # RGBAで透明度対応
            # print(f"DEBUG: Image converted to RGBA successfully.")  # DEBUG

            # --- 初期スケール計算 ---
            self.update_idletasks()  # Canvasサイズを確定させる
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                try:
                    # 親ウィジェットのサイズを使う試み
                    parent_width = self.master.winfo_width()
                    parent_height = self.master.winfo_height()
                    if parent_width > 1 and parent_height > 1:
                        canvas_width = parent_width
                        canvas_height = parent_height
                        # print(f"DEBUG: Using parent size for initial scale calculation: {canvas_width}x{canvas_height}")
                    else:
                        # print("DEBUG: Canvas and parent size not available, using default size 600x600")
                        canvas_width = 600  # 仮のデフォルト値
                        canvas_height = 600  # 仮のデフォルト値
                except Exception:
                    # print("DEBUG: Error getting parent size, using default size 600x600")
                    canvas_width = 600  # 仮のデフォルト値
                    canvas_height = 600  # 仮のデフォルト値

            img_width, img_height = self.original_pil_image.size
            if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
                scale_w = canvas_width / img_width
                scale_h = canvas_height / img_height
                initial_scale = min(scale_w, scale_h, 1.0)  # 1.0 を超える拡大はしない
                # print(
                #    f"DEBUG: Calculated initial scale: {initial_scale} (canvas: {canvas_width}x{canvas_height}, image: {img_width}x{img_height})"
                # )
            else:
                initial_scale = 1.0  # 画像サイズ不正 or Canvasサイズ取れない場合

            self.scale = initial_scale  # 計算した初期スケールを設定
            # --- 初期スケール計算ここまで ---

            self.boxes = []
            self.selected_box_tags = set()
            self.box_id_map = {}
            # print(f"DEBUG: Calling display_image with initial scale {self.scale}...")  # DEBUG
            self.display_image()
            # print(f"DEBUG: display_image finished.")  # DEBUG
        except FileNotFoundError:
            self.clear_canvas()
            self.create_text(
                self.winfo_width() // 2 if self.winfo_width() > 1 else 300,
                self.winfo_height() // 2 if self.winfo_height() > 1 else 300,
                text=f"画像が見つかりません:\n{image_path}",
                fill="red",
                anchor="center",
            )
            self.original_pil_image = None
            # print(f"DEBUG: FileNotFoundError in load_image for path: {image_path}")  # DEBUG
            print(f"Error: Image not found at {image_path}")
        except Exception as e:
            self.clear_canvas()
            self.create_text(
                self.winfo_width() // 2 if self.winfo_width() > 1 else 300,
                self.winfo_height() // 2 if self.winfo_height() > 1 else 300,
                text=f"画像読み込みエラー:\n{e}",
                fill="red",
                anchor="center",
            )
            self.original_pil_image = None
            # print(f"DEBUG: Exception in load_image for path: {image_path}.")  # DEBUG
            # print(f"DEBUG: Exception type: {type(e)}")  # DEBUG
            # print(f"DEBUG: Exception message: {e}")  # DEBUG
            # print(f"DEBUG: Traceback:\n{traceback.format_exc()}")  # DEBUG
            print(f"Error loading image {image_path}: {e}")
            traceback.print_exc()  # 詳細なエラー情報をコンソールに出力

    def display_image(self):
        # print(f"DEBUG: display_image started. self.original_pil_image is None: {self.original_pil_image is None}")  # DEBUG
        if self.original_pil_image is None:
            self.clear_canvas()
            # print(f"DEBUG: self.original_pil_image is None at the beginning of display_image. Clearing canvas.")  # DEBUG
            return

        # 古い描画要素 (画像、ボックス、テキスト) を削除
        self.delete("image")  # 古い画像も削除
        self.delete("box")
        self.delete("box_text")

        # スケールに合わせてリサイズ (高品質リサイズ)
        width = int(self.original_pil_image.width * self.scale)
        height = int(self.original_pil_image.height * self.scale)

        # 画像が大きすぎる場合の処理 (メモリとパフォーマンスのため)
        max_dim = 3000  # 必要に応じて調整
        if width > max_dim or height > max_dim:
            ratio = min(max_dim / width, max_dim / height)
            width = int(width * ratio)
            height = int(height * ratio)
            print(f"Warning: Resized large image to {width}x{height} for display.")

        if width <= 0 or height <= 0:  # ゼロまたは負のサイズは表示しない
            print(f"Warning: Invalid image dimensions after scaling ({width}x{height}). Cannot display.")
            return

        try:
            # Pillow の resize メソッド (BILINEAR) を使用してリサイズ
            self.pil_image = self.original_pil_image.resize((width, height), Image.Resampling.BILINEAR)

            try:
                # print(f"DEBUG: Calling ImageTk.PhotoImage with self.pil_image...")  # DEBUG
                self.tk_image = ImageTk.PhotoImage(self.pil_image)
                # print(f"DEBUG: ImageTk.PhotoImage created successfully.")  # DEBUG
            except Exception as tk_err:
                # print(f"DEBUG: Error during ImageTk.PhotoImage creation!")  # DEBUG
                # print(f"DEBUG: Error type: {type(tk_err)}")  # DEBUG
                # print(f"DEBUG: Error message: {tk_err}")  # DEBUG
                # print(f"DEBUG: Traceback:\n{traceback.format_exc()}")  # DEBUG
                # エラーが発生した場合、tk_image は None のままになるか、不正な状態になる可能性がある
                self.tk_image = None  # 明示的に None に設定して後続処理でのエラーを防ぐ
                raise tk_err  # エラーを再送出して、外側の except Exception で捕捉させる
            self.create_image(0, 0, anchor="nw", image=self.tk_image, tags="image")
            # print(
            #    f"DEBUG: self.pil_image created. Type: {type(self.pil_image)}, Mode: {self.pil_image.mode}, Size: {self.pil_image.size}"
            # )  # DEBUG
            self.config(scrollregion=self.bbox("all"))

            # ボックスの再描画
            self.redraw_boxes()

        except ValueError as e:
            # print(
            #    f"Error during image resize or display: {e}. Original size: {self.original_pil_image.size}, Target size: ({width}, {height})"
            # )
            self.clear_canvas()  # エラー時はクリア
            print(f"Error during image resize or display: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"Unexpected error during image display: {e}")
            traceback.print_exc()
            self.clear_canvas()

    def add_box(self, tag, x1, y1, x2, y2, color="red", width=2, text=None):
        """Canvas座標系でのボックス情報を追加"""
        self.boxes.append({"tag": tag, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "width": width, "text": text})
        # self.redraw_boxes() # 複数追加する場合は最後に1回描画する方が効率的

    def redraw_boxes(self):
        # 既存のボックスとテキストを削除
        self.delete("box")
        self.delete("box_text")
        self.box_id_map.clear()

        if not self.pil_image:
            return  # 画像がない場合は描画しない

        img_w, img_h = self.pil_image.size

        for box_info in self.boxes:
            try:
                # スケーリングされた画像上の座標に変換
                x1 = box_info["x1"] * self.scale
                y1 = box_info["y1"] * self.scale
                x2 = box_info["x2"] * self.scale
                y2 = box_info["y2"] * self.scale

                # 画面外にはみ出ないようにクリッピング (念のため)
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # 小さすぎるボックスは描画しないか、最低サイズを保証する
                min_size = 1
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    # continue # スキップする
                    pass  # 小さくても描画を試みる（選択できるように）

                color = box_info["color"]
                width = box_info["width"]
                tag = box_info["tag"]

                # 選択状態に応じて色や太さを変更
                if tag in self.selected_box_tags:
                    color = "red"
                    width = 3

                item_id = self.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=("box", tag))
                self.box_id_map[item_id] = tag  # Canvas IDとカスタムタグを紐付け

                if box_info["text"]:
                    # テキストサイズを調整（簡易的）
                    font_size = max(8, min(12, int((y2 - y1) * 0.8)))  # ボックスの高さに応じて調整

                    # tkinter 用のフォント指定 (タプル形式)
                    # 存在する可能性のあるフォントを試す
                    font_family = "Yu Gothic UI" if "Yu Gothic UI" in ImageFont.core.getfont(None).getname() else "Arial"
                    try:
                        tk_font = (font_family, font_size)
                    except tk.TclError:  # フォントが見つからない場合
                        tk_font = ("Arial", font_size)  # デフォルトフォント

                    self.create_text(
                        x2 + 5, y1 + 2, text=box_info["text"], anchor="nw", fill=color, tags=("box_text", tag), font=tk_font
                    )
            except Exception as e:
                print(f"Error drawing box or text for tag {box_info.get('tag', 'N/A')}: {e}")
                traceback.print_exc()  # デバッグ用に詳細表示

    def clear_canvas(self):
        self.delete("all")
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.original_pil_image = None
        self.boxes = []
        self.selected_box_tags = set()
        self.box_id_map = {}

    def on_button_press(self, event):
        self.scan_mark(event.x, event.y)

    def on_move_press(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event):
        # Windows/macOSでの挙動差を吸収
        delta = 0
        if event.num == 5 or event.delta == -120:  # 下スクロール（縮小）
            delta = -1
        elif event.num == 4 or event.delta == 120:  # 上スクロール（拡大）
            delta = 1
        # Linuxでのホイールイベント (delta がプラットフォームや設定依存の場合がある)
        elif event.delta < 0:
            delta = -1
        elif event.delta > 0:
            delta = 1

        if delta != 0:
            factor = 1.1**delta
            new_scale = self.scale * factor
            # スケールの制限
            min_scale = 0.05  # 最小スケールを少し小さく
            max_scale = 10.0  # 最大スケールを少し大きく
            if min_scale <= new_scale <= max_scale:
                # カーソル位置中心にズーム
                cursor_x = self.canvasx(event.x)
                cursor_y = self.canvasy(event.y)

                # 古いスケールでのCanvas座標
                old_canvas_x = cursor_x
                old_canvas_y = cursor_y

                # 新しいスケールを設定
                self.scale = new_scale
                self.display_image()  # ここで画像とボックスが再描画される

                # 再描画後の新しいスケールでの同じ画像位置に対応するCanvas座標
                new_canvas_x = old_canvas_x * factor
                new_canvas_y = old_canvas_y * factor

                # カーソル位置が画面上で動かないようにスクロール量を計算
                delta_x = event.x - new_canvas_x
                delta_y = event.y - new_canvas_y

                # Canvasをスクロールして調整
                self.scan_mark(0, 0)  # マーク位置をリセットしないと相対移動になる
                self.scan_dragto(int(delta_x), int(delta_y), gain=1)

    def on_left_click(self, event):
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)

        # Find all items with the 'box' tag that are currently visible
        all_box_items = self.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)
        clicked_box_tags = set()

        # Iterate through overlapping items to find boxes under the cursor
        for item_id in all_box_items:
            try:
                item_tags = self.gettags(item_id)
                if "box" in item_tags:  # Check if it's a box
                    # Find the custom tag (column path or char_index)
                    custom_tag = None
                    for t in item_tags:
                        if t != "box":
                            custom_tag = t
                            break
                    if custom_tag:
                        # Verify coordinates just in case find_overlapping is too broad
                        x1, y1, x2, y2 = self.coords(item_id)
                        if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                            clicked_box_tags.add(custom_tag)

            except Exception as e:
                # self.coords might fail if item is deleted concurrently? Unlikely but safe.
                print(f"Warning: Error getting coords or tags for item {item_id}: {e}")

        # Check if Ctrl key was pressed during the click
        # Modifiers for different OS (Windows/Linux: 0x4, macOS: 0x8 or 0x20000 for Command)
        ctrl_pressed = bool(event.state & (0x0004 | 0x0008 | 0x20000))

        if not clicked_box_tags:
            # Clicked outside any box
            if not ctrl_pressed:
                # If Ctrl is not pressed, clear selection
                self.selected_box_tags.clear()
            # If Ctrl is pressed, do nothing (keep existing selection)
        else:
            # Clicked inside one or more boxes (usually just one, unless overlapping)
            # For simplicity, handle the first one found if multiple overlap perfectly
            tag_to_process = list(clicked_box_tags)[0]  # Take one if multiple

            if ctrl_pressed:
                # Ctrl key is pressed: toggle selection state of the clicked box
                if tag_to_process in self.selected_box_tags:
                    self.selected_box_tags.remove(tag_to_process)  # Deselect if already selected
                else:
                    self.selected_box_tags.add(tag_to_process)  # Select if not selected
            else:
                # Ctrl key is not pressed:
                # If the click is on an already selected box, keep the current selection (allows dragging later)
                # If the click is on a non-selected box, select *only* the clicked box
                if tag_to_process not in self.selected_box_tags:
                    self.selected_box_tags = {tag_to_process}  # Select only the clicked one

        # Redraw boxes to reflect selection changes
        self.redraw_boxes()

        # Notify the main application about the click event
        # Find the top-level window master which should have the handler method
        master_app = self.winfo_toplevel()
        if hasattr(master_app, "on_canvas_click"):
            # Pass the canvas instance, the tags of boxes under the cursor, and the Ctrl key state
            master_app.on_canvas_click(self, clicked_box_tags, ctrl_pressed)  # Pass the set of clicked tags

    def get_selected_tags(self):
        return list(self.selected_box_tags)


class DataManager:
    """CSVデータの読み込み、管理、編集、保存を行う"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.processed_dir = self.base_path / "processed"
        self.column_images_base_dir = self.processed_dir / "column_images"  # data_preprocessing.pyに合わせる
        # self.input_csv_path = self.processed_dir / "column_info.csv" # 元のファイル名
        self.input_csv_path = self.processed_dir / "annotation.csv"  # 読み込み元ファイル
        self.output_csv_path = self.processed_dir / "annotation_edited.csv"  # ★変更: 保存先ファイル
        self.last_state_path = self.base_path / ".last_image_path.txt"  # 状態保存ファイル
        self.df = None
        self.original_images = []
        self.current_original_image = None
        self._last_loaded_image_path = self.load_last_state()  # 最後に表示した画像を読み込む
        self.load_data()
        self.changes_made = False  # 変更フラグ

    def load_last_state(self):
        """最後に表示していた画像のパスをファイルから読み込む"""
        if self.last_state_path.exists():
            try:
                return self.last_state_path.read_text().strip()
            except Exception as e:
                print(f"Warning: Failed to read last state file: {e}")
        return None

    def save_last_state(self):
        """現在の画像パスをファイルに保存する"""
        if self.current_original_image:
            try:
                self.last_state_path.write_text(self.current_original_image)
                # print(f"Saved last image path: {self.current_original_image}") # デバッグ用
            except Exception as e:
                print(f"Warning: Failed to save last state file: {e}")

    def load_data(self):
        # ★変更: 編集済みファイルがあればそちらを優先して読み込むか？
        #   今回はシンプルに、常に元の入力ファイルから読み込む仕様とする。
        #   もし編集済みファイルを読み込みたい場合は、input_csv_pathをoutput_csv_pathに切り替える。
        load_path = self.input_csv_path
        if not load_path.exists():
            # 編集済みファイルが存在し、元のファイルがない場合、編集済みファイルを読む試み
            if self.output_csv_path.exists():
                print(
                    f"Warning: Input CSV '{self.input_csv_path}' not found. Loading existing edited file '{self.output_csv_path}' instead."
                )
                load_path = self.output_csv_path
            else:
                raise FileNotFoundError(
                    f"必須CSVファイルが見つかりません: {self.input_csv_path} も {self.output_csv_path} も存在しません。"
                )

        print(f"Loading data from: {load_path}")
        try:
            self.df = pd.read_csv(load_path)
        except Exception as e:
            raise RuntimeError(f"CSVファイルの読み込みに失敗しました: {load_path}\nエラー: {e}")

        # データ型の修正 (CSV読み込みで文字列になるため)
        # box_in_original, char_boxes_in_column, unicode_ids はリスト形式のはず
        list_columns = ["box_in_original", "char_boxes_in_column", "unicode_ids"]
        for col in list_columns:
            if col in self.df.columns:
                # ast.literal_evalが安全だが、データ形式によってはエラーになる可能性あり
                try:
                    # 空文字列や NaN の場合の処理を追加
                    self.df[col] = self.df[col].apply(
                        lambda x: ast.literal_eval(x)
                        if isinstance(x, str) and x.strip() and x.lower() != "nan"
                        else ([] if pd.isna(x) else x)
                    )
                    # 評価後、リストであることを確認（念のため）
                    self.df[col] = self.df[col].apply(lambda x: x if isinstance(x, list) else [])
                except (ValueError, SyntaxError, TypeError) as e:
                    print(f"Warning: Could not parse column '{col}'. Check data format. Setting to empty list. Error: {e}")
                    # エラーの場合は空リストにする
                    self.df[col] = self.df[col].apply(lambda x: [])
            else:
                print(f"Warning: Expected list column '{col}' not found in CSV. Creating with empty lists.")
                self.df[col] = [[] for _ in range(len(self.df))]  # 列が存在しない場合は空リストで作成

        # 'column_image' のパスを絶対パスに (または base_path からの相対パスのまま扱う)
        # ここでは base_path からの相対パスとして扱う前提で進める
        # self.df['column_image_abs'] = self.df['column_image'].apply(lambda p: self.base_path / p)

        # processed/ がパスに含まれている場合、削除する (data_preprocessing.py との整合性のため)
        if "column_image" in self.df.columns:
            processed_dir_name = self.processed_dir.name  # "processed"

            def fix_path(p):
                if not isinstance(p, str):
                    return p  # 文字列でない場合はそのまま
                path_str = str(p).replace("\\", "/")
                # パスが "processed/" で始まっているかチェック
                prefix = processed_dir_name + "/"
                if path_str.startswith(prefix):
                    # 先頭の "processed/" を除去
                    return path_str[len(prefix) :]
                return path_str  # それ以外はそのまま

            self.df["column_image"] = self.df["column_image"].astype(str).apply(fix_path)
            # print("Applied workaround path correction to 'column_image' column.")
        else:
            print("Warning: 'column_image' column not found in CSV.")
            self.df["column_image"] = ""  # 列が存在しない場合は空文字列で作成

        # 元画像の一覧を取得
        if "original_image" in self.df.columns:
            # NaNや空文字列を除外してからuniqueを取得
            valid_orig_images = self.df["original_image"].dropna().astype(str).unique()
            self.original_images = sorted([img for img in valid_orig_images if img])  # 空文字列も除外
            if self.original_images:
                # 最後に表示していた画像が存在すればそれを、なければ最初の画像を設定
                if self._last_loaded_image_path and self._last_loaded_image_path in self.original_images:
                    self.current_original_image = self._last_loaded_image_path
                    print(f"Resuming from last viewed image: {self.current_original_image}")
                else:
                    self.current_original_image = self.original_images[0]
            else:
                self.current_original_image = None  # 画像リストが空の場合
        else:
            print("Warning: 'original_image' column not found in CSV.")
            self.original_images = []
            self.current_original_image = None

        print(f"Loaded {len(self.df)} columns for {len(self.original_images)} unique original images.")
        self.changes_made = False  # ロード直後は変更なし

    def get_original_images(self):
        return self.original_images

    def set_current_original_image(self, image_path):
        if image_path in self.original_images:
            self.current_original_image = image_path
            return True
        return False

    def get_columns_for_current_image(self):
        if self.current_original_image is None or self.df is None:
            return pd.DataFrame()  # 空のDataFrameを返す
        # original_image が一致する行をコピーして返す
        return self.df[self.df["original_image"] == self.current_original_image].copy()

    def get_column_data(self, column_image_path_relative):
        """相対パスを使って列データを取得"""
        if self.df is None or column_image_path_relative is None:
            return None
        # 完全一致で検索
        matches = self.df[self.df["column_image"] == column_image_path_relative]
        if not matches.empty:
            # 常に辞書形式で返す
            return matches.iloc[0].to_dict()
        else:
            # print(f"Warning: Column data not found for relative path: {column_image_path_relative}")
            return None

    def get_column_abs_path(self, column_image_path_relative):
        """列画像の絶対パスを取得"""
        if not column_image_path_relative:
            return None
        # column_info.csv 内のパスは processed ディレクトリからの相対パスになっている想定
        # 例: column_images/00001/00001_001/00001_001_column_001.jpg
        return self.processed_dir / column_image_path_relative

    def get_original_image_abs_path(self, original_image_path_str):
        """元画像の絶対パスを取得"""
        if not original_image_path_str or not isinstance(original_image_path_str, str):
            print(f"Warning: Invalid original_image path received: {original_image_path_str}")
            return None  # 不正な入力の場合は None を返す

        potential_path = Path(original_image_path_str)

        # 1. 絶対パスかどうかチェック
        if potential_path.is_absolute():
            if potential_path.exists():
                return potential_path
            else:
                # 絶対パスだが存在しない場合
                # print(f"Warning: Absolute path specified in CSV does not exist: {potential_path}")
                # 存在しないパスオブジェクトを返す (呼び出し元でチェック)
                # → ここで None を返した方が安全か？ いや、呼び出し元でチェックしているはず。
                return potential_path  # 存在しなくてもパスを返す

        # --- 相対パスの場合の解決ロジック ---
        # base_path (例: ユーザーが選択した 'data' ディレクトリ) の親をプロジェクトルートと仮定
        project_root = self.base_path.parent
        # if not project_root: # base_pathがルートディレクトリなどの場合
        #    project_root = Path.cwd() # カレントディレクトリをルートとする
        # ↑ base_path.parent が None になるケースは稀なので、 일단 そのまま使う

        # 2. プロジェクトルートからの相対パスとして試す (最有力候補)
        #    CSVの値が 'data/raw/dataset/...' の場合、プロジェクトルートに結合すれば正しいはず
        path_from_root = project_root / potential_path
        # print(f"DEBUG: Trying path from project root: {path_from_root}, Exists: {path_from_root.exists()}")  # DEBUG
        if path_from_root.exists():
            # #print(f"Debug: Resolved original image path (from project root): {path_from_root}")
            return path_from_root

        # 3. base_path (ユーザー選択の 'data' ディレクトリ) からの相対パスとして試す
        #    CSVの値が 'raw/dataset/...' のような形式の場合
        path_from_base = self.base_path / potential_path
        # print(f"DEBUG: Trying path from base path: {path_from_base}, Exists: {path_from_base.exists()}")  # DEBUG
        if path_from_base.exists():
            # #print(f"Debug: Resolved original image path (from base path): {path_from_base}")
            return path_from_base

        # 4. カレントワーキングディレクトリからの相対パスとして試す (最後の手段)
        path_from_cwd = Path.cwd() / potential_path
        # print(f"DEBUG: Trying path from CWD: {path_from_cwd}, Exists: {path_from_cwd.exists()}")  # DEBUG
        if path_from_cwd.exists():
            # #print(f"Debug: Resolved original image path (from CWD): {path_from_cwd}")
            return path_from_cwd

        # --- ここまでで見つからない場合 ---
        # 見つからない場合でも、最も可能性の高いパス (プロジェクトルートからのパス) を返すことにする
        # 呼び出し元で .exists() をチェックする必要がある
        # print(
        #     f"Warning: Could not determine absolute path for original image: '{original_image_path_str}'. \n"
        #     f"         Tried relative to project root ('{project_root}'), \n"
        #     f"         base path ('{self.base_path}'), \n"
        #     f"         and CWD ('{Path.cwd()}').\n"
        #     f"         Returning likely path: {path_from_root}"
        # )
        return path_from_root

    def _recalculate_column_bounds(self, char_boxes_in_orig):
        """元画像内の文字座標リストから列のバウンディングボックスを計算"""
        if (
            not char_boxes_in_orig
            or not isinstance(char_boxes_in_orig, list)
            or not all(isinstance(b, list) and len(b) == 4 for b in char_boxes_in_orig)
        ):
            return [0, 0, 0, 0]
        try:
            all_coords = np.array(char_boxes_in_orig)  # [[x1, y1, x2, y2], ...]
            if all_coords.size == 0:
                return [0, 0, 0, 0]
            x1 = np.min(all_coords[:, 0])
            y1 = np.min(all_coords[:, 1])
            x2 = np.max(all_coords[:, 2])
            y2 = np.max(all_coords[:, 3])
            return [int(x1), int(y1), int(x2), int(y2)]
        except Exception as e:
            print(f"Error calculating column bounds: {e}")
            traceback.print_exc()
            return [0, 0, 0, 0]

    def _get_char_boxes_in_original(self, column_data):
        """列データから元画像内の文字座標リストを取得"""
        if not column_data or "box_in_original" not in column_data or "char_boxes_in_column" not in column_data:
            return []

        col_box_orig = column_data["box_in_original"]  # これは切り抜き座標 (マージン込み)
        char_boxes_in_col = column_data["char_boxes_in_column"]

        # データ形式チェック
        if not isinstance(col_box_orig, list) or len(col_box_orig) != 4:
            print(f"Warning: Invalid format for 'box_in_original': {col_box_orig}")
            return []
        if not isinstance(char_boxes_in_col, list):
            print(f"Warning: Invalid format for 'char_boxes_in_column': {char_boxes_in_col}")
            return []

        col_crop_x1, col_crop_y1, _, _ = col_box_orig
        char_boxes_in_orig = []
        for i, box in enumerate(char_boxes_in_col):
            if isinstance(box, list) and len(box) == 4:
                cx1, cy1, cx2, cy2 = box
                # 元画像座標系に変換
                orig_x1 = cx1 + col_crop_x1
                orig_y1 = cy1 + col_crop_y1
                orig_x2 = cx2 + col_crop_x1
                orig_y2 = cy2 + col_crop_y1
                char_boxes_in_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
            else:
                print(f"Warning: Invalid format for char box at index {i}: {box}")
                # 不正なボックスはスキップするか、エラー処理が必要

        return char_boxes_in_orig

    # --- ★ エラーハンドリング強化とデータ保護 ---
    # 各編集メソッド (merge, split, move, delete, add) で共通して
    # 1. 処理開始前に必要なデータが揃っているか、形式が正しいかチェック
    # 2. 画像読み込み、計算、画像保存などの各ステップでエラーが発生しないか try-except で保護
    # 3. エラーが発生した場合、メッセージを表示し、False を返して操作を中断
    # 4. DataFrame の変更 (drop, concat) と古いファイルの削除は、すべての処理が成功した後に行う
    # 5. 中間ファイル（新しく生成した列画像など）は、操作が失敗した場合に削除する

    def merge_columns(self, column_paths_to_merge):
        """複数の列を結合する"""
        if len(column_paths_to_merge) < 2:
            messagebox.showerror("エラー", "結合するには少なくとも2つの列を選択してください。")
            return False

        columns_data = []
        original_image_path = None
        temp_new_col_abs_path = None  # エラー時削除用

        try:
            # 1. データ取得と検証
            for path_rel in column_paths_to_merge:
                data = self.get_column_data(path_rel)
                if not data:
                    raise ValueError(f"列データが見つかりません: {path_rel}")
                columns_data.append(data)
                current_orig_img = data.get("original_image")
                if not current_orig_img:
                    raise ValueError(f"列データに元画像パスが含まれていません: {path_rel}")
                if original_image_path is None:
                    original_image_path = current_orig_img
                elif original_image_path != current_orig_img:
                    raise ValueError("異なる元画像の列は結合できません。")

            # 2. 新しい列情報の計算
            all_char_boxes_in_orig = []
            all_unicode_ids = []
            for data in columns_data:
                char_boxes_orig = self._get_char_boxes_in_original(data)
                if not char_boxes_orig and data.get("unicode_ids"):  # 座標がないがIDはある場合
                    print(f"Warning: Column {data['column_image']} has unicode IDs but failed to get original coordinates.")
                    # ここでエラーにするか、無視するか？ 一旦無視して進める
                    # raise ValueError(f"列 {data['column_image']} の文字座標を取得できませんでした。")
                all_char_boxes_in_orig.extend(char_boxes_orig)
                ids = data.get("unicode_ids", [])
                if not isinstance(ids, list):
                    raise ValueError(f"列 {data['column_image']} の unicode_ids がリストではありません。")
                all_unicode_ids.extend(ids)

            if not all_char_boxes_in_orig or not all_unicode_ids:
                raise ValueError("結合対象の列に有効な文字情報が含まれていません。")
            if len(all_char_boxes_in_orig) != len(all_unicode_ids):
                raise ValueError("結合中に文字座標とUnicode IDの数が一致しませんでした。")

            # 文字をY座標でソート (元画像座標基準)
            if len(all_char_boxes_in_orig) > 1:
                sorted_indices = np.argsort([box[1] for box in all_char_boxes_in_orig])
                all_char_boxes_in_orig = [all_char_boxes_in_orig[i] for i in sorted_indices]
                all_unicode_ids = [all_unicode_ids[i] for i in sorted_indices]

            # 3. 新しい列画像の生成と保存 (_recreate_column_from_chars を利用)
            base_rel_path_str = column_paths_to_merge[0]  # 最初の列のパスを基準にする
            new_row_data, new_column_rel_path = self._recreate_column_from_chars(
                all_char_boxes_in_orig, all_unicode_ids, original_image_path, base_rel_path_str, "_merged"
            )

            if new_row_data is None or new_column_rel_path is None:
                # _recreate_column_from_chars 内でエラー発生済みのはず
                # ここで追加のエラーメッセージは不要かもしれない
                raise RuntimeError("結合後の新しい列の生成に失敗しました。")

            temp_new_col_abs_path = self.processed_dir / new_column_rel_path  # 正常に生成された場合、後で使う

            # 4. DataFrameの更新 (エラーが発生しなければここまできる)
            # 元の列の行を削除するインデックスを取得
            indices_to_drop = self.df[self.df["column_image"].isin(column_paths_to_merge)].index
            # 新しい行のDataFrameを作成
            new_row_df = pd.DataFrame([new_row_data])

            # --- ここからアトミックな操作 (DataFrame更新とファイル削除) ---
            # DataFrame更新
            self.df = self.df.drop(indices_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

            # 元の列画像ファイルを削除
            for path_rel in column_paths_to_merge:
                try:
                    abs_path_to_delete = self.get_column_abs_path(path_rel)
                    if abs_path_to_delete and abs_path_to_delete.exists():
                        os.remove(abs_path_to_delete)
                        # print(f"Deleted old column image: {abs_path_to_delete}")
                except Exception as e:
                    print(f"Warning: Could not delete old column image {path_rel}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Columns merged into: {new_column_rel_path}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("結合エラー", f"列の結合中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: 作成された可能性のある新しい列画像を削除
            if temp_new_col_abs_path and temp_new_col_abs_path.exists():
                try:
                    os.remove(temp_new_col_abs_path)
                    print(f"Cleaned up temporary merged file: {temp_new_col_abs_path}")
                except Exception as clean_e:
                    print(f"Error during cleanup of {temp_new_col_abs_path}: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("結合エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_col_abs_path and temp_new_col_abs_path.exists():
                try:
                    os.remove(temp_new_col_abs_path)
                    print(f"Cleaned up temporary merged file: {temp_new_col_abs_path}")
                except Exception as clean_e:
                    print(f"Error during cleanup of {temp_new_col_abs_path}: {clean_e}")
            return False

    def split_column(self, column_path_to_split, split_index):
        """指定した文字インデックスの前で列を分割する"""
        temp_new_path1 = None
        temp_new_path2 = None

        try:
            # 1. データ取得と検証
            column_data = self.get_column_data(column_path_to_split)
            if not column_data:
                raise ValueError(f"列データが見つかりません: {column_path_to_split}")

            char_boxes_col = column_data.get("char_boxes_in_column", [])
            unicode_ids = column_data.get("unicode_ids", [])
            original_image_path = column_data.get("original_image")
            col_box_orig_crop = column_data.get("box_in_original")

            if not original_image_path:
                raise ValueError("元画像パスがありません。")
            if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list):
                raise ValueError("文字ボックスまたはUnicode IDの形式が不正です。")
            if len(char_boxes_col) != len(unicode_ids):
                raise ValueError("文字ボックスとUnicode IDの数が一致しません。")
            if not isinstance(col_box_orig_crop, list) or len(col_box_orig_crop) != 4:
                raise ValueError("列の切り抜き座標(box_in_original)が不正です。")

            if not 0 < split_index < len(unicode_ids):
                raise ValueError("無効な分割位置です。")

            # 2. 元画像上の絶対座標を取得
            char_boxes_orig = self._get_char_boxes_in_original(column_data)
            if len(char_boxes_orig) != len(unicode_ids):  # _get_char_boxes_in_original内でエラーがあった場合
                raise RuntimeError("文字の元画像座標の取得に失敗しました。")

            # 3. 分割
            chars_orig1 = char_boxes_orig[:split_index]
            ids1 = unicode_ids[:split_index]
            chars_orig2 = char_boxes_orig[split_index:]
            ids2 = unicode_ids[split_index:]

            if not chars_orig1 or not chars_orig2:
                raise ValueError("分割後の列のどちらかが空になります。")

            # 4. 新しい列を生成 (2つ)
            new_data1, new_path1 = self._recreate_column_from_chars(
                chars_orig1, ids1, original_image_path, column_path_to_split, "_splitA"
            )
            if new_data1 is None:
                raise RuntimeError("分割後の列1の生成に失敗しました。")
            temp_new_path1 = self.processed_dir / new_path1  # クリーンアップ用

            new_data2, new_path2 = self._recreate_column_from_chars(
                chars_orig2, ids2, original_image_path, column_path_to_split, "_splitB"
            )
            if new_data2 is None:
                raise RuntimeError("分割後の列2の生成に失敗しました。")
            temp_new_path2 = self.processed_dir / new_path2  # クリーンアップ用

            # 5. DataFrameの更新 (エラーが発生しなければここまできる)
            index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
            new_rows_df = pd.DataFrame([new_data1, new_data2])

            # --- アトミックな操作 ---
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

            # 元の列画像ファイルを削除
            try:
                abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
                if abs_path_to_delete and abs_path_to_delete.exists():
                    os.remove(abs_path_to_delete)
                    # print(f"Deleted old column image: {abs_path_to_delete}")
            except Exception as e:
                print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Column {column_path_to_split} split into {new_path1} and {new_path2}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("分割エラー", f"列の分割中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: 作成された中間ファイルを削除
            if temp_new_path1 and temp_new_path1.exists():
                try:
                    os.remove(temp_new_path1)
                    print(f"Cleaned up {temp_new_path1}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_path2 and temp_new_path2.exists():
                try:
                    os.remove(temp_new_path2)
                    print(f"Cleaned up {temp_new_path2}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("分割エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_path1 and temp_new_path1.exists():
                try:
                    os.remove(temp_new_path1)
                    print(f"Cleaned up {temp_new_path1}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_path2 and temp_new_path2.exists():
                try:
                    os.remove(temp_new_path2)
                    print(f"Cleaned up {temp_new_path2}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False

    def split_column_by_selection(self, column_path_to_split, selected_char_indices):
        """選択された文字とそれ以外で列を2つに分割する"""
        temp_new_path_sel = None
        temp_new_path_oth = None

        try:
            # 1. データ取得と検証
            column_data = self.get_column_data(column_path_to_split)
            if not column_data:
                raise ValueError(f"列データが見つかりません: {column_path_to_split}")

            char_boxes_col = column_data.get("char_boxes_in_column", [])
            unicode_ids = column_data.get("unicode_ids", [])
            original_image_path = column_data.get("original_image")
            col_box_orig_crop = column_data.get("box_in_original")

            if not original_image_path:
                raise ValueError("元画像パスがありません。")
            if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list):
                raise ValueError("文字ボックスまたはUnicode IDの形式が不正です。")
            if len(char_boxes_col) != len(unicode_ids):
                raise ValueError("文字ボックスとUnicode IDの数が一致しません。")
            if not isinstance(col_box_orig_crop, list) or len(col_box_orig_crop) != 4:
                raise ValueError("列の切り抜き座標(box_in_original)が不正です。")
            if not selected_char_indices:
                raise ValueError("分割する文字が選択されていません。")
            if len(selected_char_indices) == len(unicode_ids):
                raise ValueError("全ての文字が選択されています。分割できません。")

            # 2. 元画像上の絶対座標を取得
            char_boxes_orig = self._get_char_boxes_in_original(column_data)
            if len(char_boxes_orig) != len(unicode_ids):
                raise RuntimeError("文字の元画像座標の取得に失敗しました。")

            # 3. 選択された文字とそれ以外に分割
            selected_indices_set = set(selected_char_indices)
            chars_orig_selected = []
            ids_selected = []
            chars_orig_other = []
            ids_other = []

            for i, (box_orig, uid) in enumerate(zip(char_boxes_orig, unicode_ids, strict=False)):
                if i in selected_indices_set:
                    chars_orig_selected.append(box_orig)
                    ids_selected.append(uid)
                else:
                    chars_orig_other.append(box_orig)
                    ids_other.append(uid)

            # Y座標でソート (元の順番を維持するため、分割後にソート)
            if chars_orig_selected and len(chars_orig_selected) > 1:
                sorted_indices_sel = np.argsort([box[1] for box in chars_orig_selected])
                chars_orig_selected = [chars_orig_selected[i] for i in sorted_indices_sel]
                ids_selected = [ids_selected[i] for i in sorted_indices_sel]
            if chars_orig_other and len(chars_orig_other) > 1:
                sorted_indices_oth = np.argsort([box[1] for box in chars_orig_other])
                chars_orig_other = [chars_orig_other[i] for i in sorted_indices_oth]
                ids_other = [ids_other[i] for i in sorted_indices_oth]

            # 4. 新しい列を生成 (2つ)
            new_data_sel, new_path_sel = self._recreate_column_from_chars(
                chars_orig_selected, ids_selected, original_image_path, column_path_to_split, "_selA"
            )
            if new_data_sel is None:
                raise RuntimeError("選択された文字グループの列生成に失敗しました。")
            temp_new_path_sel = self.processed_dir / new_path_sel

            new_data_oth, new_path_oth = self._recreate_column_from_chars(
                chars_orig_other, ids_other, original_image_path, column_path_to_split, "_selB"
            )
            if new_data_oth is None:
                raise RuntimeError("選択されなかった文字グループの列生成に失敗しました。")
            temp_new_path_oth = self.processed_dir / new_path_oth

            # 5. DataFrameの更新
            index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
            new_rows_df = pd.DataFrame([new_data_sel, new_data_oth])

            # --- アトミックな操作 ---
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

            # 元の列画像ファイルを削除
            try:
                abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
                if abs_path_to_delete and abs_path_to_delete.exists():
                    os.remove(abs_path_to_delete)
            except Exception as e:
                print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Column {column_path_to_split} split by selection into {new_path_sel} and {new_path_oth}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("選択分割エラー", f"選択分割中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_path_sel and temp_new_path_sel.exists():
                try:
                    os.remove(temp_new_path_sel)
                    print(f"Cleaned up {temp_new_path_sel}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_path_oth and temp_new_path_oth.exists():
                try:
                    os.remove(temp_new_path_oth)
                    print(f"Cleaned up {temp_new_path_oth}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("選択分割エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_path_sel and temp_new_path_sel.exists():
                try:
                    os.remove(temp_new_path_sel)
                    print(f"Cleaned up {temp_new_path_sel}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_path_oth and temp_new_path_oth.exists():
                try:
                    os.remove(temp_new_path_oth)
                    print(f"Cleaned up {temp_new_path_oth}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False

    def move_characters(self, src_column_path, target_column_path, char_indices_to_move):
        """特定の文字をソース列からターゲット列へ移動"""
        temp_new_src_path = None
        temp_new_tgt_path = None

        try:
            # 1. 基本チェック
            if src_column_path == target_column_path:
                raise ValueError("同じ列内で文字を移動することはできません。")
            if not char_indices_to_move:
                raise ValueError("移動する文字が選択されていません。")

            # 2. データ取得と検証
            src_data = self.get_column_data(src_column_path)
            tgt_data = self.get_column_data(target_column_path)
            if not src_data:
                raise ValueError(f"移動元の列データが見つかりません: {src_column_path}")
            if not tgt_data:
                raise ValueError(f"移動先の列データが見つかりません: {target_column_path}")

            src_orig_img = src_data.get("original_image")
            tgt_orig_img = tgt_data.get("original_image")
            if not src_orig_img or not tgt_orig_img:
                raise ValueError("元画像パスがありません。")
            if src_orig_img != tgt_orig_img:
                raise ValueError("異なる元画像の列間で文字を移動することはできません。")

            src_chars_col = src_data.get("char_boxes_in_column", [])
            src_ids = src_data.get("unicode_ids", [])
            tgt_chars_col = tgt_data.get("char_boxes_in_column", [])
            tgt_ids = tgt_data.get("unicode_ids", [])

            if not isinstance(src_chars_col, list) or not isinstance(src_ids, list) or len(src_chars_col) != len(src_ids):
                raise ValueError(f"移動元({src_column_path})の文字データ形式が不正です。")
            if not isinstance(tgt_chars_col, list) or not isinstance(tgt_ids, list) or len(tgt_chars_col) != len(tgt_ids):
                raise ValueError(f"移動先({target_column_path})の文字データ形式が不正です。")

            # 3. 移動対象と残すものを仕分け (元画像座標)
            src_chars_orig = self._get_char_boxes_in_original(src_data)
            if len(src_chars_orig) != len(src_ids):
                raise RuntimeError("移動元の文字座標取得に失敗しました。")

            moved_chars_orig = []
            moved_ids = []
            remaining_src_chars_orig = []
            remaining_src_ids = []
            src_indices_set = set(char_indices_to_move)

            for i, (box_orig, uid) in enumerate(zip(src_chars_orig, src_ids, strict=False)):
                if i in src_indices_set:
                    moved_chars_orig.append(box_orig)
                    moved_ids.append(uid)
                else:
                    remaining_src_chars_orig.append(box_orig)
                    remaining_src_ids.append(uid)

            if not moved_chars_orig:
                raise ValueError("指定されたインデックスの移動対象文字が見つかりませんでした。")

            # 4. 移動先の文字リストと結合、ソート (元画像座標)
            tgt_chars_orig = self._get_char_boxes_in_original(tgt_data)
            if len(tgt_chars_orig) != len(tgt_ids):
                raise RuntimeError("移動先の文字座標取得に失敗しました。")

            combined_tgt_chars_orig = tgt_chars_orig + moved_chars_orig
            combined_tgt_ids = tgt_ids + moved_ids
            if len(combined_tgt_chars_orig) > 1:
                sorted_indices = np.argsort([box[1] for box in combined_tgt_chars_orig])
                final_tgt_chars_orig = [combined_tgt_chars_orig[i] for i in sorted_indices]
                final_tgt_ids = [combined_tgt_ids[i] for i in sorted_indices]
            else:
                final_tgt_chars_orig = combined_tgt_chars_orig
                final_tgt_ids = combined_tgt_ids

            # 5. 移動元と移動先の列を再生成
            # 移動元 (空になる可能性あり)
            new_src_data, new_src_path = self._recreate_column_from_chars(
                remaining_src_chars_orig, remaining_src_ids, src_orig_img, src_column_path, "_move_src"
            )
            # new_src_data が None でもエラーではない (空になった場合)
            if new_src_path:
                temp_new_src_path = self.processed_dir / new_src_path

            # 移動先
            new_tgt_data, new_tgt_path = self._recreate_column_from_chars(
                final_tgt_chars_orig, final_tgt_ids, tgt_orig_img, target_column_path, "_move_tgt"
            )
            if new_tgt_data is None:
                raise RuntimeError("移動先の列の再生成に失敗しました。")
            temp_new_tgt_path = self.processed_dir / new_tgt_path

            # 6. DataFrame更新
            indices_to_drop = self.df[
                (self.df["column_image"] == src_column_path) | (self.df["column_image"] == target_column_path)
            ].index

            new_rows = []
            if new_src_data:  # 移動元が空でなければ追加
                new_rows.append(new_src_data)
            new_rows.append(new_tgt_data)  # 移動先は必ず追加
            new_rows_df = pd.DataFrame(new_rows)

            # --- アトミックな操作 ---
            self.df = self.df.drop(indices_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

            # 古い画像削除
            try:
                abs_src = self.get_column_abs_path(src_column_path)
                if abs_src and abs_src.exists():
                    os.remove(abs_src)
            except Exception as e:
                print(f"Warning: could not delete {src_column_path}: {e}")
            try:
                abs_tgt = self.get_column_abs_path(target_column_path)
                if abs_tgt and abs_tgt.exists():
                    os.remove(abs_tgt)
            except Exception as e:
                print(f"Warning: could not delete {target_column_path}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Characters moved from {src_column_path} to {new_tgt_path}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("文字移動エラー", f"文字移動中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_src_path and temp_new_src_path.exists():
                try:
                    os.remove(temp_new_src_path)
                    print(f"Cleaned up {temp_new_src_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_tgt_path and temp_new_tgt_path.exists():
                try:
                    os.remove(temp_new_tgt_path)
                    print(f"Cleaned up {temp_new_tgt_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("文字移動エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_src_path and temp_new_src_path.exists():
                try:
                    os.remove(temp_new_src_path)
                    print(f"Cleaned up {temp_new_src_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            if temp_new_tgt_path and temp_new_tgt_path.exists():
                try:
                    os.remove(temp_new_tgt_path)
                    print(f"Cleaned up {temp_new_tgt_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False

    def delete_column(self, column_path_to_delete):
        """列を削除する"""
        try:
            # 1. データ取得と検証
            column_data = self.get_column_data(column_path_to_delete)
            if not column_data:
                # すでに削除されているか、存在しないパスかもしれないので警告に留める
                print(f"Warning: 削除対象の列データが見つかりません: {column_path_to_delete}")
                return False  # 削除は失敗とする

            # 2. DataFrameから削除するインデックスを取得
            index_to_drop = self.df[self.df["column_image"] == column_path_to_delete].index
            if index_to_drop.empty:
                print(f"Warning: DataFrame内に列が見つかりません: {column_path_to_delete}")
                # ファイルだけ削除する？ ここでは何もしない
                return False

            # --- アトミックな操作 ---
            # DataFrameから削除
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)

            # 画像ファイルを削除
            try:
                abs_path_to_delete = self.get_column_abs_path(column_path_to_delete)
                if abs_path_to_delete and abs_path_to_delete.exists():
                    os.remove(abs_path_to_delete)
                    print(f"Deleted column image: {abs_path_to_delete}")
                    # ディレクトリが空になったら削除する (オプション)
                    try:
                        parent_dir = abs_path_to_delete.parent
                        if not any(parent_dir.iterdir()):
                            shutil.rmtree(parent_dir)
                            print(f"Deleted empty directory: {parent_dir}")
                    except Exception as dir_e:
                        print(f"Warning: Could not delete directory {parent_dir}: {dir_e}")

            except Exception as e:
                print(f"Warning: Could not delete column image file or dir {column_path_to_delete}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Deleted column: {column_path_to_delete}")
            return True

        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("列削除エラー", f"列削除中に予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            return False

    def delete_character(self, column_path, char_index_to_delete):
        """列から指定した文字を削除する"""
        temp_new_col_path = None
        try:
            # 1. データ取得と検証
            col_data = self.get_column_data(column_path)
            if not col_data:
                raise ValueError(f"Column data not found: {column_path}")

            char_boxes_col = col_data.get("char_boxes_in_column", [])
            unicode_ids = col_data.get("unicode_ids", [])
            original_image_path = col_data.get("original_image")
            col_box_orig_crop = col_data.get("box_in_original")

            if not original_image_path:
                raise ValueError("元画像パスがありません。")
            if (
                not isinstance(char_boxes_col, list)
                or not isinstance(unicode_ids, list)
                or len(char_boxes_col) != len(unicode_ids)
            ):
                raise ValueError("文字データ形式が不正です。")
            if not isinstance(col_box_orig_crop, list) or len(col_box_orig_crop) != 4:
                raise ValueError("列の切り抜き座標(box_in_original)が不正です。")
            if not 0 <= char_index_to_delete < len(unicode_ids):
                raise ValueError("無効な文字インデックスです。")

            # 2. 残す文字のリストを作成
            new_unicode_ids = [uid for i, uid in enumerate(unicode_ids) if i != char_index_to_delete]

            if not new_unicode_ids:  # 全ての文字が削除された場合
                print(f"All characters deleted from {column_path}. Deleting column.")
                # 列ごと削除して終了
                return self.delete_column(column_path)

            # 3. 残す文字の元画像座標を取得
            char_boxes_orig = self._get_char_boxes_in_original(col_data)
            if len(char_boxes_orig) != len(unicode_ids):
                raise RuntimeError("文字の元画像座標の取得に失敗しました。")

            new_char_boxes_orig = [box for i, box in enumerate(char_boxes_orig) if i != char_index_to_delete]

            # 4. 列を再生成
            new_col_data, new_col_path = self._recreate_column_from_chars(
                new_char_boxes_orig, new_unicode_ids, original_image_path, column_path, "_chardel"
            )
            if new_col_data is None:
                raise RuntimeError("文字削除後の列の再生成に失敗しました。")
            temp_new_col_path = self.processed_dir / new_col_path

            # 5. DataFrame Update
            index_to_drop = self.df[self.df["column_image"] == column_path].index
            new_row_df = pd.DataFrame([new_col_data])

            # --- アトミックな操作 ---
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

            # Delete old image
            try:
                abs_old_path = self.get_column_abs_path(column_path)
                if abs_old_path and abs_old_path.exists():
                    os.remove(abs_old_path)
            except Exception as e:
                print(f"Warning: could not delete {column_path}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Character {char_index_to_delete} deleted from {column_path}, recreated as {new_col_path}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("文字削除エラー", f"文字削除中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_col_path and temp_new_col_path.exists():
                try:
                    os.remove(temp_new_col_path)
                    print(f"Cleaned up {temp_new_col_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("文字削除エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_col_path and temp_new_col_path.exists():
                try:
                    os.remove(temp_new_col_path)
                    print(f"Cleaned up {temp_new_col_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False

    def add_character(self, column_path, new_unicode_id, insert_after_index):
        """列に新しい文字を追加する"""
        temp_new_col_path = None
        try:
            # 1. データ取得と検証
            col_data = self.get_column_data(column_path)
            if not col_data:
                raise ValueError(f"Column data not found: {column_path}")

            char_boxes_col = col_data.get("char_boxes_in_column", [])
            unicode_ids = col_data.get("unicode_ids", [])
            original_image_path = col_data.get("original_image")
            col_box_orig_crop = col_data.get("box_in_original")  # 切り抜き座標

            if not original_image_path:
                raise ValueError("元画像パスがありません。")
            if (
                not isinstance(char_boxes_col, list)
                or not isinstance(unicode_ids, list)
                or len(char_boxes_col) != len(unicode_ids)
            ):
                raise ValueError("文字データ形式が不正です。")
            if not isinstance(col_box_orig_crop, list) or len(col_box_orig_crop) != 4:
                raise ValueError("列の切り抜き座標(box_in_original)が不正です。")

            # 挿入インデックス決定 (insert_after_index の次)
            if insert_after_index == -1:  # 何も選択されていない -> 先頭に追加
                insert_index = 0
            elif 0 <= insert_after_index < len(unicode_ids):
                insert_index = insert_after_index + 1
            else:  # 末尾に追加
                insert_index = len(unicode_ids)

            # 2. 元画像上の文字座標リストを取得
            char_boxes_orig = self._get_char_boxes_in_original(col_data)
            if len(char_boxes_orig) != len(unicode_ids):
                raise RuntimeError("文字の元画像座標の取得に失敗しました。")

            # 3. 新しい文字の座標を計算 (元画像座標系で)
            crop_x1, crop_y1, crop_x2, crop_y2 = col_box_orig_crop
            col_width_crop = crop_x2 - crop_x1
            # col_height_crop = crop_y2 - crop_y1

            avg_width = 0
            avg_height = 0
            if char_boxes_orig:
                widths = [b[2] - b[0] for b in char_boxes_orig]
                heights = [b[3] - b[1] for b in char_boxes_orig]
                avg_width = np.mean(widths) if widths else col_width_crop * 0.8  # デフォルト幅
                avg_height = np.mean(heights) if heights else 20  # デフォルト高さ
            else:  # 列に文字が一つもない場合
                avg_width = col_width_crop * 0.8
                avg_height = 20

            # X座標: 列の中心付近 (切り抜き座標基準 -> 元画像座標)
            new_x1_orig = crop_x1 + (col_width_crop - avg_width) / 2
            new_x2_orig = new_x1_orig + avg_width

            # Y座標: 前後の文字から決定 (元画像座標)
            spacing = 5  # 文字間のデフォルトスペース
            new_y1_orig, new_y2_orig = 0, 0

            if insert_index == 0:  # 先頭に追加
                if char_boxes_orig:  # 既存文字がある
                    next_box_orig = char_boxes_orig[0]
                    new_y1_orig = next_box_orig[1] - avg_height - spacing
                    new_y2_orig = next_box_orig[1] - spacing
                else:  # 最初の文字
                    new_y1_orig = crop_y1 + spacing  # 切り抜き領域の上端近く
                    new_y2_orig = new_y1_orig + avg_height
            elif insert_index == len(char_boxes_orig):  # 末尾に追加
                prev_box_orig = char_boxes_orig[-1]
                new_y1_orig = prev_box_orig[3] + spacing
                new_y2_orig = new_y1_orig + avg_height
            else:  # 中間に追加
                prev_box_orig = char_boxes_orig[insert_index - 1]
                next_box_orig = char_boxes_orig[insert_index]
                # 前後のボックスの中間に配置
                center_y = (prev_box_orig[3] + next_box_orig[1]) / 2
                new_y1_orig = center_y - avg_height / 2
                new_y2_orig = center_y + avg_height / 2

            # 座標を整数化し、負にならないようにする
            new_box_orig = [
                max(0, int(new_x1_orig)),
                max(0, int(new_y1_orig)),
                max(0, int(new_x2_orig)),
                max(0, int(new_y2_orig)),
            ]

            # 4. 新しい文字リストを作成
            final_char_boxes_orig = char_boxes_orig[:insert_index] + [new_box_orig] + char_boxes_orig[insert_index:]
            final_unicode_ids = unicode_ids[:insert_index] + [new_unicode_id] + unicode_ids[insert_index:]

            # 5. 列を再生成
            new_col_data, new_col_path = self._recreate_column_from_chars(
                final_char_boxes_orig, final_unicode_ids, original_image_path, column_path, "_charadd"
            )
            if new_col_data is None:
                raise RuntimeError("文字追加後の列の再生成に失敗しました。")
            temp_new_col_path = self.processed_dir / new_col_path

            # 6. DataFrame Update
            index_to_drop = self.df[self.df["column_image"] == column_path].index
            new_row_df = pd.DataFrame([new_col_data])

            # --- アトミックな操作 ---
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

            # Delete old image
            try:
                abs_old_path = self.get_column_abs_path(column_path)
                if abs_old_path and abs_old_path.exists():
                    os.remove(abs_old_path)
            except Exception as e:
                print(f"Warning: could not delete {column_path}: {e}")
            # --- アトミックな操作ここまで ---

            self.changes_made = True
            print(f"Character {new_unicode_id} added to {column_path} (index {insert_index}), recreated as {new_col_path}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            messagebox.showerror("文字追加エラー", f"文字追加中にエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_col_path and temp_new_col_path.exists():
                try:
                    os.remove(temp_new_col_path)
                    print(f"Cleaned up {temp_new_col_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False
        except Exception as e:  # 予期せぬエラー
            messagebox.showerror("文字追加エラー", f"予期せぬエラーが発生しました:\n{e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if temp_new_col_path and temp_new_col_path.exists():
                try:
                    os.remove(temp_new_col_path)
                    print(f"Cleaned up {temp_new_col_path}")
                except Exception as clean_e:
                    print(f"Cleanup error: {clean_e}")
            return False

    def _recreate_column_from_chars(self, chars_in_orig, u_ids, original_img_path, base_rel_path_str, suffix):
        """
        指定された元画像上の文字座標リストから新しい列データと列画像を生成する共通ヘルパー関数。
        エラーハンドリングを強化。
        Returns: (new_column_data_dict, new_column_relative_path) or (None, None) on failure.
        """
        new_col_abs_path = None  # エラー時のクリーンアップ用
        try:
            # 引数チェック (u_ids もリストであることを期待)
            if not isinstance(u_ids, list):
                raise ValueError("_recreate_column_from_chars: u_ids must be a list.")
            # 空リストの場合、列データは None を返すが、パスも None で返す
            if not chars_in_orig and not u_ids:
                # 空の列も許容する場合 (例: 移動元が空になる) は、None, None を返す
                # print(f"Debug: Creating an empty column data for suffix {suffix}")
                return None, None  # 空の列データを示す
            elif not chars_in_orig or len(chars_in_orig) != len(u_ids):
                # 座標とIDの数が合わない場合はエラー
                raise ValueError(
                    f"_recreate_column_from_chars: Mismatch between char boxes ({len(chars_in_orig)}) and unicode IDs ({len(u_ids)}) for suffix {suffix}."
                )

            # 1. 新しい列のバウンディングボックス (元画像上、マージンなし)
            new_col_bounds_no_margin = self._recalculate_column_bounds(chars_in_orig)
            nc_x1, nc_y1, nc_x2, nc_y2 = new_col_bounds_no_margin

            # 2. マージン追加と切り抜き座標計算
            orig_img_abs_path = self.get_original_image_abs_path(original_img_path)
            if not orig_img_abs_path or not orig_img_abs_path.exists():
                raise FileNotFoundError(f"元画像が見つかりません: {orig_img_abs_path}")
            try:
                orig_img = Image.open(orig_img_abs_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"元画像の読み込みに失敗しました {orig_img_abs_path}: {e}")

            margin = 5  # TODO: 設定可能にするか、より賢いマージン計算を検討
            crop_x1 = max(0, int(nc_x1 - margin))
            crop_y1 = max(0, int(nc_y1 - margin))
            crop_x2 = min(orig_img.width, int(nc_x2 + margin))
            crop_y2 = min(orig_img.height, int(nc_y2 + margin))

            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                # 有効な切り抜き領域がない場合 (文字サイズが0など)
                # 空の列データではなくエラーとして扱う方が良いか？
                # → 空の座標リストで呼ばれた場合以外はエラーとする
                if chars_in_orig:
                    raise ValueError(
                        f"計算された切り抜き領域が無効です for {base_rel_path_str}{suffix} ([{crop_x1},{crop_y1},{crop_x2},{crop_y2}])"
                    )
                else:
                    # chars_in_orig が空なら、空のデータとして None, None を返す
                    return None, None

            # 3. 列画像切り出し
            try:
                new_col_img_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                # 切り出した画像が有効か確認 (サイズが0でないか)
                if new_col_img_pil.width <= 0 or new_col_img_pil.height <= 0:
                    raise RuntimeError(
                        f"切り出した列画像のサイズが無効です ({new_col_img_pil.width}x{new_col_img_pil.height}) for {base_rel_path_str}{suffix}"
                    )
            except Exception as e:
                raise RuntimeError(f"列画像の切り出しに失敗しました for {base_rel_path_str}{suffix}: {e}")

            # 4. 新しい列画像のパス決定と保存
            base_path = Path(base_rel_path_str)
            # タイムスタンプにマイクロ秒まで含めてファイル名の衝突を避ける
            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")
            new_filename = f"{base_path.stem}{suffix}_{timestamp}.jpg"
            # 親ディレクトリが存在することを確認してから結合
            if base_path.parent:
                new_col_rel_path = base_path.parent / new_filename
            else:  # 親がない場合(通常ありえないが)、column_images直下など？
                # ここではエラーとする
                raise ValueError(f"基準パス '{base_rel_path_str}' から親ディレクトリを取得できません。")

            new_col_abs_path = self.processed_dir / new_col_rel_path
            new_col_abs_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                new_col_img_pil.save(new_col_abs_path, "JPEG", quality=95)  # 品質指定
            except Exception as e:
                raise RuntimeError(f"新しい列画像の保存に失敗しました {new_col_abs_path}: {e}")

            # 5. 列画像内の文字座標 (相対座標、マージン考慮)
            char_boxes_in_cropped = []
            for box_orig in chars_in_orig:
                rel_x1 = box_orig[0] - crop_x1
                rel_y1 = box_orig[1] - crop_y1
                rel_x2 = box_orig[2] - crop_x1
                rel_y2 = box_orig[3] - crop_y1
                # 座標がマイナスにならないように & 整数化
                char_boxes_in_cropped.append(
                    [max(0, int(rel_x1)), max(0, int(rel_y1)), max(0, int(rel_x2)), max(0, int(rel_y2))]
                )

            # 6. 新しい行データ作成
            new_data = {
                # パスは常に / 区切りで保存
                "column_image": str(new_col_rel_path).replace("\\", "/"),
                "original_image": original_img_path,
                "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],  # マージン込みの切り抜き座標
                "char_boxes_in_column": char_boxes_in_cropped,  # 切り出した画像内での相対座標
                "unicode_ids": u_ids,
            }
            return new_data, new_col_rel_path  # 成功

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"Error in _recreate_column_from_chars: {e}")
            traceback.print_exc()
            # ★エラー時: 作成された中間ファイルを削除
            if new_col_abs_path and new_col_abs_path.exists():
                try:
                    os.remove(new_col_abs_path)
                    print(f"Cleaned up temporary file: {new_col_abs_path}")
                except Exception as clean_e:
                    print(f"Error during cleanup of {new_col_abs_path}: {clean_e}")
            return None, None  # 失敗
        except Exception as e:  # 予期せぬエラー
            print(f"Unexpected error in _recreate_column_from_chars: {e}")
            traceback.print_exc()
            # ★エラー時: クリーンアップ
            if new_col_abs_path and new_col_abs_path.exists():
                try:
                    os.remove(new_col_abs_path)
                    print(f"Cleaned up temporary file: {new_col_abs_path}")
                except Exception as clean_e:
                    print(f"Error during cleanup of {new_col_abs_path}: {clean_e}")
            return None, None  # 失敗

    def save_changes(self):
        if not self.changes_made:
            # print("変更はありません。")
            return True
        if self.df is None:
            print("Warning: No data loaded, cannot save changes.")
            return False

        # ★変更: 元ファイルは上書きせず、新しいファイルに保存する
        save_path = self.output_csv_path
        print(f"Saving changes to: {save_path}")

        # バックアップを作成 (保存先ファイルのバックアップ)
        if save_path.exists():
            backup_path = save_path.with_suffix(f".{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.bak")
            try:
                shutil.copy2(save_path, backup_path)
                print(f"バックアップを作成しました: {backup_path}")
            except Exception as e:
                print(f"警告: バックアップの作成に失敗しました。 {e}")

        # CSVファイルに保存
        try:
            # DataFrameを保存用に整形（リストや辞書が文字列になるように）
            # ただし、pandas の to_csv は通常リストを適切に文字列化するはず
            # df_to_save = self.df.copy()
            # for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]:
            #     if col in df_to_save.columns:
            #          # NaNやNoneを空リスト文字列'[]'に変換してから文字列化
            #          df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if isinstance(x, list) else '[]')

            # to_csvでリストが引用符で囲まれるようにする (quoting=csv.QUOTE_NONNUMERIC など)
            # 標準の動作で問題ないか確認。問題があれば修正。
            # 標準ではリストはそのまま '[...]' という文字列として保存されるはず。読み込み時に literal_eval するのでOK。
            self.df.to_csv(save_path, index=False, encoding="utf-8-sig")  # UTF-8 BOM付きでExcelでの互換性向上
            self.changes_made = False
            print(f"変更を保存しました: {save_path}")
            return True
        except Exception as e:
            messagebox.showerror("保存エラー", f"CSVファイルへの保存中にエラーが発生しました:\n{save_path}\n{e}")
            print(f"Error saving CSV to {save_path}: {e}")
            traceback.print_exc()
            return False


class AnnotatorApp:
    def __init__(self, root, base_data_dir):
        self.root = root
        self.root.title("縦書き列アノテーション修正ツール")
        # ウィンドウサイズ調整
        self.root.geometry("1300x850")  # 少し広げる

        try:
            self.data_manager = DataManager(base_data_dir)
        except (FileNotFoundError, RuntimeError) as e:
            messagebox.showerror("起動エラー", f"データ読み込み中にエラーが発生しました:\n{e}\nアプリケーションを終了します。")
            self.root.destroy()
            return  # アプリケーション初期化中断
        except Exception as e:
            messagebox.showerror("起動エラー", f"予期せぬエラーが発生しました:\n{e}\nアプリケーションを終了します。")
            traceback.print_exc()
            self.root.destroy()
            return  # アプリケーション初期化中断

        # --- メインフレーム ---
        main_frame = ttk.Frame(root, padding="5")
        main_frame.pack(expand=True, fill="both")
        main_frame.rowconfigure(1, weight=1)  # 画像表示エリアを拡大可能に
        main_frame.columnconfigure(0, weight=3)  # 元画像エリアを広く
        main_frame.columnconfigure(1, weight=1)  # 列リストエリア
        main_frame.columnconfigure(2, weight=2)  # 詳細表示エリア

        # --- 上部: ナビゲーション ---
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)

        self.prev_button = ttk.Button(nav_frame, text="<< 前の画像 (←)", command=self.prev_image)
        self.prev_button.pack(side="left", padx=5)
        # 画像ラベルは中央揃えで伸縮させるためにFrameに入れる
        image_label_frame = ttk.Frame(nav_frame)
        image_label_frame.pack(side="left", padx=5, fill="x", expand=True)
        self.image_label = ttk.Label(image_label_frame, text="画像: ", anchor="center")  # 中央揃え
        self.image_label.pack(fill="x")

        self.next_button = ttk.Button(nav_frame, text="次の画像 (→) >>", command=self.next_image)
        self.next_button.pack(side="left", padx=5)  # 次へボタンはラベルの右

        # 右詰めの要素 (保存ボタンとページ情報) 用のフレーム
        right_nav_frame = ttk.Frame(nav_frame)
        right_nav_frame.pack(side="right", padx=5)

        self.save_button = ttk.Button(right_nav_frame, text="変更を保存", command=self.save_all_changes)
        self.save_button.pack(side="left", padx=5)  # 保存ボタン
        self.page_info_label = ttk.Label(right_nav_frame, text="- / -", width=10, anchor="e")  # ページ情報ラベル
        self.page_info_label.pack(side="left", padx=5)

        # --- 左: 元画像表示 ---
        orig_frame = ttk.LabelFrame(main_frame, text="元画像と列範囲")
        orig_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        orig_frame.rowconfigure(0, weight=1)
        orig_frame.columnconfigure(0, weight=1)
        self.orig_canvas = ImageCanvas(orig_frame, bg="lightgrey")
        self.orig_canvas.grid(row=0, column=0, sticky="nsew")
        # スクロールバー追加 (Canvasが大きい場合に機能)
        vsb_orig = ttk.Scrollbar(orig_frame, orient="vertical", command=self.orig_canvas.yview)
        hsb_orig = ttk.Scrollbar(orig_frame, orient="horizontal", command=self.orig_canvas.xview)
        self.orig_canvas.configure(yscrollcommand=vsb_orig.set, xscrollcommand=hsb_orig.set)
        vsb_orig.grid(row=0, column=1, sticky="ns")
        hsb_orig.grid(row=1, column=0, sticky="ew")

        # --- 中央: 列リストと操作 ---
        list_op_frame = ttk.Frame(main_frame)
        list_op_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        list_op_frame.rowconfigure(1, weight=1)  # リストボックスを拡大
        list_op_frame.columnconfigure(0, weight=1)

        op_buttons_frame = ttk.LabelFrame(list_op_frame, text="操作")  # ラベルフレームに変更
        op_buttons_frame.grid(row=0, column=0, sticky="ew", pady=5)
        # ボタンをグリッドレイアウトに変更して均等配置
        op_buttons_frame.columnconfigure((0, 1, 2), weight=1)

        self.merge_button = ttk.Button(op_buttons_frame, text="結合 (Enter)", command=self.merge_selected_columns)
        self.merge_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.split_button = ttk.Button(op_buttons_frame, text="1点分割", command=self.split_selected_column)
        self.split_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.split_selection_button = ttk.Button(op_buttons_frame, text="選択分割", command=self.split_column_by_selection)
        self.split_selection_button.grid(row=0, column=2, padx=2, pady=2, sticky="ew")

        self.move_char_button = ttk.Button(op_buttons_frame, text="文字移動", command=self.initiate_move_character)
        self.move_char_button.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self.delete_col_button = ttk.Button(op_buttons_frame, text="列削除", command=self.delete_selected_column)
        self.delete_col_button.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        self.delete_char_button = ttk.Button(op_buttons_frame, text="文字削除", command=self.delete_selected_character)
        self.delete_char_button.grid(row=1, column=2, padx=2, pady=2, sticky="ew")
        # ★追加: 文字追加ボタン
        self.add_char_button = ttk.Button(op_buttons_frame, text="文字追加", command=self.add_character)
        self.add_char_button.grid(row=2, column=0, columnspan=3, padx=2, pady=2, sticky="ew")

        list_frame = ttk.LabelFrame(list_op_frame, text="現在の画像の列一覧")
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.column_listbox = tk.Listbox(list_frame, selectmode="extended")  # extended で複数選択可能
        self.column_listbox.grid(row=0, column=0, sticky="nsew")
        self.column_listbox.bind("<<ListboxSelect>>", self.on_column_select)
        # スクロールバー
        list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.column_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.column_listbox.config(yscrollcommand=list_scrollbar.set)

        # --- 右: 選択列詳細 ---
        detail_frame = ttk.LabelFrame(main_frame, text="選択された列の詳細")
        detail_frame.grid(row=1, column=2, sticky="nsew", padx=5)
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        self.detail_canvas = ImageCanvas(detail_frame, bg="lightgrey")
        self.detail_canvas.grid(row=0, column=0, sticky="nsew")
        # スクロールバー追加 (オプション)
        vsb_detail = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_canvas.yview)
        hsb_detail = ttk.Scrollbar(detail_frame, orient="horizontal", command=self.detail_canvas.xview)
        self.detail_canvas.configure(yscrollcommand=vsb_detail.set, xscrollcommand=hsb_detail.set)
        vsb_detail.grid(row=0, column=1, sticky="ns")
        hsb_detail.grid(row=1, column=0, sticky="ew")

        # --- ステータスバー (オプション) ---
        self.status_bar = ttk.Label(root, text="準備完了", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- 初期化 ---
        self.selected_column_paths = []  # Listboxで選択中の列の相対パス
        self.current_detail_column_path = None  # detail_canvas に表示中の列の相対パス
        self.moving_characters_info = None  # 文字移動中の情報 {src_path: str, char_indices: list}

        # Canvasクリックのコールバックを設定 (トップレベルウィンドウにメソッドを持たせる)
        # self.orig_canvas.master.on_canvas_click = self.handle_orig_canvas_click # これだとフレームを指してしまう
        # self.detail_canvas.master.on_canvas_click = self.handle_detail_canvas_click
        # トップレベルウィンドウ (self.root) にメソッドを持たせ、Canvas側から呼び出すようにする -> ImageCanvasの on_left_click 内で修正済み

        self.load_current_image_data()

        # ウィンドウを閉じる際の処理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- キーボードショートカット (アプリケーション全体にバインド) ---
        self.root.bind_all("<KeyPress>", self.handle_key_press)

        # --- アプリケーション状態の更新 ---
        self.update_status("データディレクトリ: " + str(base_data_dir))

    # --- ステータスバー更新用メソッド ---
    def update_status(self, message):
        self.status_bar.config(text=message)
        # print(message) # コンソールにも表示

    def handle_key_press(self, event):
        """キーボードショートカットを処理する"""
        # print(f"Key pressed: {event.keysym}, state: {event.state}") # デバッグ用
        # テキスト入力中などはショートカットを無効にする (今回は入力フィールドがないので不要)

        if event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()
        elif event.keysym == "Return" or event.keysym == "KP_Enter":  # Enterキー (テンキー含む)
            # 複数列選択時のみ結合を実行
            if len(self.selected_column_paths) >= 2 and self.merge_button["state"] == tk.NORMAL:
                self.update_status("Enterキーで列を結合します...")
                self.merge_selected_columns()
            else:
                self.update_status("結合するには列を2つ以上選択してください。")

        # 他のキーバインドもここに追加可能
        # 例: Ctrl+S で保存
        elif event.keysym == "s" and bool(event.state & 0x0004):  # Ctrl+S
            self.update_status("Ctrl+S で保存します...")
            self.save_all_changes()

    def load_current_image_data(self):
        """現在選択されている元画像のデータを読み込み、表示を更新する"""
        self.update_status("画像データを読み込み中...")
        self.orig_canvas.clear_canvas()
        self.column_listbox.delete(0, tk.END)
        self.detail_canvas.clear_canvas()
        self.selected_column_paths = []
        self.current_detail_column_path = None
        self.moving_characters_info = None  # 移動状態をリセット
        self.root.config(cursor="")  # カーソルを標準に戻す

        current_image_path = self.data_manager.current_original_image
        all_images = self.data_manager.get_original_images()
        total_pages = len(all_images)

        if not current_image_path or not all_images:
            self.image_label.config(text="画像: (データなし)")
            self.page_info_label.config(text="0 / 0")  # ページ情報も更新
            self.update_status("表示する画像データがありません。")
            self.update_button_states()  # ボタン状態も更新
            return

        # ページ情報と画像ラベルの更新
        try:
            current_page_index = all_images.index(current_image_path)
            current_page_num = current_page_index + 1
            self.page_info_label.config(text=f"{current_page_num} / {total_pages}")
            # ラベル更新 (パスが長すぎる場合があるので調整)
            label_path = Path(current_image_path)
            # 例: '.../data/raw/dataset/doc/img.jpg' -> 'raw/dataset/doc/img.jpg'
            display_parts = label_path.parts[-4:]  # 末尾4要素を取得
            # base_path の親からの相対パスにする試み
            try:
                relative_display_path = label_path.relative_to(self.data_manager.base_path.parent)
                display_name = str(relative_display_path).replace("\\", "/")
            except ValueError:
                # base_path の外にある場合など
                display_name = "/".join(display_parts)

            self.image_label.config(text=f"画像: {display_name}")
        except ValueError:
            self.image_label.config(text=f"画像: {current_image_path} (リストに不整合?)")
            self.page_info_label.config(text=f"? / {total_pages}")
            self.update_status(f"エラー: 現在の画像パス {current_image_path} がリストに見つかりません。")

        # 元画像表示
        orig_abs_path = self.data_manager.get_original_image_abs_path(current_image_path)
        if orig_abs_path and orig_abs_path.exists():
            self.orig_canvas.load_image(orig_abs_path)
        else:
            print(f"Error: Original image file not found or path is incorrect: {orig_abs_path}")
            self.orig_canvas.clear_canvas()
            # エラーメッセージ表示
            error_text = f"元画像が見つかりません:\n{orig_abs_path}"
            self.orig_canvas.create_text(
                self.orig_canvas.winfo_width() // 2 if self.orig_canvas.winfo_width() > 1 else 300,
                self.orig_canvas.winfo_height() // 2 if self.orig_canvas.winfo_height() > 1 else 30,
                text=error_text,
                fill="red",
                anchor="center",
            )  # 中央表示に変更
            self.update_status(f"エラー: 元画像が見つかりません ({orig_abs_path})")

        # 列データを取得してリストボックスと元画像上のボックス表示
        columns_df = self.data_manager.get_columns_for_current_image()
        self.orig_canvas.boxes = []  # 元画像のボックスリストを初期化
        if not columns_df.empty:
            # 列を左端のX座標でソートして表示順を決定
            # box_in_original がリストでない場合や空の場合を考慮
            def get_sort_key(box):
                if isinstance(box, list) and len(box) > 0:
                    try:
                        return int(box[0])
                    except (ValueError, TypeError):
                        return float("inf")
                return float("inf")

            columns_df["sort_key"] = columns_df["box_in_original"].apply(get_sort_key)
            columns_df = columns_df.sort_values("sort_key").reset_index(drop=True)

            colors = ["orange", "blue", "green", "purple", "cyan", "magenta", "yellow"]  # 色の順番変更
            for i, row in columns_df.iterrows():
                col_rel_path = row["column_image"]
                # リストボックスに追加 (表示名と実際のパスを保持)
                display_text = f"{i:02d}: {Path(col_rel_path).name}"  # インデックスも表示
                self.column_listbox.insert(tk.END, display_text)
                # self.column_listbox.itemconfig(tk.END, {"fg": "black"}) # 必要ならリセット

                # 元画像に列の範囲を描画
                col_box = row["box_in_original"]
                if isinstance(col_box, list) and len(col_box) == 4 and all(isinstance(n, int | float) for n in col_box):
                    self.orig_canvas.add_box(
                        tag=col_rel_path,  # 列の相対パスをタグとして使用
                        x1=col_box[0],
                        y1=col_box[1],
                        x2=col_box[2],
                        y2=col_box[3],
                        color=colors[i % len(colors)],
                        width=2,  # 少し太く
                        text=f"{i:02d}",  # ボックスにも番号表示
                    )
                else:
                    print(f"Warning: Invalid or malformed column box format for {col_rel_path}: {col_box}. Skipping drawing.")

            self.orig_canvas.redraw_boxes()  # まとめて描画

        # リストボックスの最初の項目を選択状態にする (項目があれば)
        if self.column_listbox.size() > 0:
            self.column_listbox.selection_clear(0, tk.END)  # 念のため選択解除
            self.column_listbox.selection_set(0)
            self.column_listbox.activate(0)  # アクティブな項目も設定
            self.column_listbox.see(0)  # 最初の項目が見えるようにスクロール
            # 選択イベントを手動で発火させる
            self.root.after(10, self.column_listbox.event_generate, "<<ListboxSelect>>")  # 少し遅延させて実行
        else:
            # 列がない場合は詳細表示もクリア
            self.detail_canvas.clear_canvas()
            self.current_detail_column_path = None

        # ボタンの状態更新
        self.update_button_states()
        # アイドルタスクを処理し、フォーカスを強制的に設定
        self.root.update_idletasks()  # GUI更新を待機
        # self.column_listbox.focus_force() # フォーカスを強制設定 (場合によっては邪魔かも)
        self.update_status("画像データの読み込み完了。")

    def on_column_select(self, event=None):
        """リストボックスで列が選択されたときの処理"""
        selected_indices = self.column_listbox.curselection()
        # print(f"DEBUG: Listbox selection changed: {selected_indices}") # DEBUG

        # 現在の画像に対応する列データフレームを取得 (ソート済み)
        all_columns_df = self.data_manager.get_columns_for_current_image()
        if all_columns_df.empty and selected_indices:
            # データがないのに選択が発生した場合 (通常はないはず)
            self.selected_column_paths = []
            self.current_detail_column_path = None
            print("Warning: Column selected but no data available for current image.")
        elif not all_columns_df.empty:
            # ソートキーを再計算してソート (念のため)
            def get_sort_key(box):
                if isinstance(box, list) and len(box) > 0:
                    try:
                        return int(box[0])
                    except (ValueError, TypeError):
                        return float("inf")
                return float("inf")

            all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(get_sort_key)
            all_columns_df = all_columns_df.sort_values("sort_key").reset_index(drop=True)

            if selected_indices:
                # 選択されたインデックスに対応する列の相対パスを取得
                try:
                    self.selected_column_paths = [all_columns_df.iloc[idx]["column_image"] for idx in selected_indices]
                    # 最後に選択された列を詳細表示
                    last_selected_idx = selected_indices[-1]  # 複数選択の場合、最後にクリックされたもの
                    self.current_detail_column_path = all_columns_df.iloc[last_selected_idx]["column_image"]
                    self.load_detail_column(self.current_detail_column_path)
                    listbox_text = self.column_listbox.get(last_selected_idx)
                    self.update_status(f"列 '{listbox_text}' を選択中。")

                except IndexError:
                    print(
                        f"Error: Listbox index out of range. Indices: {selected_indices}, DataFrame rows: {len(all_columns_df)}"
                    )
                    self.selected_column_paths = []
                    self.current_detail_column_path = None
                    self.detail_canvas.clear_canvas()
                    self.update_status("エラー: 列選択インデックスが不正です。")
            else:
                # 何も選択されていない場合
                self.selected_column_paths = []
                self.current_detail_column_path = None
                self.detail_canvas.clear_canvas()
                self.update_status("列が選択されていません。")
        else:
            # all_columns_df が空で selected_indices もない場合
            self.selected_column_paths = []
            self.current_detail_column_path = None
            self.detail_canvas.clear_canvas()

        # 元画像の選択列ハイライト更新
        self.highlight_selected_columns_on_orig()
        # ボタン状態更新
        self.update_button_states()

    def load_detail_column(self, column_rel_path):
        """指定された列を詳細表示エリアに表示"""
        self.update_status(f"列詳細を読み込み中: {Path(column_rel_path).name}")
        self.detail_canvas.clear_canvas()
        column_data = self.data_manager.get_column_data(column_rel_path)
        if not column_data:
            error_msg = f"エラー: 列データが見つかりません ({column_rel_path})"
            print(error_msg)
            self.detail_canvas.create_text(10, 10, anchor="nw", text=error_msg, fill="red")
            self.update_status(error_msg)
            return

        col_abs_path = self.data_manager.get_column_abs_path(column_rel_path)
        if col_abs_path and col_abs_path.exists():
            self.detail_canvas.load_image(col_abs_path)
            self.detail_canvas.boxes = []  # 詳細ボックスリスト初期化

            # 文字のボックスを描画
            char_boxes = column_data.get("char_boxes_in_column")
            uids = column_data.get("unicode_ids")
            if isinstance(char_boxes, list) and isinstance(uids, list) and len(char_boxes) == len(uids):
                for i, (box, uid) in enumerate(zip(char_boxes, uids, strict=False)):
                    if isinstance(box, list) and len(box) == 4 and all(isinstance(n, int | float) for n in box):
                        x1, y1, x2, y2 = box
                        char_tag = f"char_{i}"  # タグ: char_インデックス
                        char_display = unicode_to_char(uid) if uid else "?"  # Unicodeを文字に変換
                        if char_display is None:
                            char_display = "?"  # 変換失敗時

                        self.detail_canvas.add_box(
                            tag=char_tag,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            color="green",
                            width=1,
                            text=f"{i}: {char_display}",  # インデックスと文字表示
                        )
                    else:
                        print(f"Warning: Invalid char box format for index {i} in {column_rel_path}: {box}")
                self.detail_canvas.redraw_boxes()  # まとめて描画
                self.update_status(f"列詳細表示完了 ({len(uids)}文字)。")
            elif char_boxes or uids:  # どちらか一方だけ存在するか、形式が違う場合
                warning_msg = f"警告: 列 {column_rel_path} の文字ボックス/Unicode IDの形式が不正か、数が一致しません。"
                print(warning_msg)
                self.detail_canvas.create_text(10, 30, anchor="nw", text=warning_msg, fill="orange")
                self.update_status(warning_msg)
            else:
                self.update_status("列詳細表示完了 (文字なし)。")

        else:
            error_msg = f"エラー: 列画像が見つかりません:\n{col_abs_path}"
            print(error_msg)
            self.detail_canvas.create_text(
                self.detail_canvas.winfo_width() // 2 if self.detail_canvas.winfo_width() > 1 else 150,
                self.detail_canvas.winfo_height() // 2 if self.detail_canvas.winfo_height() > 1 else 30,
                text=error_msg,
                fill="red",
                anchor="center",
            )
            self.update_status(f"エラー: 列画像が見つかりません ({col_abs_path})")

    def highlight_selected_columns_on_orig(self):
        """元画像上で選択中の列をハイライト"""
        if not hasattr(self.orig_canvas, "boxes") or not self.orig_canvas.boxes:
            return  # 初期化中など

        all_col_tags_on_orig = {box["tag"] for box in self.orig_canvas.boxes if not box["tag"].startswith("char_")}

        self.orig_canvas.selected_box_tags.clear()
        for path in self.selected_column_paths:
            if path in all_col_tags_on_orig:
                self.orig_canvas.selected_box_tags.add(path)

        self.orig_canvas.redraw_boxes()

    def update_button_states(self):
        """選択状態に応じてボタンの有効/無効を切り替え"""
        num_selected_cols = len(self.selected_column_paths)
        num_selected_chars = len(self.detail_canvas.get_selected_tags()) if self.current_detail_column_path else 0
        is_detail_view_active = bool(self.current_detail_column_path)

        # 結合: 2つ以上の列を選択
        self.merge_button.config(state=tk.NORMAL if num_selected_cols >= 2 else tk.DISABLED)
        # 1点分割: 1つの列を選択し、その詳細表示で1つの文字を選択 (分割点として)
        self.split_button.config(
            state=tk.NORMAL if num_selected_cols == 1 and num_selected_chars == 1 and is_detail_view_active else tk.DISABLED
        )
        # 選択分割: 1つの列を選択し、詳細表示で1つ以上かつ全部未満の文字を選択
        col_data = self.data_manager.get_column_data(self.current_detail_column_path) if is_detail_view_active else None
        total_chars = len(col_data.get("unicode_ids", [])) if col_data else 0
        self.split_selection_button.config(
            state=tk.NORMAL
            if num_selected_cols == 1 and is_detail_view_active and 0 < num_selected_chars < total_chars
            else tk.DISABLED
        )
        # 文字移動開始: 詳細表示で1つ以上の文字を選択
        self.move_char_button.config(state=tk.NORMAL if num_selected_chars > 0 and is_detail_view_active else tk.DISABLED)
        # 列削除: 1つ以上の列を選択
        self.delete_col_button.config(state=tk.NORMAL if num_selected_cols > 0 else tk.DISABLED)
        # 文字削除: 詳細表示で1つ以上の文字を選択
        self.delete_char_button.config(state=tk.NORMAL if num_selected_chars > 0 and is_detail_view_active else tk.DISABLED)
        # 文字追加: 1つの列を選択 (詳細表示中)
        self.add_char_button.config(state=tk.NORMAL if num_selected_cols == 1 and is_detail_view_active else tk.DISABLED)

        # 移動中の状態表示
        if self.moving_characters_info:
            self.root.config(cursor="crosshair")  # カーソル変更
            self.update_status(
                f"{len(self.moving_characters_info['char_indices'])}文字移動中。移動先の列を元画像でクリックしてください。"
            )
        elif self.root.cget("cursor") == "crosshair":  # 移動モードがキャンセルされた場合など
            self.root.config(cursor="")  # カーソルを戻す

    # --- ボタンアクション ---
    # 各アクションメソッド内で、実行前に update_status で実行中であることを示し、
    # 完了後またはエラー時に再度 update_status で結果を通知する

    def merge_selected_columns(self):
        if len(self.selected_column_paths) < 2:
            self.update_status("結合するには列を2つ以上選択してください。")
            return
        self.update_status(f"{len(self.selected_column_paths)}個の列を結合中...")
        success = self.data_manager.merge_columns(self.selected_column_paths)
        if success:
            self.update_status("列の結合が完了しました。表示を更新します...")
            self.load_current_image_data()  # 表示更新
        else:
            # エラーメッセージは DataManager 内で表示されるはず
            self.update_status("列の結合に失敗しました。")
            self.update_button_states()  # エラー後もボタン状態を更新

    def split_selected_column(self):
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            self.update_status("1点分割するには、列を1つだけ選択してください。")
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if len(selected_char_tags) != 1:
            messagebox.showinfo("情報", "分割点を指定するため、詳細表示で文字を1つだけ選択してください。")
            self.update_status("1点分割: 分割点の文字を1つ選択してください。")
            return

        try:
            # タグ 'char_i' からインデックス i を抽出
            split_char_index = int(selected_char_tags[0].split("_")[1])
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            self.update_status("エラー: 分割点の文字インデックス取得失敗。")
            return

        col_name = Path(self.current_detail_column_path).name
        self.update_status(f"列 '{col_name}' を文字 {split_char_index} の前で分割中...")
        success = self.data_manager.split_column(self.current_detail_column_path, split_char_index)
        if success:
            self.update_status("1点分割が完了しました。表示を更新します...")
            self.load_current_image_data()
        else:
            self.update_status(f"列 '{col_name}' の1点分割に失敗しました。")
            self.update_button_states()

    def split_column_by_selection(self):
        """詳細表示で選択された文字に基づいて列を分割する"""
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            messagebox.showerror("エラー", "選択分割を行うには、列リストで列を1つだけ選択してください。")
            self.update_status("選択分割: 列を1つ選択してください。")
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            messagebox.showerror("エラー", "分割する文字が選択されていません。詳細表示で文字をクリックして選択してください。")
            self.update_status("選択分割: 分割する文字を選択してください。")
            return

        try:
            selected_char_indices = sorted([int(tag.split("_")[1]) for tag in selected_char_tags])
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            self.update_status("エラー: 選択文字インデックス取得失敗。")
            return

        # 全ての文字が選択されていないか確認
        col_data = self.data_manager.get_column_data(self.current_detail_column_path)
        if col_data and len(selected_char_indices) == len(col_data.get("unicode_ids", [])):
            messagebox.showerror("エラー", "全ての文字が選択されています。分割できません。")
            self.update_status("選択分割: 全ての文字は選択できません。")
            return

        col_name = Path(self.current_detail_column_path).name
        confirm = messagebox.askyesno(
            "選択分割確認",
            f"列 '{col_name}' を、選択された {len(selected_char_indices)} 文字のグループと、\n"
            f"残りの文字のグループの2つに分割しますか？",
        )
        if confirm:
            self.update_status(f"列 '{col_name}' を選択分割中...")
            success = self.data_manager.split_column_by_selection(self.current_detail_column_path, selected_char_indices)
            if success:
                self.update_status("選択分割が完了しました。表示を更新します...")
                self.load_current_image_data()  # 表示更新
            else:
                self.update_status(f"列 '{col_name}' の選択分割に失敗しました。")
                self.update_button_states()
        else:
            self.update_status("選択分割をキャンセルしました。")

    def initiate_move_character(self):
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return

        try:
            char_indices = sorted([int(tag.split("_")[1]) for tag in selected_char_tags])
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            self.update_status("エラー: 移動文字インデックス取得失敗。")
            return

        self.moving_characters_info = {"src_path": self.current_detail_column_path, "char_indices": char_indices}
        messagebox.showinfo(
            "文字移動", f"{len(char_indices)}文字を選択しました。\n移動先の列を「元画像エリア」でクリックしてください。"
        )
        self.update_button_states()  # カーソル変更とステータス更新

    # ★変更: on_canvas_click をトップレベルクラスに移動
    def on_canvas_click(self, canvas_instance, clicked_tags, ctrl_pressed):
        """Canvasがクリックされたときのコールバックハンドラ"""
        if canvas_instance == self.orig_canvas:
            self.handle_orig_canvas_click(canvas_instance, clicked_tags, ctrl_pressed)
        elif canvas_instance == self.detail_canvas:
            self.handle_detail_canvas_click(canvas_instance, clicked_tags, ctrl_pressed)

    def handle_orig_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        """元画像Canvasがクリックされたときの処理 (主に文字移動先指定)"""
        if self.moving_characters_info and clicked_tags:
            # クリックされたタグの中から列パスと思われるものを探す
            target_col_path = None
            for tag in clicked_tags:
                # クリックされたのが列のボックスか簡易判定 (パス区切り文字を含むか)
                if isinstance(tag, str) and ("/" in tag or "\\" in tag):
                    # さらに、現在表示中の画像の列か確認
                    current_cols_df = self.data_manager.get_columns_for_current_image()
                    if tag in current_cols_df["column_image"].values:
                        target_col_path = tag
                        break  # 最初に見つかった列をターゲットとする

            if target_col_path:
                # print(f"移動先候補: {target_col_path}")
                src_path = self.moving_characters_info["src_path"]
                indices = self.moving_characters_info["char_indices"]
                src_name = Path(src_path).name
                tgt_name = Path(target_col_path).name

                if src_path == target_col_path:
                    messagebox.showinfo("情報", "同じ列には移動できません。")
                    # 移動モードは継続
                else:
                    # 確認ダイアログ
                    confirm = messagebox.askyesnocancel(
                        "文字移動確認",
                        f"{len(indices)}個の文字を\nFrom: {src_name}\nTo:   {tgt_name}\nに移動しますか？",
                        icon=messagebox.QUESTION,  # アイコン指定
                    )
                    if confirm is True:  # Yes
                        self.update_status(f"文字を {src_name} から {tgt_name} へ移動中...")
                        success = self.data_manager.move_characters(src_path, target_col_path, indices)
                        if success:
                            self.update_status("文字移動が完了しました。表示を更新します...")
                            # 移動モード解除は load_current_image_data の最初で行われる
                            self.load_current_image_data()  # 成功したら更新
                        else:
                            self.update_status("文字移動に失敗しました。")
                            # 移動モード解除
                            self.moving_characters_info = None
                            self.update_button_states()

                    elif confirm is False:  # No
                        self.update_status("文字移動をキャンセルしました。移動モード継続中。")
                        # 移動モードは継続
                    else:  # Cancel
                        self.update_status("文字移動を中断しました。移動モードを解除します。")
                        # 移動モード解除
                        self.moving_characters_info = None
                        self.update_button_states()

            else:  # 列以外の場所がクリックされた場合
                # messagebox.showinfo("情報", "文字を移動するには、移動先の列（枠）をクリックしてください。")
                self.update_status("文字移動: 移動先の列をクリックしてください。")
                # 移動モードは継続

        elif not self.moving_characters_info:
            # 通常のクリック（元画像上の列選択）
            # Canvas側で選択状態 (selected_box_tags) は更新済み
            selected_tags_on_orig = canvas.get_selected_tags()

            # クリックされたタグとリストボックスの選択状態を同期させる
            all_columns_df = self.data_manager.get_columns_for_current_image()
            if not all_columns_df.empty:
                # ソートキーでソート
                def get_sort_key(box):
                    if isinstance(box, list) and len(box) > 0:
                        try:
                            return int(box[0])
                        except (ValueError, TypeError):
                            return float("inf")
                    return float("inf")

                all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(get_sort_key)
                all_columns_df = all_columns_df.sort_values("sort_key").reset_index(drop=True)
                all_paths = all_columns_df["column_image"].tolist()

                # リストボックスの現在の選択を取得
                current_listbox_selection = self.column_listbox.curselection()
                # Canvas上で選択されたタグに対応するインデックスを取得
                indices_to_select_on_canvas = []
                for tag in selected_tags_on_orig:
                    if tag in all_paths:
                        try:
                            indices_to_select_on_canvas.append(all_paths.index(tag))
                        except ValueError:
                            pass

                # リストボックスの選択状態をCanvasの選択状態に合わせる
                # (イベントハンドラが無限ループしないように注意)
                if set(current_listbox_selection) != set(indices_to_select_on_canvas):
                    self.column_listbox.selection_clear(0, tk.END)
                    for idx in indices_to_select_on_canvas:
                        self.column_listbox.selection_set(idx)
                        if not ctrl_pressed:  # Ctrlなしクリックならアクティブも移動
                            self.column_listbox.activate(idx)
                            self.column_listbox.see(idx)

                    # リストボックスの選択イベントを発火させて詳細表示などを更新
                    # ただし、これを呼ぶと on_column_select が呼ばれ、再度 highlight_selected_columns_on_orig -> redraw_boxes が発生する可能性
                    # → on_column_select 内で現在の選択と比較して不要な更新を避けるか、ここでは発火させない方が良いかも？
                    # → on_column_select を直接呼ぶ方が安全か？
                    self.on_column_select()  # 同期のために直接呼ぶ

            self.update_button_states()

    def handle_detail_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        """詳細表示Canvasがクリックされたときの処理"""
        # 選択状態はCanvasクラス内で管理されている
        num_selected = len(canvas.get_selected_tags())
        if num_selected == 1:
            tag = canvas.get_selected_tags()[0]
            try:
                idx = int(tag.split("_")[1])
                self.update_status(f"詳細表示: 文字 {idx} を選択中。")
            except (IndexError, ValueError):
                self.update_status("詳細表示: 選択された文字のインデックスを取得できませんでした。")
        elif num_selected > 1:
            self.update_status(f"詳細表示: {num_selected} 文字を選択中。")
        else:
            self.update_status("詳細表示: 文字選択なし。")

        # ボタンの状態だけ更新
        self.update_button_states()

    def delete_selected_column(self):
        if not self.selected_column_paths:
            self.update_status("削除する列が選択されていません。")
            return
        num_to_delete = len(self.selected_column_paths)
        col_names = [Path(p).name for p in self.selected_column_paths]
        confirm = messagebox.askyesno(
            "列削除確認",
            f"{num_to_delete}個の列を削除しますか？\n({', '.join(col_names[:5])}{'...' if num_to_delete > 5 else ''})\nこの操作は元に戻せません。",
            icon=messagebox.WARNING,
        )
        if confirm:
            self.update_status(f"{num_to_delete}個の列を削除中...")
            deleted_count = 0
            paths_to_delete = list(self.selected_column_paths)  # イテレート中にリストが変わるのを防ぐ
            for path in paths_to_delete:
                success = self.data_manager.delete_column(path)
                if success:
                    deleted_count += 1
            if deleted_count > 0:
                self.update_status(f"{deleted_count}個の列を削除しました。表示を更新します...")
                self.load_current_image_data()
                # messagebox.showinfo("削除完了", f"{deleted_count}個の列を削除しました。")
            else:
                self.update_status("列の削除に失敗しました。")
                self.update_button_states()
        else:
            self.update_status("列の削除をキャンセルしました。")

    def delete_selected_character(self):
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return

        try:
            char_indices = sorted([int(tag.split("_")[1]) for tag in selected_char_tags], reverse=True)  # 後ろから消す
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            self.update_status("エラー: 削除文字インデックス取得失敗。")
            return

        num_to_delete = len(char_indices)
        col_name = Path(self.current_detail_column_path).name
        confirm = messagebox.askyesnocancel(
            "文字削除確認",
            f"{num_to_delete}個の文字 (インデックス: {', '.join(map(str, sorted(char_indices)))}) を\n列 '{col_name}' から削除しますか？",
            icon=messagebox.WARNING,
        )
        if confirm is True:  # Yes
            self.update_status(f"{num_to_delete}個の文字を列 '{col_name}' から削除中...")
            deleted_count = 0
            current_col_path = self.current_detail_column_path  # 削除処理中にパスが変わる可能性があるため保持
            needs_reload = False
            error_occurred = False

            # 削除は1文字ずつ行う (DataManagerの実装依存)
            for _index_to_delete in char_indices:
                # 注意: delete_character は成功すると列を再生成し、パスが変わる可能性がある。
                #       そのため、複数文字削除は現状の実装では難しい。
                #       ここでは最初の1文字だけ削除し、リロードを促す。
                # → DataManager 側で複数インデックスを受け付けるように修正するか、
                #   ここでは1文字削除のみを許容するようにする。
                #   今回は、選択された最初のインデックス（ソート後なので最大のインデックス）のみ削除する。
                if len(char_indices) > 1:
                    messagebox.showinfo(
                        "情報",
                        "現在、文字の複数同時削除はサポートされていません。\n選択されたうちの1文字（インデックスが最大のもの）を削除します。",
                    )

                index_to_delete_single = char_indices[0]  # 最大インデックス
                self.update_status(f"文字 {index_to_delete_single} を削除中...")
                success = self.data_manager.delete_character(current_col_path, index_to_delete_single)
                if success:
                    deleted_count = 1
                    needs_reload = True  # 削除したら必ずリロード
                else:
                    # エラーメッセージは DataManager 内で表示されるはず
                    error_occurred = True
                    self.update_status(f"文字 {index_to_delete_single} の削除に失敗しました。")
                break  # 最初の1つを処理したらループを抜ける

            if needs_reload:
                self.update_status(f"{deleted_count}個の文字を削除しました。表示を更新します...")
                self.load_current_image_data()
                # messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。(画面更新)")
            elif deleted_count > 0:
                # ここには通常到達しないはず (needs_reload=Trueになるため)
                messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。")
            elif error_occurred:
                self.update_button_states()  # エラー後もボタン状態更新
            else:
                # 削除しなかった場合？
                self.update_status("文字削除処理で問題が発生しました。")

        elif confirm is False:  # No
            self.update_status("文字削除をキャンセルしました。")
        # Cancel の場合は何もしない

    def add_character(self):
        """文字追加ボタンのアクション"""
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            messagebox.showerror("エラー", "文字を追加する列を1つだけ選択してください。")
            self.update_status("文字追加: 列を1つ選択してください。")
            return

        selected_char_tags = self.detail_canvas.get_selected_tags()
        insert_after_index = -1  # デフォルトは先頭に追加

        if len(selected_char_tags) == 1:
            try:
                insert_after_index = int(selected_char_tags[0].split("_")[1])
                prompt_message = f"選択した文字 {insert_after_index} の後に追加する文字（またはU+XXXX）を入力:"
            except (IndexError, ValueError):
                messagebox.showerror("エラー", "選択された文字のインデックス取得に失敗しました。")
                self.update_status("エラー: 追加基準文字インデックス取得失敗。")
                return
        elif len(selected_char_tags) > 1:
            messagebox.showinfo(
                "情報",
                "文字を追加するには、挿入位置の基準となる文字を1つだけ選択するか、何も選択しないでください（先頭に追加されます）。",
            )
            self.update_status("文字追加: 挿入基準文字は1つまで。")
            return
        else:  # 何も選択されていない
            prompt_message = "列の先頭に追加する文字（またはU+XXXX）を入力:"

        char_or_unicode = simpledialog.askstring("文字追加", prompt_message, parent=self.root)

        if not char_or_unicode:
            self.update_status("文字追加をキャンセルしました。")
            return

        new_unicode_id = None
        if re.match(r"^U\+[0-9A-Fa-f]{4,}$", char_or_unicode.upper()):  # U+XXXX 形式 (4桁以上)
            new_unicode_id = char_or_unicode.upper()
        elif len(char_or_unicode) == 1:  # 1文字入力
            new_unicode_id = char_to_unicode_str(char_or_unicode)
            if new_unicode_id is None:  # 変換失敗 (通常は起こらない)
                messagebox.showerror("エラー", "文字からUnicode IDへの変換に失敗しました。")
                self.update_status("エラー: Unicode ID変換失敗。")
                return
        else:
            messagebox.showerror("エラー", "不正な入力です。1文字または U+XXXX 形式で入力してください。")
            self.update_status("エラー: 追加文字入力形式不正。")
            return

        col_name = Path(self.current_detail_column_path).name
        self.update_status(f"列 '{col_name}' に文字 '{char_or_unicode}' ({new_unicode_id}) を追加中...")
        success = self.data_manager.add_character(self.current_detail_column_path, new_unicode_id, insert_after_index)

        if success:
            self.update_status("文字追加が完了しました。表示を更新します...")
            self.load_current_image_data()
        else:
            self.update_status(f"列 '{col_name}' への文字追加に失敗しました。")
            self.update_button_states()

    # --- ナビゲーション ---
    def next_image(self):
        if not self.data_manager.original_images:
            return
        # 未保存の変更があれば保存を試みる
        if not self.check_unsaved_changes():
            self.update_status("次の画像への移動をキャンセルしました。")
            return  # 保存がキャンセルされた場合

        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        next_idx = (current_idx + 1) % len(self.data_manager.original_images)
        # self.save_all_changes() # check_unsaved_changes で処理済みのはず
        self.data_manager.set_current_original_image(self.data_manager.original_images[next_idx])
        self.load_current_image_data()

    def prev_image(self):
        if not self.data_manager.original_images:
            return
        # 未保存の変更があれば保存を試みる
        if not self.check_unsaved_changes():
            self.update_status("前の画像への移動をキャンセルしました。")
            return  # 保存がキャンセルされた場合

        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        prev_idx = (current_idx - 1 + len(self.data_manager.original_images)) % len(self.data_manager.original_images)
        # self.save_all_changes() # check_unsaved_changes で処理済みのはず
        self.data_manager.set_current_original_image(self.data_manager.original_images[prev_idx])
        self.load_current_image_data()

    def save_all_changes(self):
        """全ての変更をCSVに保存"""
        if self.data_manager.changes_made:
            self.update_status("変更を保存中...")
            if self.data_manager.save_changes():
                # 保存成功時は DataManager 内でメッセージ表示されるはず
                # messagebox.showinfo("保存完了", f"変更が {self.data_manager.output_csv_path.name} に保存されました。")
                self.update_status(f"変更が {self.data_manager.output_csv_path.name} に保存されました。")
            else:
                # 保存失敗時は DataManager 内でメッセージ表示されるはず
                self.update_status("変更の保存に失敗しました。")
        else:
            # messagebox.showinfo("情報", "保存すべき変更はありません。")
            self.update_status("保存すべき変更はありません。")

    def check_unsaved_changes(self):
        """
        未保存の変更があるか確認し、ユーザーに尋ねる。
        Returns:
            True: 操作を続行してよい (保存した or 破棄した or 変更なし)
            False: 操作をキャンセルすべき
        """
        if self.data_manager.changes_made:
            response = messagebox.askyesnocancel(
                "未保存の変更",
                "未保存の変更があります。保存しますか？\n"
                f"(保存先: {self.data_manager.output_csv_path.name})\n\n"
                "「はい」: 保存して続行\n"
                "「いいえ」: 変更を破棄して続行\n"
                "「キャンセル」: 操作を中断",
                icon=messagebox.WARNING,
                parent=self.root,  # 親ウィンドウ指定
            )
            if response is True:  # Yes
                self.update_status("未保存の変更を保存中...")
                if self.data_manager.save_changes():
                    self.update_status("変更を保存しました。")
                    return True
                else:
                    self.update_status("変更の保存に失敗しました。操作を中断します。")
                    return False  # 保存失敗時は中断
            elif response is False:  # No
                self.data_manager.changes_made = False  # 変更を破棄
                # 必要なら元の状態にロードし直す処理（現状は破棄のみ）
                self.update_status("未保存の変更を破棄しました。")
                # 再読み込みした方が安全かもしれない
                # self.data_manager.load_data() # 元のCSVを再読み込み
                # self.load_current_image_data()
                return True
            else:  # Cancel
                self.update_status("操作をキャンセルしました。")
                return False
        return True  # 変更がない場合は True

    def on_closing(self):
        """ウィンドウを閉じる際の処理"""
        if self.check_unsaved_changes():
            self.data_manager.save_last_state()  # 最後に表示した画像を保存
            self.update_status("アプリケーションを終了します。")
            self.root.destroy()
        else:
            self.update_status("終了をキャンセルしました。")


if __name__ == "__main__":
    root = tk.Tk()
    # root.withdraw() # 初期ウィンドウを隠す (ダイアログ表示のため)

    # --- データディレクトリの選択 ---
    script_dir = Path(__file__).resolve().parent  # スクリプトの絶対パスを取得
    default_data_dir = script_dir / "data"
    if not default_data_dir.exists():
        default_data_dir = script_dir.parent / "data"  # 親のdataも探す
    if not default_data_dir.exists():
        default_data_dir = script_dir  # スクリプトのディレクトリ

    initial_dir = str(default_data_dir)

    data_dir = filedialog.askdirectory(
        title="データディレクトリを選択してください (例: 'data' フォルダ)", initialdir=initial_dir
    )

    if not data_dir:
        print("データディレクトリが選択されませんでした。終了します。")
        messagebox.showerror("エラー", "データディレクトリが選択されませんでした。\nアプリケーションを終了します。")
        root.destroy()
    else:
        data_path = Path(data_dir).resolve()  # 絶対パスに変換
        # csv が存在するか簡易チェック
        input_csv = data_path / "processed" / "annotation.csv"
        output_csv = data_path / "processed" / "annotation_edited.csv"

        if not input_csv.exists() and not output_csv.exists():
            messagebox.showwarning(
                "確認",
                f"選択されたディレクトリに '{input_csv.name}' も\n"
                f"'{output_csv.name}' も見つかりません。\n"
                f"パス: {data_path / 'processed'}\n\n"
                "ツールは起動しますが、データは読み込めない可能性があります。",
                parent=root,
            )
        elif not input_csv.exists() and output_csv.exists():
            messagebox.showinfo(
                "情報",
                f"入力ファイル '{input_csv.name}' が見つかりません。\n"
                f"既存の編集済みファイル '{output_csv.name}' を読み込みます。",
                parent=root,
            )

        try:
            # root.deiconify() # メインウィンドウ表示
            app = AnnotatorApp(root, data_path)
            # AnnotatorApp の __init__ でエラーが発生した場合、すでに root.destroy() されている可能性がある
            if root.winfo_exists():
                root.mainloop()
        # except FileNotFoundError as e: # DataManagerの__init__内で処理されるはず
        #     messagebox.showerror("起動エラー", f"必要なファイルが見つかりません:\n{e}\nデータディレクトリ構造を確認してください。", parent=root)
        #     if root.winfo_exists(): root.destroy()
        except Exception as e:
            messagebox.showerror("起動エラー", f"予期せぬエラーが発生したため終了します:\n{e}", parent=root)
            traceback.print_exc()
            if root.winfo_exists():
                root.destroy()
