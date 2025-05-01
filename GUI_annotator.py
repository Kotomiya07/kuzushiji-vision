"""
GUIアノテーションツール
画面上で画像を表示し、ボックスを選択、編集、保存する機能を提供します。

操作説明:
- "次の画像": 次の画像を表示します。
- "前の画像": 前の画像を表示します。
- "保存": 現在の画像のボックス情報を保存します。
- "結合": 選択した列を結合します。
- "１点分割": 選択した文字から２つに分割します。
- "選択分割": 選択した文字とそれ以外の文字で分割します。
- "列削除": 選択した列を削除します。
- Ctrl + 左クリック: ボックスの選択/解除
- 中クリック: ズーム/パン
- ホイール: ズームイン/アウト
- 左クリック: ボックスの選択/解除

tips:
- 文字を複数選択している状態で、エンターキーを押すと、選択した文字のボックスが結合されます。
"""
import ast  # 文字列で保存されたリストを評価するため
import os
import re
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
  if not isinstance(unicode_str, str) or not re.match(r'^U\+[0-9A-Fa-f]+$', unicode_str):
      print(f"エラー: 不正な形式です。'U+XXXX' の形式で入力してください。入力値: {unicode_str}")
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
    print(f"エラー: コードポイントの変換に失敗しました。入力値: {unicode_str}")
    return None
  except Exception as e:
    # その他の予期せぬエラー
    print(f"予期せぬエラーが発生しました: {e}")
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
        #print(f"DEBUG: ImageCanvas.load_image called with path: {image_path}")  # DEBUG
        img = None  # Initialize img to None
        try:
            self.image_path = Path(image_path)
            #print(f"DEBUG: Opening image: {self.image_path}")  # DEBUG
            img = Image.open(self.image_path)
            #print(f"DEBUG: Image opened successfully. Type: {type(img)}")  # DEBUG
            #print(f"DEBUG: Converting image to RGBA...")  # DEBUG
            self.original_pil_image = img.convert("RGBA")  # RGBAで透明度対応
            #print(f"DEBUG: Image converted to RGBA successfully.")  # DEBUG

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
                        #print(f"DEBUG: Using parent size for initial scale calculation: {canvas_width}x{canvas_height}")
                    else:
                        #print("DEBUG: Canvas and parent size not available, using default size 600x600")
                        canvas_width = 600  # 仮のデフォルト値
                        canvas_height = 600  # 仮のデフォルト値
                except Exception:
                    #print("DEBUG: Error getting parent size, using default size 600x600")
                    canvas_width = 600  # 仮のデフォルト値
                    canvas_height = 600  # 仮のデフォルト値

            img_width, img_height = self.original_pil_image.size
            if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
                scale_w = canvas_width / img_width
                scale_h = canvas_height / img_height
                initial_scale = min(scale_w, scale_h, 1.0)  # 1.0 を超える拡大はしない
                #print(
                #    f"DEBUG: Calculated initial scale: {initial_scale} (canvas: {canvas_width}x{canvas_height}, image: {img_width}x{img_height})"
                #)
            else:
                initial_scale = 1.0  # 画像サイズ不正 or Canvasサイズ取れない場合

            self.scale = initial_scale  # 計算した初期スケールを設定
            # --- 初期スケール計算ここまで ---

            self.boxes = []
            self.selected_box_tags = set()
            self.box_id_map = {}
            #print(f"DEBUG: Calling display_image with initial scale {self.scale}...")  # DEBUG
            self.display_image()
            #print(f"DEBUG: display_image finished.")  # DEBUG
        except FileNotFoundError:
            self.clear_canvas()
            self.create_text(
                self.winfo_width() // 2,
                self.winfo_height() // 2,
                text=f"画像が見つかりません:\n{image_path}",
                fill="red",
                anchor="center",
            )
            self.original_pil_image = None
            #print(f"DEBUG: FileNotFoundError in load_image for path: {image_path}")  # DEBUG
            print(f"Error: Image not found at {image_path}")
        except Exception as e:
            self.clear_canvas()
            self.create_text(
                self.winfo_width() // 2,
                self.winfo_height() // 2,
                text=f"画像読み込みエラー:\n{e}",
                fill="red",
                anchor="center",
            )
            self.original_pil_image = None
            import traceback

            #print(f"DEBUG: Exception in load_image for path: {image_path}.")  # DEBUG
            #print(f"DEBUG: Exception type: {type(e)}")  # DEBUG
            #print(f"DEBUG: Exception message: {e}")  # DEBUG
            #print(f"DEBUG: Traceback:\n{traceback.format_exc()}")  # DEBUG
            print(f"Error loading image {image_path}: {e}")

    def display_image(self):
        #print(f"DEBUG: display_image started. self.original_pil_image is None: {self.original_pil_image is None}")  # DEBUG
        if self.original_pil_image is None:
            self.clear_canvas()
            #print(f"DEBUG: self.original_pil_image is None at the beginning of display_image. Clearing canvas.")  # DEBUG
            return

        # 古い描画要素 (ボックスとテキスト) のみを削除
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
                #print(f"DEBUG: Calling ImageTk.PhotoImage with self.pil_image...")  # DEBUG
                self.tk_image = ImageTk.PhotoImage(self.pil_image)
                #print(f"DEBUG: ImageTk.PhotoImage created successfully.")  # DEBUG
            except Exception as tk_err:
                import traceback

                #print(f"DEBUG: Error during ImageTk.PhotoImage creation!")  # DEBUG
                #print(f"DEBUG: Error type: {type(tk_err)}")  # DEBUG
                #print(f"DEBUG: Error message: {tk_err}")  # DEBUG
                #print(f"DEBUG: Traceback:\n{traceback.format_exc()}")  # DEBUG
                # エラーが発生した場合、tk_image は None のままになるか、不正な状態になる可能性がある
                self.tk_image = None  # 明示的に None に設定して後続処理でのエラーを防ぐ
                raise tk_err  # エラーを再送出して、外側の except Exception で捕捉させる
            self.create_image(0, 0, anchor="nw", image=self.tk_image, tags="image")
            #print(
            #    f"DEBUG: self.pil_image created. Type: {type(self.pil_image)}, Mode: {self.pil_image.mode}, Size: {self.pil_image.size}"
            #)  # DEBUG
            self.config(scrollregion=self.bbox("all"))

            # ボックスの再描画
            self.redraw_boxes()

        except ValueError as e:
            #print(
            #    f"Error during image resize or display: {e}. Original size: {self.original_pil_image.size}, Target size: ({width}, {height})"
            #)
            self.clear_canvas()  # エラー時はクリア
        except Exception as e:
            print(f"Unexpected error during image display: {e}")
            self.clear_canvas()

    def add_box(self, tag, x1, y1, x2, y2, color="red", width=2, text=None):
        """Canvas座標系でのボックス情報を追加"""
        self.boxes.append({"tag": tag, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "width": width, "text": text})
        self.redraw_boxes()  # すぐに描画

    def redraw_boxes(self):
        # 既存のボックスを削除
        self.delete("box")
        self.delete("box_text")
        self.box_id_map.clear()

        if not self.pil_image:
            return  # 画像がない場合は描画しない

        img_w, img_h = self.pil_image.size

        for box_info in self.boxes:
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
                tk_font = ("游ゴシック", font_size)

                self.create_text(
                    x2 + 20, y1 + 2, text=box_info["text"], anchor="nw", fill=color, tags=("box_text", tag), font=tk_font
                )

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
        if event.num == 4 or event.delta == 120:  # 上スクロール（拡大）
            delta = 1

        if delta != 0:
            factor = 1.1**delta
            new_scale = self.scale * factor
            # スケールの制限
            min_scale = 0.1
            max_scale = 5.0
            if min_scale <= new_scale <= max_scale:
                # カーソル位置中心にズーム
                cursor_x = self.canvasx(event.x)
                cursor_y = self.canvasy(event.y)
                self.scale = new_scale
                self.display_image()

                # ズーム後のスクロール位置調整
                new_canvas_x = cursor_x * factor
                new_canvas_y = cursor_y * factor
                delta_x = event.x - new_canvas_x
                delta_y = event.y - new_canvas_y
                self.scan_mark(0, 0)  # マーク位置をリセット
                self.scan_dragto(delta_x, delta_y, gain=1)  # 調整

    def on_left_click(self, event):
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)

        # Find all items with the 'box' tag
        all_box_items = self.find_withtag('box')
        clicked_box_tags = set()

        # Iterate through all box items to find which one(s) contain the click point
        for item_id in all_box_items:
            try:
                x1, y1, x2, y2 = self.coords(item_id)
                # Check if the click coordinates are within the item's bounds
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    if item_id in self.box_id_map:
                        clicked_box_tags.add(self.box_id_map[item_id])
            except Exception as e:
                # self.coords might fail if item is deleted concurrently? Unlikely but safe.
                print(f"Warning: Error getting coords for item {item_id}: {e}")

        # Check if Ctrl key was pressed during the click
        ctrl_pressed = bool(event.state & 0x0004) # Define ctrl_pressed here

        if not clicked_box_tags:
            # Clicked outside any box
            if not ctrl_pressed:
                # If Ctrl is not pressed, clear selection
                self.selected_box_tags.clear()
            # If Ctrl is pressed, do nothing (keep existing selection)
        else:
            # Clicked inside one or more boxes
            if ctrl_pressed:
                # Ctrl key is pressed: toggle selection state of clicked boxes
                for tag in clicked_box_tags:
                    if tag in self.selected_box_tags:
                        self.selected_box_tags.remove(tag)  # Deselect if already selected
                    else:
                        self.selected_box_tags.add(tag)  # Select if not selected
            else:
                # Ctrl key is not pressed:
                # If the click is on an already selected box, do nothing (allows dragging later if needed)
                # If the click is on a non-selected box, select *only* the clicked box(es)
                # Check if *any* of the clicked tags are *not* currently selected
                needs_new_selection = any(tag not in self.selected_box_tags for tag in clicked_box_tags)
                if needs_new_selection:
                    self.selected_box_tags = clicked_box_tags # Select only the clicked one(s)
                # If all clicked tags were already selected, keep the current selection

        # Redraw boxes to reflect selection changes
        self.redraw_boxes()

        # Notify the main application about the click event
        if hasattr(self.master, "on_canvas_click"):
            # Pass the canvas instance, the tags of boxes under the cursor, and the Ctrl key state
            self.master.on_canvas_click(self, clicked_box_tags, ctrl_pressed) # Now ctrl_pressed is defined

    def get_selected_tags(self):
        return list(self.selected_box_tags)


class DataManager:
    """CSVデータの読み込み、管理、編集、保存を行う"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.processed_dir = self.base_path / "processed"
        self.column_images_base_dir = self.processed_dir / "column_images"  # data_preprocessing.pyに合わせる
        #self.csv_path = self.processed_dir / "column_info.csv"
        self.csv_path = self.processed_dir / "f1_score_below_1.0.csv"
        self.last_state_path = self.base_path / ".last_image_path.txt" # 状態保存ファイル
        self.df = None
        self.original_images = []
        self.current_original_image = None
        self._last_loaded_image_path = self.load_last_state() # 最後に表示した画像を読み込む
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
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # データ型の修正 (CSV読み込みで文字列になるため)
        for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]:
            # ast.literal_evalが安全だが、データ形式によってはエラーになる可能性あり
            try:
                self.df[col] = self.df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse column '{col}'. Check data format. Error: {e}")
                # エラーの場合はそのままにするか、デフォルト値を入れる
                # self.df[col] = self.df[col].apply(lambda x: [] if isinstance(x, str) else x)

        # 'column_image' のパスを絶対パスに (または base_path からの相対パスのまま扱う)
        # ここでは base_path からの相対パスとして扱う前提で進める
        # self.df['column_image_abs'] = self.df['column_image'].apply(lambda p: self.base_path / p)

        # 元画像の一覧を取得
        if "column_image" in self.df.columns:
            processed_dir_name = self.processed_dir.name  # "processed"

            def fix_path(p):
                path_str = str(p).replace("\\", "/")
                # パスが "processed/" で始まっているかチェック
                if path_str.startswith(processed_dir_name + "/"):
                    # 先頭の "processed/" を除去
                    return path_str[len(processed_dir_name) + 1 :]
                return path_str  # それ以外はそのまま

            self.df["column_image"] = self.df["column_image"].apply(fix_path)
            print("Applied workaround path correction to 'column_image' column.")
        # --- 回避策ここまで ---

        if "original_image" in self.df.columns:
            self.original_images = self.df["original_image"].unique().tolist()
            self.original_images.sort()
            if self.original_images:
                # 最後に表示していた画像が存在すればそれを、なければ最初の画像を設定
                if self._last_loaded_image_path and self._last_loaded_image_path in self.original_images:
                    self.current_original_image = self._last_loaded_image_path
                    print(f"Resuming from last viewed image: {self.current_original_image}")
                else:
                    self.current_original_image = self.original_images[0]
            else:
                self.current_original_image = None # 画像リストが空の場合
        else:
            print("Warning: 'original_image' column not found in CSV.")
            self.original_images = []

        print(f"Loaded {len(self.df)} columns for {len(self.original_images)} original images.")

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
        return self.df[self.df["original_image"] == self.current_original_image].copy()

    def get_column_data(self, column_image_path_relative):
        """相対パスを使って列データを取得"""
        if self.df is None:
            return None
        # 完全一致で検索
        matches = self.df[self.df["column_image"] == column_image_path_relative]
        if not matches.empty:
            return matches.iloc[0].to_dict()  # Seriesを辞書に変換して返す
        else:
            print(f"Warning: Column data not found for relative path: {column_image_path_relative}")
            return None

    def get_column_abs_path(self, column_image_path_relative):
        """列画像の絶対パスを取得"""
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
                print(f"Warning: Absolute path specified in CSV does not exist: {potential_path}")
                # 存在しないパスオブジェクトを返す (呼び出し元でチェック)
                return potential_path

        # --- 相対パスの場合の解決ロジック ---
        # base_path (例: ユーザーが選択した 'data' ディレクトリ) の親をプロジェクトルートと仮定
        project_root = self.base_path.parent
        if not project_root:  # base_pathがルートディレクトリなどの場合
            project_root = Path.cwd()  # カレントディレクトリをルートとする

        # 2. プロジェクトルートからの相対パスとして試す (最有力候補)
        #    CSVの値が 'data/raw/dataset/...' の場合、プロジェクトルートに結合すれば正しいはず
        path_from_root = project_root / potential_path
        #print(f"DEBUG: Trying path from project root: {path_from_root}, Exists: {path_from_root.exists()}")  # DEBUG
        if path_from_root.exists():
            # #print(f"Debug: Resolved original image path (from project root): {path_from_root}")
            return path_from_root

        # 3. base_path (ユーザー選択の 'data' ディレクトリ) からの相対パスとして試す
        #    CSVの値が 'raw/dataset/...' のような形式の場合
        path_from_base = self.base_path / potential_path
        #print(f"DEBUG: Trying path from base path: {path_from_base}, Exists: {path_from_base.exists()}")  # DEBUG
        if path_from_base.exists():
            # #print(f"Debug: Resolved original image path (from base path): {path_from_base}")
            return path_from_base

        # 4. カレントワーキングディレクトリからの相対パスとして試す (最後の手段)
        path_from_cwd = Path.cwd() / potential_path
        #print(f"DEBUG: Trying path from CWD: {path_from_cwd}, Exists: {path_from_cwd.exists()}")  # DEBUG
        if path_from_cwd.exists():
            # #print(f"Debug: Resolved original image path (from CWD): {path_from_cwd}")
            return path_from_cwd

        # --- ここまでで見つからない場合 ---
        print(
            f"Warning: Could not determine absolute path for original image: '{original_image_path_str}'. \n"
            f"         Tried relative to project root ('{project_root}'), \n"
            f"         base path ('{self.base_path}'), \n"
            f"         and CWD ('{Path.cwd()}')."
        )
        # 存在しないパスを返すが、呼び出し元でチェックされることを期待
        #print(f"DEBUG: Returning path (may not exist): {path_from_root}")  # DEBUG
        # どの基準でのパスを返すか？ -> 最も可能性が高かった root からのパスを返す
        return path_from_root

    def _recalculate_column_bounds(self, char_boxes_in_orig):
        """元画像内の文字座標リストから列のバウンディングボックスを計算"""
        if not char_boxes_in_orig:
            return [0, 0, 0, 0]
        all_coords = np.array(char_boxes_in_orig)  # [[x1, y1, x2, y2], ...]
        x1 = np.min(all_coords[:, 0])
        y1 = np.min(all_coords[:, 1])
        x2 = np.max(all_coords[:, 2])
        y2 = np.max(all_coords[:, 3])
        return [x1, y1, x2, y2]

    def _get_char_boxes_in_original(self, column_data):
        """列データから元画像内の文字座標リストを取得"""
        col_x1, col_y1, _, _ = column_data["box_in_original"]
        char_boxes_in_col = column_data["char_boxes_in_column"]
        char_boxes_in_orig = []
        for box in char_boxes_in_col:
            cx1, cy1, cx2, cy2 = box
            orig_x1 = cx1 + col_x1
            orig_y1 = cy1 + col_y1
            orig_x2 = cx2 + col_x1
            orig_y2 = cy2 + col_y1
            char_boxes_in_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
        return char_boxes_in_orig

    def merge_columns(self, column_paths_to_merge):
        """複数の列を結合する"""
        if len(column_paths_to_merge) < 2:
            print("結合するには少なくとも2つの列を選択してください。")
            return False

        columns_data = []
        original_image_path = None
        for path_rel in column_paths_to_merge:
            data = self.get_column_data(path_rel)
            if data:
                columns_data.append(data)
                if original_image_path is None:
                    original_image_path = data["original_image"]
                elif original_image_path != data["original_image"]:
                    messagebox.showerror("エラー", "異なる元画像の列は結合できません。")
                    return False
            else:
                messagebox.showerror("エラー", f"列データが見つかりません: {path_rel}")
                return False

        # 新しい列の情報を計算
        all_char_boxes_in_orig = []
        all_unicode_ids = []
        for data in columns_data:
            char_boxes_orig = self._get_char_boxes_in_original(data)
            all_char_boxes_in_orig.extend(char_boxes_orig)
            all_unicode_ids.extend(data["unicode_ids"])

        if not all_char_boxes_in_orig:
            print("結合対象の列に文字が含まれていません。")
            return False

        # 文字をY座標でソート (元画像座標基準)
        sorted_indices = np.argsort([box[1] for box in all_char_boxes_in_orig])
        all_char_boxes_in_orig = [all_char_boxes_in_orig[i] for i in sorted_indices]
        all_unicode_ids = [all_unicode_ids[i] for i in sorted_indices]

        # 新しい列のバウンディングボックス (元画像上)
        new_col_box_orig = self._recalculate_column_bounds(all_char_boxes_in_orig)
        new_col_x1, new_col_y1, new_col_x2, new_col_y2 = new_col_box_orig

        # 新しい列画像内の文字座標 (相対座標)
        new_char_boxes_in_col = []
        for box_orig in all_char_boxes_in_orig:
            rel_x1 = box_orig[0] - new_col_x1
            rel_y1 = box_orig[1] - new_col_y1
            rel_x2 = box_orig[2] - new_col_x1
            rel_y2 = box_orig[3] - new_col_y1
            new_char_boxes_in_col.append([rel_x1, rel_y1, rel_x2, rel_y2])

        # --- 新しい列画像の生成 ---
        # 元画像を開く
        orig_img_abs_path = self.get_original_image_abs_path(original_image_path)
        try:
            orig_img = Image.open(orig_img_abs_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("エラー", f"元画像の読み込みに失敗しました: {orig_img_abs_path}\n{e}")
            return False

        # マージンを追加 (data_preprocessing.py と同様のロジックを適用すると良い)
        # ここでは固定マージンまたは簡易的な計算
        margin = 5  # 仮
        crop_x1 = max(0, int(new_col_x1 - margin))
        crop_y1 = max(0, int(new_col_y1 - margin))
        crop_x2 = min(orig_img.width, int(new_col_x2 + margin))
        crop_y2 = min(orig_img.height, int(new_col_y2 + margin))

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            messagebox.showerror("エラー", "計算された切り抜き領域が無効です。")
            return False

        new_column_image_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # --- 新しい列画像のパス決定と保存 ---
        # 元の列があったディレクトリ構造を利用するか、新しい命名規則を作る
        # ここでは最初の列のパスをベースに新しい名前を生成
        base_rel_path = Path(column_paths_to_merge[0])
        new_filename = f"{base_rel_path.stem}_merged_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
        # new_column_rel_path = base_rel_path.parent / new_filename
        # ↑だとディレクトリが深くなる可能性。column_images 直下などに保存する方が管理しやすいかも。
        # column_images/{doc_id}/{image_stem}/ の構造を維持する
        new_column_rel_path = base_rel_path.parent / new_filename

        new_column_abs_path = self.processed_dir / new_column_rel_path
        new_column_abs_path.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリ作成
        new_column_image_pil.save(new_column_abs_path, "JPEG")

        # --- DataFrameの更新 ---
        # 新しい行のデータを作成
        new_row_data = {
            "column_image": str(new_column_rel_path).replace("\\", "/"),  # pathlib -> str, Windowsパス対策
            "original_image": original_image_path,
            "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],  # マージン込みの座標
            "char_boxes_in_column": new_char_boxes_in_col,  # これはマージンなしの座標基準
            "unicode_ids": all_unicode_ids,
        }
        # TODO: ↑ char_boxes_in_column もマージンを考慮した座標に変換する必要がある
        new_char_boxes_in_cropped_col = []
        crop_offset_x = crop_x1
        crop_offset_y = crop_y1
        new_col_inner_x1 = new_col_x1 - crop_offset_x
        new_col_inner_y1 = new_col_y1 - crop_offset_y
        for box_orig in all_char_boxes_in_orig:
            rel_x1 = box_orig[0] - crop_offset_x
            rel_y1 = box_orig[1] - crop_offset_y
            rel_x2 = box_orig[2] - crop_offset_x
            rel_y2 = box_orig[3] - crop_offset_y
            new_char_boxes_in_cropped_col.append([rel_x1, rel_y1, rel_x2, rel_y2])
        new_row_data["char_boxes_in_column"] = new_char_boxes_in_cropped_col

        # 元の列の行を削除
        indices_to_drop = self.df[self.df["column_image"].isin(column_paths_to_merge)].index
        self.df = self.df.drop(indices_to_drop)

        # 新しい行を追加 (DataFrameに追加する方法はいくつかある)
        # self.df = self.df.append(new_row_data, ignore_index=True) # 古い方法
        new_row_df = pd.DataFrame([new_row_data])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # 元の列画像ファイルを削除 (オプション)
        for path_rel in column_paths_to_merge:
            try:
                abs_path_to_delete = self.get_column_abs_path(path_rel)
                if abs_path_to_delete.exists():
                    os.remove(abs_path_to_delete)
                    print(f"Deleted old column image: {abs_path_to_delete}")
            except Exception as e:
                print(f"Warning: Could not delete old column image {path_rel}: {e}")

        self.changes_made = True
        print(f"Columns merged into: {new_column_rel_path}")
        return True

    def split_column(self, column_path_to_split, split_index):
        """指定した文字インデックスの前で列を分割する"""
        column_data = self.get_column_data(column_path_to_split)
        if not column_data:
            messagebox.showerror("エラー", f"列データが見つかりません: {column_path_to_split}")
            return False

        char_boxes_col = column_data["char_boxes_in_column"]
        unicode_ids = column_data["unicode_ids"]
        original_image_path = column_data["original_image"]
        col_box_orig = column_data["box_in_original"]  # これはマージン込みの切り抜き座標

        if not 0 < split_index < len(unicode_ids):
            messagebox.showerror("エラー", "無効な分割位置です。")
            return False

        # --- 元画像上の絶対座標を取得 ---
        # マージン込みの切り抜き座標からのオフセットを計算
        crop_x1, crop_y1, _, _ = col_box_orig
        char_boxes_orig = []
        for box in char_boxes_col:
            orig_x1 = box[0] + crop_x1
            orig_y1 = box[1] + crop_y1
            orig_x2 = box[2] + crop_x1
            orig_y2 = box[3] + crop_y1
            char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])

        # 分割
        chars_orig1 = char_boxes_orig[:split_index]
        ids1 = unicode_ids[:split_index]
        chars_orig2 = char_boxes_orig[split_index:]
        ids2 = unicode_ids[split_index:]

        if not chars_orig1 or not chars_orig2:
            messagebox.showerror("エラー", "分割後の列が空になります。")
            return False

        # --- 新しい列情報を計算 & 画像生成 (関数化推奨) ---
        def create_new_column_from_chars(chars_in_orig, u_ids, orig_img_path, base_rel_path_str, suffix):
            if not chars_in_orig:
                return None, None

            # 1. 新しい列のバウンディングボックス (元画像上、マージンなし)
            new_col_bounds_no_margin = self._recalculate_column_bounds(chars_in_orig)
            nc_x1, nc_y1, nc_x2, nc_y2 = new_col_bounds_no_margin

            # 2. マージン追加と切り抜き座標計算
            orig_img = Image.open(self.get_original_image_abs_path(orig_img_path)).convert("RGB")
            margin = 5  # 仮
            crop_x1 = max(0, int(nc_x1 - margin))
            crop_y1 = max(0, int(nc_y1 - margin))
            crop_x2 = min(orig_img.width, int(nc_x2 + margin))
            crop_y2 = min(orig_img.height, int(nc_y2 + margin))
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                return None, None

            # 3. 列画像切り出し
            new_col_img_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # 4. 新しい列画像のパス決定と保存
            base_path = Path(base_rel_path_str)
            new_filename = f"{base_path.stem}_split{suffix}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
            new_col_rel_path = base_path.parent / new_filename
            new_col_abs_path = self.processed_dir / new_col_rel_path
            new_col_abs_path.parent.mkdir(parents=True, exist_ok=True)
            new_col_img_pil.save(new_col_abs_path, "JPEG")

            # 5. 列画像内の文字座標 (相対座標、マージン考慮)
            char_boxes_in_cropped = []
            for box_orig in chars_in_orig:
                rel_x1 = box_orig[0] - crop_x1
                rel_y1 = box_orig[1] - crop_y1
                rel_x2 = box_orig[2] - crop_x1
                rel_y2 = box_orig[3] - crop_y1
                char_boxes_in_cropped.append([rel_x1, rel_y1, rel_x2, rel_y2])

            # 6. 新しい行データ作成
            new_data = {
                "column_image": str(new_col_rel_path).replace("\\", "/"),
                "original_image": orig_img_path,
                "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],
                "char_boxes_in_column": char_boxes_in_cropped,
                "unicode_ids": u_ids,
            }
            return new_data, new_col_rel_path

        # 新しい列1の作成
        new_data1, new_path1 = create_new_column_from_chars(chars_orig1, ids1, original_image_path, column_path_to_split, "A")
        # 新しい列2の作成
        new_data2, new_path2 = create_new_column_from_chars(chars_orig2, ids2, original_image_path, column_path_to_split, "B")

        if new_data1 is None or new_data2 is None:
            messagebox.showerror("エラー", "分割後の列の生成に失敗しました。")
            # 作成途中のファイルがあれば削除処理を追加
            if new_path1:
                os.remove(self.processed_dir / new_path1)
            if new_path2:
                os.remove(self.processed_dir / new_path2)
            return False

        # --- DataFrameの更新 ---
        # 元の列を削除
        index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
        self.df = self.df.drop(index_to_drop)
        # 新しい列を追加
        new_rows_df = pd.DataFrame([new_data1, new_data2])
        self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

        # 元の列画像ファイルを削除 (オプション)
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted old column image: {abs_path_to_delete}")
        except Exception as e:
            print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")

        self.changes_made = True
        print(f"Column {column_path_to_split} split into {new_path1} and {new_path2}")
        return True

    def split_column_by_selection(self, column_path_to_split, selected_char_indices):
        """選択された文字とそれ以外で列を2つに分割する"""
        column_data = self.get_column_data(column_path_to_split)
        if not column_data:
            messagebox.showerror("エラー", f"列データが見つかりません: {column_path_to_split}")
            return False

        char_boxes_col = column_data["char_boxes_in_column"]
        unicode_ids = column_data["unicode_ids"]
        original_image_path = column_data["original_image"]
        col_box_orig_crop = column_data["box_in_original"] # マージン込み

        if not selected_char_indices:
            messagebox.showerror("エラー", "分割する文字が選択されていません。")
            return False
        if len(selected_char_indices) == len(unicode_ids):
             messagebox.showerror("エラー", "全ての文字が選択されています。分割できません。")
             return False

        # --- 元画像上の絶対座標を取得 ---
        crop_x1, crop_y1, _, _ = col_box_orig_crop
        char_boxes_orig = []
        for box in char_boxes_col:
            orig_x1 = box[0] + crop_x1
            orig_y1 = box[1] + crop_y1
            orig_x2 = box[2] + crop_x1
            orig_y2 = box[3] + crop_y1
            char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])

        # 選択された文字とそれ以外に分割
        selected_indices_set = set(selected_char_indices)
        chars_orig_selected = []
        ids_selected = []
        chars_orig_other = []
        ids_other = []

        for i, (box_orig, uid) in enumerate(zip(char_boxes_orig, unicode_ids)):
            if i in selected_indices_set:
                chars_orig_selected.append(box_orig)
                ids_selected.append(uid)
            else:
                chars_orig_other.append(box_orig)
                ids_other.append(uid)

        # Y座標でソート (元の順番を維持するため、分割後にソート)
        if chars_orig_selected:
            sorted_indices_sel = np.argsort([box[1] for box in chars_orig_selected])
            chars_orig_selected = [chars_orig_selected[i] for i in sorted_indices_sel]
            ids_selected = [ids_selected[i] for i in sorted_indices_sel]
        if chars_orig_other:
            sorted_indices_oth = np.argsort([box[1] for box in chars_orig_other])
            chars_orig_other = [chars_orig_other[i] for i in sorted_indices_oth]
            ids_other = [ids_other[i] for i in sorted_indices_oth]

        # --- 新しい列を生成 ---
        new_data_sel, new_path_sel = self._recreate_column_from_chars(
            chars_orig_selected, ids_selected, original_image_path, column_path_to_split, "_selA"
        )
        new_data_oth, new_path_oth = self._recreate_column_from_chars(
            chars_orig_other, ids_other, original_image_path, column_path_to_split, "_selB"
        )

        if new_data_sel is None or new_data_oth is None:
            messagebox.showerror("エラー", "選択分割後の列の生成に失敗しました。")
            # 作成途中のファイルを削除
            if new_path_sel: os.remove(self.processed_dir / new_path_sel)
            if new_path_oth: os.remove(self.processed_dir / new_path_oth)
            return False

        # --- DataFrameの更新 ---
        index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
        self.df = self.df.drop(index_to_drop)
        new_rows_df = pd.DataFrame([new_data_sel, new_data_oth])
        self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

        # --- 元の列画像ファイルを削除 ---
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted old column image: {abs_path_to_delete}")
        except Exception as e:
            print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")

        self.changes_made = True
        print(f"Column {column_path_to_split} split by selection into {new_path_sel} and {new_path_oth}")
        return True


    def move_characters(self, src_column_path, target_column_path, char_indices_to_move):
        """特定の文字をソース列からターゲット列へ移動"""
        if src_column_path == target_column_path:
            messagebox.showerror("エラー", "同じ列内で文字を移動することはできません（分割または結合を使用）。")
            return False
        if not char_indices_to_move:
            messagebox.showinfo("情報", "移動する文字が選択されていません。")
            return False

        src_data = self.get_column_data(src_column_path)
        tgt_data = self.get_column_data(target_column_path)

        if not src_data or not tgt_data:
            messagebox.showerror("エラー", "移動元または移動先の列データが見つかりません。")
            return False
        if src_data["original_image"] != tgt_data["original_image"]:
            messagebox.showerror("エラー", "異なる元画像の列間で文字を移動することはできません。")
            return False

        src_chars_col = src_data["char_boxes_in_column"]
        src_ids = src_data["unicode_ids"]
        src_crop_x1, src_crop_y1, _, _ = src_data["box_in_original"]
        tgt_crop_x1, tgt_crop_y1, _, _ = tgt_data["box_in_original"]

        moved_chars_orig = []
        moved_ids = []
        remaining_src_chars_orig = []
        remaining_src_ids = []

        # 移動対象と残すものを仕分け (元画像座標に変換しながら)
        src_indices_set = set(char_indices_to_move)
        for i, (box_col, uid) in enumerate(zip(src_chars_col, src_ids)):
            orig_x1 = box_col[0] + src_crop_x1
            orig_y1 = box_col[1] + src_crop_y1
            orig_x2 = box_col[2] + src_crop_x1
            orig_y2 = box_col[3] + src_crop_y1
            char_orig = [orig_x1, orig_y1, orig_x2, orig_y2]
            if i in src_indices_set:
                moved_chars_orig.append(char_orig)
                moved_ids.append(uid)
            else:
                remaining_src_chars_orig.append(char_orig)
                remaining_src_ids.append(uid)

        # 移動先の文字リスト (元画像座標)
        tgt_chars_col = tgt_data["char_boxes_in_column"]
        tgt_ids = tgt_data["unicode_ids"]
        tgt_chars_orig = []
        for box_col in tgt_chars_col:
            orig_x1 = box_col[0] + tgt_crop_x1
            orig_y1 = box_col[1] + tgt_crop_y1
            orig_x2 = box_col[2] + tgt_crop_x1
            orig_y2 = box_col[3] + tgt_crop_y1
            tgt_chars_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])

        # 移動対象をターゲットに追加し、Y座標でソート
        combined_tgt_chars_orig = tgt_chars_orig + moved_chars_orig
        combined_tgt_ids = tgt_ids + moved_ids
        sorted_indices = np.argsort([box[1] for box in combined_tgt_chars_orig])
        final_tgt_chars_orig = [combined_tgt_chars_orig[i] for i in sorted_indices]
        final_tgt_ids = [combined_tgt_ids[i] for i in sorted_indices]

        # --- 移動元と移動先の列を再生成 (共通ヘルパー関数を使用) ---
        # 移動元の新しい列データ (空になる場合もある)
        new_src_data, new_src_path = self._recreate_column_from_chars(
            remaining_src_chars_orig, remaining_src_ids, src_data["original_image"], src_column_path, "_move_src"
        )

        # 移動先の新しい列データ
        new_tgt_data, new_tgt_path = self._recreate_column_from_chars(
            final_tgt_chars_orig, final_tgt_ids, tgt_data["original_image"], target_column_path, "_move_tgt"
        )

        if new_tgt_data is None:  # 移動先が不正
            messagebox.showerror("エラー", "移動先の列の再生成に失敗しました。")
            # 作成途中のファイルを削除
            if new_src_path:
                os.remove(self.processed_dir / new_src_path)
            return False

        # --- DataFrame更新 ---
        indices_to_drop = self.df[
            (self.df["column_image"] == src_column_path) | (self.df["column_image"] == target_column_path)
        ].index
        self.df = self.df.drop(indices_to_drop)

        new_rows = []
        if new_src_data:  # 移動元が空でなければ追加
            new_rows.append(new_src_data)
        new_rows.append(new_tgt_data)  # 移動先は必ず追加

        new_rows_df = pd.DataFrame(new_rows)
        self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

        # --- 古い画像削除 ---
        try:
            os.remove(self.get_column_abs_path(src_column_path))
            print(f"Deleted old column image: {src_column_path}")
        except Exception as e:
            print(f"Warning: could not delete {src_column_path}: {e}")
        try:
            os.remove(self.get_column_abs_path(target_column_path))
            print(f"Deleted old column image: {target_column_path}")
        except Exception as e:
            print(f"Warning: could not delete {target_column_path}: {e}")

        self.changes_made = True
        print(f"Characters moved from {src_column_path} to {new_tgt_path}")
        return True

    def delete_column(self, column_path_to_delete):
        """列を削除する"""
        column_data = self.get_column_data(column_path_to_delete)
        if not column_data:
            messagebox.showerror("エラー", f"削除対象の列データが見つかりません: {column_path_to_delete}")
            return False

        # DataFrameから削除
        index_to_drop = self.df[self.df["column_image"] == column_path_to_delete].index
        self.df = self.df.drop(index_to_drop)

        # 画像ファイルを削除
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_delete)
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted column image: {abs_path_to_delete}")
            # ディレクトリが空になったら削除する (オプション)
            if not any(abs_path_to_delete.parent.iterdir()):
                shutil.rmtree(abs_path_to_delete.parent)
                print(f"Deleted empty directory: {abs_path_to_delete.parent}")

        except Exception as e:
            print(f"Warning: Could not delete column image file or dir {column_path_to_delete}: {e}")

        self.changes_made = True
        print(f"Deleted column: {column_path_to_delete}")
        return True

    def delete_character(self, column_path, char_index_to_delete):
        """列から指定した文字を削除する"""
        col_data = self.get_column_data(column_path)
        if not col_data:
            messagebox.showerror("Error", f"Column data not found: {column_path}")
            return False

        char_boxes_col = col_data["char_boxes_in_column"]
        unicode_ids = col_data["unicode_ids"]
        original_image_path = col_data["original_image"]
        col_box_orig_crop = col_data["box_in_original"]  # マージン込み

        if not 0 <= char_index_to_delete < len(unicode_ids):
            messagebox.showerror("Error", "Invalid character index to delete.")
            return False

        # 削除対象を除いたリストを作成
        new_char_boxes_col = [box for i, box in enumerate(char_boxes_col) if i != char_index_to_delete]
        new_unicode_ids = [uid for i, uid in enumerate(unicode_ids) if i != char_index_to_delete]

        if not new_unicode_ids:  # 全ての文字が削除された場合
            print(f"All characters deleted from {column_path}. Deleting column.")
            return self.delete_column(column_path)  # 列ごと削除

        # --- 元画像上の絶対座標に変換 ---
        crop_x1_orig, crop_y1_orig, _, _ = col_box_orig_crop
        new_char_boxes_orig = []
        for box_col in new_char_boxes_col:
            orig_x1 = box_col[0] + crop_x1_orig
            orig_y1 = box_col[1] + crop_y1_orig
            orig_x2 = box_col[2] + crop_x1_orig
            orig_y2 = box_col[3] + crop_y1_orig # ★以前の修正で crop_x1 -> crop_y1 になっているはずだが、念のため確認
            new_char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])

        # --- 列を再生成 (共通ヘルパー関数を使用) ---
        new_col_data, new_col_path = self._recreate_column_from_chars(
            new_char_boxes_orig, new_unicode_ids, original_image_path, column_path, "_chardel"
        )

        if new_col_data is None:
            messagebox.showerror("Error", "Failed to recreate column after deleting character.")
            return False

        # --- DataFrame Update ---
        index_to_drop = self.df[self.df["column_image"] == column_path].index
        self.df = self.df.drop(index_to_drop)
        new_row_df = pd.DataFrame([new_col_data])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # --- Delete old image ---
        try:
            os.remove(self.get_column_abs_path(column_path))
            print(f"Deleted old column image: {column_path}")
        except Exception as e:
            print(f"Warning: could not delete {column_path}: {e}")

        self.changes_made = True
        print(f"Character {char_index_to_delete} deleted from {column_path}, recreated as {new_col_path}")
        return True

    def _recreate_column_from_chars(self, chars_in_orig, u_ids, original_img_path, base_rel_path_str, suffix):
        """
        指定された元画像上の文字座標リストから新しい列データと列画像を生成する共通ヘルパー関数。
        Returns: (new_column_data_dict, new_column_relative_path) or (None, None) on failure.
        """
        if not chars_in_orig:
            return None, None

        # 1. 新しい列のバウンディングボックス (元画像上、マージンなし)
        new_col_bounds_no_margin = self._recalculate_column_bounds(chars_in_orig)
        nc_x1, nc_y1, nc_x2, nc_y2 = new_col_bounds_no_margin

        # 2. マージン追加と切り抜き座標計算
        try:
            orig_img_abs_path = self.get_original_image_abs_path(original_img_path)
            if not orig_img_abs_path or not orig_img_abs_path.exists():
                 raise FileNotFoundError(f"Original image not found: {orig_img_abs_path}")
            orig_img = Image.open(orig_img_abs_path).convert("RGB")
        except Exception as e:
            print(f"Error opening original image {original_img_path}: {e}")
            return None, None

        margin = 5 # TODO: 設定可能にするか、より賢いマージン計算を検討
        crop_x1 = max(0, int(nc_x1 - margin))
        crop_y1 = max(0, int(nc_y1 - margin))
        crop_x2 = min(orig_img.width, int(nc_x2 + margin))
        crop_y2 = min(orig_img.height, int(nc_y2 + margin))

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            print(f"Warning: Invalid crop region calculated for {base_rel_path_str}{suffix} ([{crop_x1},{crop_y1},{crop_x2},{crop_y2}])")
            return None, None

        # 3. 列画像切り出し
        try:
            new_col_img_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        except Exception as e:
            print(f"Error cropping image for {base_rel_path_str}{suffix}: {e}")
            return None, None

        # 4. 新しい列画像のパス決定と保存
        base_path = Path(base_rel_path_str)
        # タイムスタンプにマイクロ秒まで含めてファイル名の衝突を避ける
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
        new_filename = f"{base_path.stem}{suffix}_{timestamp}.jpg"
        new_col_rel_path = base_path.parent / new_filename
        new_col_abs_path = self.processed_dir / new_col_rel_path
        new_col_abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            new_col_img_pil.save(new_col_abs_path, "JPEG")
        except Exception as e:
            print(f"Error saving new column image {new_col_abs_path}: {e}")
            return None, None

        # 5. 列画像内の文字座標 (相対座標、マージン考慮)
        char_boxes_in_cropped = []
        for box_orig in chars_in_orig:
            rel_x1 = box_orig[0] - crop_x1
            rel_y1 = box_orig[1] - crop_y1
            rel_x2 = box_orig[2] - crop_x1
            rel_y2 = box_orig[3] - crop_y1 # ★タイポ修正済み
            char_boxes_in_cropped.append([rel_x1, rel_y1, rel_x2, rel_y2])

        # 6. 新しい行データ作成
        new_data = {
            "column_image": str(new_col_rel_path).replace("\\", "/"),
            "original_image": original_img_path,
            "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2], # マージン込みの切り抜き座標
            "char_boxes_in_column": char_boxes_in_cropped, # 切り出した画像内での相対座標
            "unicode_ids": u_ids,
        }
        return new_data, new_col_rel_path

    def save_changes(self):
        if not self.changes_made:
            #print("変更はありません。")
            return True

        # バックアップを作成 (オプション)
        # backup_path = self.csv_path.with_suffix(f".{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.bak")
        # try:
        #     shutil.copy2(self.csv_path, backup_path)
        #     print(f"バックアップを作成しました: {backup_path}")
        # except Exception as e:
        #     print(f"警告: バックアップの作成に失敗しました。 {e}")

        # CSVファイルに保存
        try:
            # listやdictが文字列にならないように注意 (通常は大丈夫なはず)
            self.df.to_csv(self.csv_path, index=False)
            self.changes_made = False
            #print(f"変更を保存しました: {self.csv_path}")
            return True
        except Exception as e:
            messagebox.showerror("保存エラー", f"CSVファイルへの保存中にエラーが発生しました:\n{e}")
            print(f"Error saving CSV: {e}")
            return False


class AnnotatorApp:
    def __init__(self, root, base_data_dir):
        self.root = root
        self.root.title("縦書き列アノテーション修正ツール")
        # ウィンドウサイズ調整
        self.root.geometry("1200x800")

        self.data_manager = DataManager(base_data_dir)

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

        self.prev_button = ttk.Button(nav_frame, text="<< 前の画像", command=self.prev_image)
        self.prev_button.pack(side="left", padx=5)
        self.image_label = ttk.Label(nav_frame, text="画像: ", width=60, anchor="w")  # 幅調整
        self.image_label.pack(side="left", padx=5, fill="x", expand=True) # ラベルは中央で伸縮
        self.next_button = ttk.Button(nav_frame, text="次の画像 >>", command=self.next_image)
        self.next_button.pack(side="left", padx=5) # 次へボタンはラベルの右

        # 右詰めの要素 (保存ボタンとページ情報)
        self.page_info_label = ttk.Label(nav_frame, text="- / -", width=10, anchor="e") # ページ情報ラベル (右寄せ)
        self.page_info_label.pack(side="right", padx=5)
        self.save_button = ttk.Button(nav_frame, text="変更を保存", command=self.save_all_changes)
        self.save_button.pack(side="right", padx=5) # 保存ボタンはページ情報の左

        # --- 左: 元画像表示 ---
        orig_frame = ttk.LabelFrame(main_frame, text="元画像と列範囲")
        orig_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        orig_frame.rowconfigure(0, weight=1)
        orig_frame.columnconfigure(0, weight=1)
        self.orig_canvas = ImageCanvas(orig_frame, bg="lightgrey")
        self.orig_canvas.grid(row=0, column=0, sticky="nsew")
        # スクロールバー追加 (オプション)
        # vsb_orig = ttk.Scrollbar(orig_frame, orient="vertical", command=self.orig_canvas.yview)
        # hsb_orig = ttk.Scrollbar(orig_frame, orient="horizontal", command=self.orig_canvas.xview)
        # self.orig_canvas.configure(yscrollcommand=vsb_orig.set, xscrollcommand=hsb_orig.set)
        # vsb_orig.grid(row=0, column=1, sticky='ns')
        # hsb_orig.grid(row=1, column=0, sticky='ew')

        # --- 中央: 列リストと操作 ---
        list_op_frame = ttk.Frame(main_frame)
        list_op_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        list_op_frame.rowconfigure(1, weight=1)  # リストボックスを拡大
        list_op_frame.columnconfigure(0, weight=1)

        op_buttons_frame = ttk.Frame(list_op_frame)
        op_buttons_frame.grid(row=0, column=0, sticky="ew", pady=5)

        self.merge_button = ttk.Button(op_buttons_frame, text="結合", command=self.merge_selected_columns)
        self.merge_button.pack(side="left", padx=2, fill="x", expand=True)
        self.split_button = ttk.Button(op_buttons_frame, text="1点分割", command=self.split_selected_column) # 名前変更
        self.split_button.pack(side="left", padx=2, fill="x", expand=True)
        self.split_selection_button = ttk.Button(op_buttons_frame, text="選択分割", command=self.split_column_by_selection) # ★追加
        self.split_selection_button.pack(side="left", padx=2, fill="x", expand=True) # ★追加
        self.move_char_button = ttk.Button(op_buttons_frame, text="文字移動", command=self.initiate_move_character)
        self.move_char_button.pack(side="left", padx=2, fill="x", expand=True)
        self.delete_col_button = ttk.Button(op_buttons_frame, text="列削除", command=self.delete_selected_column)
        self.delete_col_button.pack(side="left", padx=2, fill="x", expand=True)
        self.delete_char_button = ttk.Button(op_buttons_frame, text="文字削除", command=self.delete_selected_character)
        self.delete_char_button.pack(side="left", padx=2, fill="x", expand=True)

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
        # vsb_detail = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_canvas.yview)
        # hsb_detail = ttk.Scrollbar(detail_frame, orient="horizontal", command=self.detail_canvas.xview)
        # self.detail_canvas.configure(yscrollcommand=vsb_detail.set, xscrollcommand=hsb_detail.set)
        # vsb_detail.grid(row=0, column=1, sticky='ns')
        # hsb_detail.grid(row=1, column=0, sticky='ew')

        # --- 初期化 ---
        self.selected_column_paths = []  # Listboxで選択中の列の相対パス
        self.current_detail_column_path = None  # detail_canvas に表示中の列の相対パス
        self.moving_characters_info = None  # 文字移動中の情報 {src_path: str, char_indices: list}

        # Canvasクリックのコールバックを設定
        self.orig_canvas.master.on_canvas_click = self.handle_orig_canvas_click
        self.detail_canvas.master.on_canvas_click = self.handle_detail_canvas_click

        self.load_current_image_data()

        # ウィンドウを閉じる際の処理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- キーボードショートカット (アプリケーション全体にバインド) ---
        self.root.bind_all("<KeyPress>", self.handle_key_press)

    def handle_key_press(self, event):
        """キーボードショートカットを処理する"""
        # print(f"Key pressed: {event.keysym}") # デバッグ用
        if event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()
        elif event.keysym == "Return": # Enterキー
            # 複数列選択時のみ結合を実行
            if len(self.selected_column_paths) >= 2:
                self.merge_selected_columns()
        # 他のキーバインドもここに追加可能

    def load_current_image_data(self):
        """現在選択されている元画像のデータを読み込み、表示を更新する"""
        self.orig_canvas.clear_canvas()
        self.column_listbox.delete(0, tk.END)
        self.detail_canvas.clear_canvas()
        self.selected_column_paths = []
        self.current_detail_column_path = None
        self.moving_characters_info = None  # 移動状態をリセット

        current_image_path = self.data_manager.current_original_image
        all_images = self.data_manager.get_original_images()
        total_pages = len(all_images)

        if not current_image_path or not all_images:
            self.image_label.config(text="画像: (データなし)")
            self.page_info_label.config(text="0 / 0") # ページ情報も更新
            return
        else:
            try:
                current_page_index = all_images.index(current_image_path)
                current_page_num = current_page_index + 1
                self.page_info_label.config(text=f"{current_page_num} / {total_pages}")
            except ValueError:
                # current_image_path がリストにない場合 (通常は起こらないはず)
                self.image_label.config(text=f"画像: {current_image_path} (リストに不整合?)")
                self.page_info_label.config(text=f"? / {total_pages}")

        # ラベル更新 (パスが長すぎる場合があるので調整)
        label_path = Path(current_image_path)
        display_name = "/".join(label_path.parts[-4:])  # 例: raw/dataset/doc/img.jpg
        self.image_label.config(text=f"画像: {display_name}")
        # 元画像表示
        orig_abs_path = self.data_manager.get_original_image_abs_path(current_image_path)
        #print(
        #    f"DEBUG: Attempting to load original image from: {orig_abs_path}, Exists: {orig_abs_path.exists() if orig_abs_path else 'N/A'}"
        #)  # DEBUG
        if orig_abs_path and orig_abs_path.exists():
            #print(f"DEBUG: Original image found, calling orig_canvas.load_image with path: {orig_abs_path}")  # DEBUG
            self.orig_canvas.load_image(orig_abs_path)
        else:
            print(f"Error: Original image file not found or path is incorrect: {orig_abs_path}")
            self.orig_canvas.clear_canvas()
            #print(f"DEBUG: Original image not found in load_current_image_data for path: {orig_abs_path}")  # DEBUG
            # エラーメッセージ表示
            self.orig_canvas.create_text(
                self.orig_canvas.winfo_width() // 2,
                20,
                text=f"元画像が見つかりません:\n{orig_abs_path}",
                fill="red",
                anchor="n",
            )

        # 列データを取得してリストボックスと元画像上のボックス表示
        columns_df = self.data_manager.get_columns_for_current_image()
        if not columns_df.empty:
            # 列を左端のX座標でソートして表示順を決定
            columns_df["sort_key"] = columns_df["box_in_original"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else float("inf")
            )
            columns_df = columns_df.sort_values("sort_key")

            colors = ["orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
            for i, (_, row) in enumerate(columns_df.iterrows()):
                col_rel_path = row["column_image"]
                # リストボックスに追加 (表示名と実際のパスを保持)
                display_text = Path(col_rel_path).name  # ファイル名だけ表示
                self.column_listbox.insert(tk.END, display_text)
                self.column_listbox.itemconfig(tk.END, {"fg": "black"})  # 必要ならリセット

                # 元画像に列の範囲を描画
                col_box = row["box_in_original"]
                col_path_rel = row["column_image"]
                # ★座標形式チェックを追加
                if isinstance(col_box, list) and len(col_box) == 4 and all(isinstance(n, (int, float)) for n in col_box):
                    #print(f"DEBUG: Adding column box to orig_canvas: tag={col_path_rel}, box={col_box}")  # DEBUG
                    self.orig_canvas.add_box(
                        tag=col_path_rel,  # 列の相対パスをタグとして使用
                        x1=col_box[0],
                        y1=col_box[1],
                        x2=col_box[2],
                        y2=col_box[3],
                        color=colors[i % len(colors)],
                        width=1,  # ボックスの幅
                    )
                else:
                    print(f"Warning: Invalid or malformed column box format for {col_path_rel}: {col_box}. Skipping drawing.")

        # ボタンの状態更新
        self.update_button_states()

        # リストボックスの最初の項目を選択状態にする (項目があれば)
        if self.column_listbox.size() > 0:
            self.column_listbox.selection_clear(0, tk.END) # 念のため選択解除
            self.column_listbox.selection_set(0)
            self.column_listbox.selection_anchor(0) # ★範囲選択のアンカーも設定
            self.column_listbox.event_generate("<<ListboxSelect>>") # 選択イベントを発火

        # アイドルタスクを処理し、フォーカスを強制的に設定
        self.root.update_idletasks() # GUI更新を待機
        self.column_listbox.focus_force() # フォーカスを強制設定

    def on_column_select(self, event=None):
        """リストボックスで列が選択されたときの処理"""
        selected_indices = self.column_listbox.curselection()
        if not selected_indices:
            self.selected_column_paths = []
            self.detail_canvas.clear_canvas()
            self.current_detail_column_path = None
        else:
            # 対応する列の相対パスを取得
            all_columns_df = self.data_manager.get_columns_for_current_image()
            # 表示順にソートされているはずなので、インデックスで引ける
            all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else float("inf")
            )
            all_columns_df = all_columns_df.sort_values("sort_key")
            self.selected_column_paths = [all_columns_df.iloc[idx]["column_image"] for idx in selected_indices]

            # 最後に選択された列を詳細表示
            last_selected_idx = selected_indices[-1]
            self.current_detail_column_path = all_columns_df.iloc[last_selected_idx]["column_image"]
            self.load_detail_column(self.current_detail_column_path)

        # 元画像の選択列ハイライト更新
        self.highlight_selected_columns_on_orig()
        # ボタン状態更新
        self.update_button_states()

    def load_detail_column(self, column_rel_path):
        """指定された列を詳細表示エリアに表示"""
        self.detail_canvas.clear_canvas()
        column_data = self.data_manager.get_column_data(column_rel_path)
        if not column_data:
            print(f"Error: Cannot load detail, column data not found for {column_rel_path}")
            return

        col_abs_path = self.data_manager.get_column_abs_path(column_rel_path)
        if col_abs_path.exists():
            self.detail_canvas.load_image(col_abs_path)

            # 文字のボックスを描画
            if "char_boxes_in_column" in column_data and "unicode_ids" in column_data:
                char_boxes = column_data["char_boxes_in_column"]
                uids = column_data["unicode_ids"]
                if isinstance(char_boxes, list) and isinstance(uids, list) and len(char_boxes) == len(uids):
                    for i, (box, uid) in enumerate(zip(char_boxes, uids)):
                        if isinstance(box, list) and len(box) == 4:
                            x1, y1, x2, y2 = box
                            # タグとして (列パス, 文字インデックス) を使う (より詳細な識別のため)
                            char_tag = f"char_{i}"  # シンプルにインデックスのみ
                            self.detail_canvas.add_box(
                                tag=char_tag, x1=x1, y1=y1, x2=x2, y2=y2, color="green", width=1, text=f"{unicode_to_char(uid)}"
                            )
                        else:
                            print(f"Warning: Invalid char box format for index {i} in {column_rel_path}: {box}")
                else:
                    print(f"Warning: Mismatch or invalid format for char_boxes/unicode_ids in {column_rel_path}")

        else:
            print(f"Error: Column image file not found: {col_abs_path}")
            self.detail_canvas.create_text(
                self.detail_canvas.winfo_width() // 2,
                20,
                text=f"列画像が見つかりません:\n{col_abs_path}",
                fill="red",
                anchor="n",
            )

    def highlight_selected_columns_on_orig(self):
        """元画像上で選択中の列をハイライト"""
        if not hasattr(self.orig_canvas, "boxes"):
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

        # 結合: 2つ以上の列を選択
        self.merge_button.config(state=tk.NORMAL if num_selected_cols >= 2 else tk.DISABLED)
        # 分割: 1つの列を選択し、その詳細表示で1つの文字を選択 (分割点として)
        self.split_button.config(
            state=tk.NORMAL
            if num_selected_cols == 1 and num_selected_chars == 1 and self.current_detail_column_path
            else tk.DISABLED
        )
        # 選択分割: 1つの列を選択し、詳細表示で1つ以上かつ全部未満の文字を選択
        col_data = self.data_manager.get_column_data(self.current_detail_column_path) if self.current_detail_column_path else None
        total_chars = len(col_data.get("unicode_ids", [])) if col_data else 0
        self.split_selection_button.config(
            state=tk.NORMAL
            if num_selected_cols == 1 and self.current_detail_column_path and 0 < num_selected_chars < total_chars
            else tk.DISABLED
        )
        # 文字移動開始: 詳細表示で1つ以上の文字を選択
        self.move_char_button.config(
            state=tk.NORMAL if num_selected_chars > 0 and self.current_detail_column_path else tk.DISABLED
        )
        # 列削除: 1つ以上の列を選択
        self.delete_col_button.config(state=tk.NORMAL if num_selected_cols > 0 else tk.DISABLED)
        # 文字削除: 詳細表示で1つ以上の文字を選択
        self.delete_char_button.config(
            state=tk.NORMAL if num_selected_chars > 0 and self.current_detail_column_path else tk.DISABLED
        )

        # 移動中の状態表示
        if self.moving_characters_info:
            self.root.config(cursor="crosshair")  # カーソル変更
            # 必要ならステータスバー等でメッセージ表示
        else:
            self.root.config(cursor="")

    # --- ボタンアクション ---
    def merge_selected_columns(self):
        if len(self.selected_column_paths) < 2:
            return
        success = self.data_manager.merge_columns(self.selected_column_paths)
        if success:
            self.load_current_image_data()  # 表示更新

    def split_selected_column(self):
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if len(selected_char_tags) != 1:
            messagebox.showinfo("情報", "分割点を指定するため、詳細表示で文字を1つだけ選択してください。")
            return

        # タグ 'char_i' からインデックス i を抽出
        try:
            split_char_index = int(selected_char_tags[0].split("_")[1])
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            return

        # split_index は、選択した文字の前で分割するので、その文字のインデックスを渡す
        success = self.data_manager.split_column(self.current_detail_column_path, split_char_index)
        if success:
            self.load_current_image_data()

    def split_column_by_selection(self):
        """詳細表示で選択された文字に基づいて列を分割する"""
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            messagebox.showerror("エラー", "選択分割を行うには、列リストで列を1つだけ選択してください。")
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            messagebox.showerror("エラー", "分割する文字が選択されていません。詳細表示で文字をクリックして選択してください。")
            return

        # タグ 'char_i' からインデックス i を抽出
        try:
            selected_char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            selected_char_indices.sort() # 念のためソート
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            return

        # 全ての文字が選択されていないか確認
        col_data = self.data_manager.get_column_data(self.current_detail_column_path)
        if col_data and len(selected_char_indices) == len(col_data.get("unicode_ids", [])):
             messagebox.showerror("エラー", "全ての文字が選択されています。分割できません。")
             return

        confirm = messagebox.askyesno(
            "選択分割確認",
            f"列 '{Path(self.current_detail_column_path).name}' を、選択された {len(selected_char_indices)} 文字のグループと、\n"
            f"残りの文字のグループの2つに分割しますか？"
        )
        if confirm:
            success = self.data_manager.split_column_by_selection(self.current_detail_column_path, selected_char_indices)
            if success:
                self.load_current_image_data() # 表示更新

    def initiate_move_character(self):
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return

        try:
            # タグ 'char_i' からインデックス i を抽出
            char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            char_indices.sort()  # 念のためソート
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            return

        self.moving_characters_info = {"src_path": self.current_detail_column_path, "char_indices": char_indices}
        print(f"文字移動モード開始: {len(char_indices)}文字を選択中。移動先の列を元画像エリアでクリックしてください。")
        messagebox.showinfo(
            "文字移動", f"{len(char_indices)}文字を選択しました。\n移動先の列を「元画像エリア」でクリックしてください。"
        )
        self.update_button_states()  # カーソル変更など

    def handle_orig_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        """元画像Canvasがクリックされたときの処理 (主に文字移動先指定)"""
        if self.moving_characters_info and clicked_tags:
            # ボックスがクリックされたか確認 (タグが相対パス形式か)
            target_col_path = None
            for tag in clicked_tags:
                # クリックされたのが列のボックスか簡易判定 (パスっぽいか)
                if "/" in tag or "\\" in tag:
                    target_col_path = tag
                    break

            if target_col_path:
                print(f"移動先候補: {target_col_path}")
                src_path = self.moving_characters_info["src_path"]
                indices = self.moving_characters_info["char_indices"]

                if src_path == target_col_path:
                    messagebox.showinfo("情報", "同じ列には移動できません。")
                else:
                    # 確認ダイアログ
                    confirm = messagebox.askyesnocancel(
                        "文字移動確認",
                        f"{len(indices)}個の文字を\nFrom: {Path(src_path).name}\nTo:   {Path(target_col_path).name}\nに移動しますか？",
                    )
                    if confirm:
                        success = self.data_manager.move_characters(src_path, target_col_path, indices)
                        if success:
                            self.load_current_image_data()  # 成功したら更新
            # 移動モード解除
            self.moving_characters_info = None
            self.update_button_states()
            print("文字移動モード終了")

        elif not self.moving_characters_info:
            # 通常のクリック（複数選択など、必要なら実装）
            selected_tags_on_orig = canvas.get_selected_tags()
            # クリックされたタグとリストボックスの選択状態を同期させる
            all_columns_df = self.data_manager.get_columns_for_current_image()
            if not all_columns_df.empty:
                all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else float("inf")
                )
                all_columns_df = all_columns_df.sort_values("sort_key").reset_index()
                all_paths = all_columns_df["column_image"].tolist()

                # リストボックスの選択解除
                self.column_listbox.selection_clear(0, tk.END)
                self.selected_column_paths = []

                # Canvas上で選択されたタグに対応するインデックスをリストボックスで選択
                indices_to_select = []
                for tag in selected_tags_on_orig:
                    if tag in all_paths:
                        try:
                            idx = all_paths.index(tag)
                            indices_to_select.append(idx)
                            self.selected_column_paths.append(tag)
                        except ValueError:
                            pass  # 起こらないはずだが念のため
                for idx in indices_to_select:
                    self.column_listbox.selection_set(idx)

                # 最後に選択されたものを詳細表示 (リストボックスの選択イベントを発火させるのが楽)
                if indices_to_select:
                    self.column_listbox.event_generate("<<ListboxSelect>>")
                else:
                    # 何も選択されなかった場合
                    self.detail_canvas.clear_canvas()
                    self.current_detail_column_path = None

            self.update_button_states()

    def handle_detail_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        """詳細表示Canvasがクリックされたときの処理"""
        # print(f"Detail canvas clicked. Tags: {clicked_tags}")
        # 選択状態はCanvasクラス内で管理されている
        # ボタンの状態だけ更新
        self.update_button_states()

    def delete_selected_column(self):
        if not self.selected_column_paths:
            return
        confirm = messagebox.askyesno(
            "列削除確認", f"{len(self.selected_column_paths)}個の列を削除しますか？\nこの操作は元に戻せません。"
        )
        if confirm:
            deleted_count = 0
            for path in self.selected_column_paths:
                success = self.data_manager.delete_column(path)
                if success:
                    deleted_count += 1
            if deleted_count > 0:
                self.load_current_image_data()
            messagebox.showinfo("削除完了", f"{deleted_count}個の列を削除しました。")

    def delete_selected_character(self):
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return

        try:
            # タグ 'char_i' からインデックス i を抽出
            char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            char_indices.sort(reverse=True)  # 後ろから消さないとインデックスがずれる
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            return

        confirm = messagebox.askyesnocancel(
            "文字削除確認",
            f"{len(char_indices)}個の文字を列 '{Path(self.current_detail_column_path).name}' から削除しますか？",
        )
        if confirm:
            deleted_count = 0
            current_col_path = self.current_detail_column_path  # ループ中に変わる可能性があるので保持
            needs_reload = False
            # 削除は1文字ずつ行う必要がある（DataManagerの実装による）
            # もしDataManagerが複数インデックス同時削除に対応していれば修正
            for index_to_delete in char_indices:
                success = self.data_manager.delete_character(current_col_path, index_to_delete)
                if success:
                    deleted_count += 1
                    # 削除後、列のパスが変わる可能性があるため、次のループのために更新が必要
                    # ただし、現状の実装では1文字削除でも列が再生成されパスが変わる
                    # そのため、一度削除したら表示をリロードするのが安全
                    needs_reload = True
                    break  # 複数選択からの複数削除は一旦リロード必須とする
                else:
                    # エラーが発生したら中断
                    messagebox.showerror("削除エラー", f"文字インデックス {index_to_delete} の削除中にエラーが発生しました。")
                    break

            if needs_reload:
                self.load_current_image_data()
                messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。(画面更新)")
            elif deleted_count > 0:
                messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。")

    # --- ナビゲーション ---
    def next_image(self):
        if not self.data_manager.original_images:
            return
        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        next_idx = (current_idx + 1) % len(self.data_manager.original_images)
        self.save_all_changes() # 画像遷移前に自動保存
        self.data_manager.set_current_original_image(self.data_manager.original_images[next_idx])
        self.load_current_image_data()

    def prev_image(self):
        if not self.data_manager.original_images:
            return
        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        prev_idx = (current_idx - 1 + len(self.data_manager.original_images)) % len(self.data_manager.original_images)
        self.save_all_changes() # 画像遷移前に自動保存
        self.data_manager.set_current_original_image(self.data_manager.original_images[prev_idx])
        self.load_current_image_data()

    def save_all_changes(self):
        """全ての変更をCSVに保存"""
        if self.data_manager.changes_made:
            if self.data_manager.save_changes():
                #messagebox.showinfo("保存完了", "変更がCSVファイルに保存されました。")
                pass
            # 保存失敗時は DataManager 内でメッセージ表示されるはず
        else:
            #messagebox.showinfo("情報", "保存すべき変更はありません。")
            pass

    def check_unsaved_changes(self):
        """未保存の変更があるか確認し、ユーザーに尋ねる"""
        if self.data_manager.changes_made:
            response = messagebox.askyesnocancel(
                "未保存の変更",
                "未保存の変更があります。保存しますか？\n(「いいえ」で変更を破棄して続行、「キャンセル」で操作中断)",
            )
            if response is True:  # Yes
                return self.data_manager.save_changes()
            elif response is False:  # No
                self.data_manager.changes_made = False  # 変更を破棄
                # 必要なら元の状態にロードし直す処理（現状は破棄のみ）
                print("変更を破棄しました。")
                return True
            else:  # Cancel
                return False
        return True  # 変更がない場合は True

    def on_closing(self):
        """ウィンドウを閉じる際の処理"""
        if self.check_unsaved_changes():
            self.data_manager.save_last_state() # 最後に表示した画像を保存
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    # --- データディレクトリの選択 ---
    # カレントディレクトリを基準に 'data' を探す
    script_dir = Path(__file__).parent
    default_data_dir = script_dir / "data"
    if not default_data_dir.exists():
        # スクリプトの親ディレクトリなども探してみる
        default_data_dir = script_dir.parent / "data"

    initial_dir = str(default_data_dir) if default_data_dir.exists() else str(script_dir)

    data_dir = filedialog.askdirectory(
        title="データディレクトリを選択してください (例: 'data' フォルダ)", initialdir=initial_dir
    )

    if not data_dir:
        print("データディレクトリが選択されませんでした。終了します。")
        root.destroy()
    else:
        data_path = Path(data_dir)
        # processed/column_info.csv が存在するか簡易チェック
        if not (data_path / "processed" / "f1_score_below_1.0.csv").exists():
            messagebox.showwarning(
                "確認",
                f"選択されたディレクトリに 'processed/column_info.csv' が見つかりません。\nパス: {data_path / 'processed'}\n\nツールは起動しますが、データは読み込めません。",
            )
            # それでも起動は試みる

        try:
            app = AnnotatorApp(root, data_path)
            root.mainloop()
        except FileNotFoundError as e:
            messagebox.showerror(
                "起動エラー", f"必要なファイルが見つかりません:\n{e}\n\nデータディレクトリ構造を確認してください。"
            )
            root.destroy()
        except Exception as e:
            messagebox.showerror(
                "起動エラー", f"予期せぬエラーが発生しました:\n{e}"
            )  # 修正: インデントエラーを解消し、エラーメッセージ表示を追加
            root.destroy()  # エラー発生時はウィンドウを閉じる
