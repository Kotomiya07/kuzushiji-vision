"""
GUIアノテーションツール (変更版)
画面上で画像を表示し、ボックスを選択、編集、保存する機能を提供します。

変更点:
- エラー発生時のデータ整合性向上 (ファイル操作順序変更、エラーハンドリング強化)
- アノテーションは元画像ページごとに別ファイル (JSON Lines) に保存
- 文字(列)結合時、元の画像ファイルを backup フォルダに移動
- 文字ボックスと文字コードの追加機能実装 (右クリックから)
- ボックス内部クリックで選択可能に (fill='' を追加)

操作説明:
- "次の画像": 次の画像を表示します。(変更はページ別ファイルに自動保存)
- "前の画像": 前の画像を表示します。(変更はページ別ファイルに自動保存)
- "保存": 現在の変更をページ別ファイルに保存します。
- "結合": 選択した列を結合します。(元の列画像は backup へ)
- "１点分割": 選択した文字から２つに分割します。
- "選択分割": 選択した文字とそれ以外の文字で分割します。
- "列削除": 選択した列を削除します。
- "文字追加": 詳細表示エリアを右クリック -> Unicode入力 で文字を追加。
- "文字移動": 詳細表示エリアで文字選択 -> 文字移動ボタン -> 元画像エリアで移動先列をクリック。
- "文字削除": 詳細表示エリアで文字選択 -> 文字削除ボタン。
- Ctrl + 左クリック: ボックスの選択/解除
- 中クリック: ズーム/パン
- ホイール: ズームイン/アウト
- 左クリック: ボックスの枠線または内部をクリックして選択/解除
- 詳細エリア右クリック: 文字追加開始

tips:
- 文字を複数選択している状態で、エンターキーを押すと、選択した文字のボックスが結合されます。(この機能は列結合です)
"""
import ast  # 文字列で保存されたリストを評価するため
import json  # JSON Lines 保存のため
import os
import re
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk

# from skimage.transform import resize # 不要

CSV_PATH = Path("data/processed/100241706_annotations.csv")

# --- ヘルパー関数 (変更なし) ---
def unicode_to_char(unicode_str):
  """
  'U+XXXX' 形式のUnicodeコードポイント文字列を文字に変換します。
  (省略 - 元コードと同じ)
  """
  # 'U+'で始まり、その後に16進数が続く形式かチェック
  if not isinstance(unicode_str, str) or not re.match(r'^U\+[0-9A-Fa-f]+$', unicode_str):
      #print(f"エラー: 不正な形式です。'U+XXXX' の形式で入力してください。入力値: {unicode_str}")
      # アラート表示は add_character 内で行う
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
    #print(f"エラー: コードポイントの変換に失敗しました。入力値: {unicode_str}")
    return None
  except Exception as e:
    # その他の予期せぬエラー
    #print(f"予期せぬエラーが発生しました: {e}")
    return None


class ImageCanvas(tk.Canvas):
    """画像表示とインタラクション用Canvas (変更あり)"""

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
        # ★変更: 右クリックイベントを追加 (文字追加用)
        self.bind("<ButtonPress-3>", self.on_right_click) # 右クリック

    def load_image(self, image_path):
        # (省略 - 元コードと同じ、デバッグプリントは除去推奨)
        img = None
        try:
            self.image_path = Path(image_path)
            img = Image.open(self.image_path)
            self.original_pil_image = img.convert("RGBA")
            self.update_idletasks()
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                try:
                    parent_width = self.master.winfo_width()
                    parent_height = self.master.winfo_height()
                    if parent_width > 1 and parent_height > 1:
                        canvas_width = parent_width
                        canvas_height = parent_height
                    else: canvas_width, canvas_height = 600, 600
                except Exception: canvas_width, canvas_height = 600, 600
            img_width, img_height = self.original_pil_image.size
            if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
                scale_w = canvas_width / img_width
                scale_h = canvas_height / img_height
                initial_scale = min(scale_w, scale_h, 1.0)
            else: initial_scale = 1.0
            self.scale = initial_scale
            self.boxes = []
            self.selected_box_tags = set()
            self.box_id_map = {}
            self.display_image()
        except FileNotFoundError:
            self.clear_canvas()
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text=f"画像が見つかりません:\n{image_path}", fill="red", anchor="center")
            self.original_pil_image = None
            print(f"Error: Image not found at {image_path}")
        except Exception as e:
            self.clear_canvas()
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text=f"画像読み込みエラー:\n{e}", fill="red", anchor="center")
            self.original_pil_image = None
            import traceback
            print(f"Error loading image {image_path}: {e}\n{traceback.format_exc()}")

    def display_image(self):
        # (省略 - 元コードと同じ、デバッグプリントは除去推奨)
        if self.original_pil_image is None:
            self.clear_canvas()
            return
        self.delete("box")
        self.delete("box_text")
        width = int(self.original_pil_image.width * self.scale)
        height = int(self.original_pil_image.height * self.scale)
        max_dim = 3000
        if width > max_dim or height > max_dim:
            ratio = min(max_dim / width, max_dim / height)
            width = int(width * ratio)
            height = int(height * ratio)
            print(f"Warning: Resized large image to {width}x{height} for display.")
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid image dimensions after scaling ({width}x{height}). Cannot display.")
            return
        try:
            self.pil_image = self.original_pil_image.resize((width, height), Image.Resampling.BILINEAR)
            try:
                self.tk_image = ImageTk.PhotoImage(self.pil_image)
            except Exception as tk_err:
                import traceback
                print(f"DEBUG: Error during ImageTk.PhotoImage creation!\n{traceback.format_exc()}")
                self.tk_image = None
                raise tk_err
            self.create_image(0, 0, anchor="nw", image=self.tk_image, tags="image")
            self.config(scrollregion=self.bbox("all"))
            self.redraw_boxes()
        except ValueError as e:
            print(f"Error during image resize or display: {e}. Original size: {self.original_pil_image.size}, Target size: ({width}, {height})")
            self.clear_canvas()
        except Exception as e:
            print(f"Unexpected error during image display: {e}")
            self.clear_canvas()

    def add_box(self, tag, x1, y1, x2, y2, color="red", width=2, text=None):
        # (省略 - 元コードと同じ)
        self.boxes.append({"tag": tag, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "width": width, "text": text})
        self.redraw_boxes()

    def redraw_boxes(self):
        # (省略 - create_rectangle に fill='' を追加)
        self.delete("box")
        self.delete("box_text")
        self.box_id_map.clear()

        if not self.pil_image:
            return

        img_w, img_h = self.pil_image.size

        for box_info in self.boxes:
            x1 = box_info["x1"] * self.scale
            y1 = box_info["y1"] * self.scale
            x2 = box_info["x2"] * self.scale
            y2 = box_info["y2"] * self.scale

            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            min_size = 1
            if x2 - x1 < min_size or y2 - y1 < min_size:
                pass

            color = box_info["color"]
            width = box_info["width"]
            tag = box_info["tag"]

            if tag in self.selected_box_tags:
                color = "red"
                width = 3

            # ★変更: fill='' を追加して内部クリックを可能にする
            item_id = self.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=("box", tag), fill='')
            self.box_id_map[item_id] = tag

            if box_info["text"]:
                font_size = max(8, min(12, int((y2 - y1) * 0.8)))
                # tk_font = ("游ゴシック", font_size) # フォントが存在しない可能性
                tk_font = ("Arial", font_size) # より一般的なフォント

                # テキストの位置を微調整 (枠の外右上に)
                text_x = x2 + 5 * self.scale # 少し右に
                text_y = y1 # 上端に合わせる
                self.create_text(
                    text_x, text_y, text=box_info["text"], anchor="nw", fill=color, tags=("box_text", tag), font=tk_font
                )

    def clear_canvas(self):
        # (省略 - 元コードと同じ)
        self.delete("all")
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.original_pil_image = None
        self.boxes = []
        self.selected_box_tags = set()
        self.box_id_map = {}

    def on_button_press(self, event):
        # (省略 - 元コードと同じ)
        self.scan_mark(event.x, event.y)

    def on_move_press(self, event):
        # (省略 - 元コードと同じ)
        self.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event):
        # (省略 - 元コードと同じ)
        delta = 0
        if event.num == 5 or event.delta == -120: delta = -1
        if event.num == 4 or event.delta == 120: delta = 1

        if delta != 0:
            factor = 1.1**delta
            new_scale = self.scale * factor
            min_scale, max_scale = 0.1, 5.0
            if min_scale <= new_scale <= max_scale:
                cursor_x = self.canvasx(event.x)
                cursor_y = self.canvasy(event.y)
                self.scale = new_scale
                self.display_image()
                new_canvas_x = cursor_x * factor
                new_canvas_y = cursor_y * factor
                delta_x = event.x - new_canvas_x
                delta_y = event.y - new_canvas_y
                self.scan_mark(0, 0)
                self.scan_dragto(delta_x, delta_y, gain=1)

    def on_left_click(self, event):
        # (省略 - ロジックは元と同じ、fill=''により内部クリックが有効になるはず)
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        all_box_items = self.find_withtag('box')
        clicked_box_tags = set()
        for item_id in all_box_items:
            try:
                x1, y1, x2, y2 = self.coords(item_id)
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    if item_id in self.box_id_map:
                        clicked_box_tags.add(self.box_id_map[item_id])
            except Exception as e:
                print(f"Warning: Error getting coords for item {item_id}: {e}")
        ctrl_pressed = bool(event.state & 0x0004)
        if not clicked_box_tags:
            if not ctrl_pressed:
                self.selected_box_tags.clear()
        else:
            if ctrl_pressed:
                for tag in clicked_box_tags:
                    if tag in self.selected_box_tags:
                        self.selected_box_tags.remove(tag)
                    else:
                        self.selected_box_tags.add(tag)
            else:
                needs_new_selection = any(tag not in self.selected_box_tags for tag in clicked_box_tags)
                if needs_new_selection:
                    self.selected_box_tags = clicked_box_tags
        self.redraw_boxes()
        if hasattr(self.master, "on_canvas_click"):
            self.master.on_canvas_click(self, clicked_box_tags, ctrl_pressed)

    # ★変更: 右クリックハンドラ追加
    def on_right_click(self, event):
        """右クリックイベントを処理し、マスターに通知する"""
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        # print(f"Right click at ({event.x}, {event.y}) -> canvas ({canvas_x}, {canvas_y})") # Debug
        if hasattr(self.master, "on_canvas_right_click"):
            # キャンバス座標を渡す
            self.master.on_canvas_right_click(self, canvas_x, canvas_y)

    def get_selected_tags(self):
        # (省略 - 元コードと同じ)
        return list(self.selected_box_tags)


class DataManager:
    """CSVデータの読み込み、管理、編集、保存を行う (変更あり)"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.processed_dir = self.base_path / "processed"
        self.column_images_base_dir = self.processed_dir / "column_images"
        self.csv_path = CSV_PATH # 元データソース

        # --- ▼▼▼ 追加 ▼▼▼ ---
        # 変更を保存する新しいCSVファイルのパスを設定
        original_csv_path = Path(self.csv_path)
        modified_csv_filename = f"{original_csv_path.stem}_modified{original_csv_path.suffix}"
        # 元のCSVと同じディレクトリに保存する場合
        self.modified_csv_path = original_csv_path.parent / modified_csv_filename
        # processed ディレクトリに保存したい場合など、必要に応じて変更してください
        # self.modified_csv_path = self.processed_dir / modified_csv_filename
        print(f"変更後のアノテーションは '{self.modified_csv_path}' に保存されます。")
        # --- ▲▲▲ 追加 ▲▲▲ ---

        self.page_annotations_dir = self.processed_dir / "page_annotations" # JSONL用 (今回は未使用だが残す)
        self.page_annotations_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.processed_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.last_state_path = self.base_path / ".last_image_path.txt"
        self.df = None
        self.original_images = []
        self.current_original_image = None
        self._last_loaded_image_path = self.load_last_state()
        self.changed_image_paths = set()
        self.load_and_merge_data() # 元CSVと修正済みCSVをロード・マージするメソッドを呼ぶ

    @property
    def changes_made(self):
        """変更があったかどうかを判定するプロパティ"""
        return bool(self.changed_image_paths)

    def load_last_state(self):
        # (省略 - 元コードと同じ)
        if self.last_state_path.exists():
            try: return self.last_state_path.read_text().strip()
            except Exception as e: print(f"Warning: Failed to read last state file: {e}")
        return None

    def save_last_state(self):
        # (省略 - 元コードと同じ)
        if self.current_original_image:
            try: self.last_state_path.write_text(self.current_original_image)
            except Exception as e: print(f"Warning: Failed to save last state file: {e}")

    def load_and_merge_data(self):
        """元のCSVと、存在すれば修正済みCSVを読み込み、マージして self.df を作成する"""
        # 1. 元のCSVを読み込む
        if not self.csv_path.exists():
            print(f"Error: 元のCSVファイルが見つかりません: {self.csv_path}")
            # messagebox は GUI スレッドから呼ぶべきなのでここでは表示しない
            # messagebox.showerror("エラー", f"元のCSVファイルが見つかりません:\n{self.csv_path}")
            # 元のCSVがない場合は空のDataFrameで初期化するか、エラー終了するか選択
            self.df = pd.DataFrame() # 空で初期化する例
        else:
            try:
                # list/dictをパースする関数 (以前のload_dataから移動)
                def safe_literal_eval(x):
                    if isinstance(x, str):
                        if not x or x.strip() == '[]': return []
                        try: return ast.literal_eval(x)
                        except (ValueError, SyntaxError): return None
                    return x

                # read_csv 時に list/dict 列の converter を指定
                converters = {
                    col: safe_literal_eval
                    for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]
                }
                self.df = pd.read_csv(self.csv_path, converters=converters)

                # パス修正 (以前のload_dataから移動)
                if "column_image" in self.df.columns:
                    processed_dir_name = self.processed_dir.name
                    def fix_path(p):
                        path_str = str(p).replace("\\", "/")
                        if path_str.startswith(processed_dir_name + "/"):
                            return path_str[len(processed_dir_name) + 1 :]
                        return path_str
                    self.df["column_image"] = self.df["column_image"].apply(fix_path)

            except Exception as e:
                print(f"元のCSVファイルの読み込み中にエラーが発生しました: {e}")
                # messagebox.showerror("エラー", f"元のCSVファイルの読み込みに失敗しました:\n{self.csv_path}\n{e}")
                self.df = pd.DataFrame() # エラー時は空で初期化

        # 2. 修正済みCSVが存在すれば読み込む
        df_modified = None
        if self.modified_csv_path.exists():
            print(f"修正済みCSVを読み込みます: {self.modified_csv_path}")
            try:
                 # 修正済みCSVも同じコンバーターで読み込む
                converters_mod = {
                    col: safe_literal_eval
                    for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]
                }
                df_modified = pd.read_csv(self.modified_csv_path, converters=converters_mod)
            except Exception as e:
                print(f"修正済みCSVファイルの読み込み中にエラーが発生しました: {e}")
                # messagebox.showerror("エラー", f"修正済みCSVファイルの読み込みに失敗しました:\n{self.modified_csv_path}\n{e}")
                df_modified = None # 読み込み失敗

        # 3. マージ処理 (修正済みデータで元データを更新)
        if df_modified is not None and not df_modified.empty:
            print(f"元のデータ ({len(self.df)}行) を修正済みデータ ({len(df_modified)}行) で更新します。")
            # マージのキーとなる列を確認 (通常は original_image と column_image だが、column_image は変更される可能性あり)
            # ここでは original_image をベースに、修正済みファイルにあるページはごっそり置き換える方針とする
            # (より厳密には column_image もキーにすべきだが、列操作で column_image は変化するため難しい)

            # 修正済みファイルに含まれる original_image のリストを取得
            modified_original_images = df_modified['original_image'].unique()

            # 元のDataFrameから、修正済みファイルに含まれるページのデータを削除
            df_base_filtered = self.df[~self.df['original_image'].isin(modified_original_images)]

            # フィルタリングされた元のデータと、修正済みデータを結合
            self.df = pd.concat([df_base_filtered, df_modified], ignore_index=True)
            print(f"マージ後のデータ: {len(self.df)}行")


        # --- 共通の後処理 (以前のload_dataから移動) ---
        # 元画像リスト取得など
        if self.df is not None and not self.df.empty and "original_image" in self.df.columns:
            self.original_images = self.df["original_image"].dropna().unique().tolist()
            self.original_images.sort()
            if self.original_images:
                if self._last_loaded_image_path and self._last_loaded_image_path in self.original_images:
                    self.current_original_image = self._last_loaded_image_path
                else:
                    self.current_original_image = self.original_images[0]
            else:
                self.current_original_image = None
        else:
            print("Warning: 'original_image' column not found or DataFrame is empty.")
            self.original_images = []
            self.current_original_image = None

        # parse_error_cols のチェック (必要なら復活させる)
        # ...

        print(f"ロード/マージ完了。{len(self.df)}件の列データ、{len(self.original_images)}枚の元画像。")
        self.changed_image_paths = set() # ロード直後は変更なし


    def load_data(self):
        # (省略 - 元コードと同じ、エラーハンドリングを少し丁寧にする)
        if not self.csv_path.exists():
            # messagebox は GUI スレッドから呼ぶべきなので、ここでは print と例外送出
            print(f"Error: CSV file not found: {self.csv_path}")
            raise FileNotFoundError(f"CSVファイルが見つかりません: {self.csv_path}")

        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise IOError(f"CSVファイルの読み込みに失敗しました: {self.csv_path}\n{e}")

        # データ型の修正
        parse_error_cols = []
        for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]:
            if col in self.df.columns:
                try:
                    # ast.literal_eval は安全だが、不正な形式に弱い
                    # 空文字列や '[]' 以外の不正な形式を None や [] に変換する試み
                    def safe_literal_eval(x):
                        if isinstance(x, str):
                            if not x or x.strip() == '[]':
                                return []
                            try:
                                return ast.literal_eval(x)
                            except (ValueError, SyntaxError):
                                return None # パース失敗
                        return x # 文字列以外はそのまま
                    self.df[col] = self.df[col].apply(safe_literal_eval)
                    # パース失敗した行がないかチェック
                    if self.df[col].isnull().any():
                        parse_error_cols.append(col)
                except Exception as e:
                    print(f"Warning: Error processing column '{col}'. Check data format. Error: {e}")
                    parse_error_cols.append(col)
            else:
                 print(f"Warning: Expected column '{col}' not found in CSV.")

        if parse_error_cols:
             print(f"Warning: Could not parse list/dict format in columns: {parse_error_cols}. Some data might be missing or incorrect.")
             # 必要ならここでエラー終了させるか、該当行を削除するなどの処理を追加

        # パス修正 (元コードのまま)
        if "column_image" in self.df.columns:
            processed_dir_name = self.processed_dir.name
            def fix_path(p):
                path_str = str(p).replace("\\", "/")
                if path_str.startswith(processed_dir_name + "/"):
                    return path_str[len(processed_dir_name) + 1 :]
                return path_str
            self.df["column_image"] = self.df["column_image"].apply(fix_path)

        # 元画像リスト取得
        if "original_image" in self.df.columns:
            # NaN や None を除去してから unique() を呼ぶ
            self.original_images = self.df["original_image"].dropna().unique().tolist()
            self.original_images.sort()
            if self.original_images:
                if self._last_loaded_image_path and self._last_loaded_image_path in self.original_images:
                    self.current_original_image = self._last_loaded_image_path
                else:
                    self.current_original_image = self.original_images[0]
            else:
                self.current_original_image = None
        else:
            print("Warning: 'original_image' column not found in CSV.")
            self.original_images = []
            self.current_original_image = None

        print(f"Loaded {len(self.df)} columns for {len(self.original_images)} original images from {self.csv_path}")
        self.changed_image_paths = set() # ロード直後は変更なし

    def get_original_images(self):
        # (省略 - 元コードと同じ)
        return self.original_images

    def set_current_original_image(self, image_path):
        # (省略 - 元コードと同じ)
        if image_path in self.original_images:
            self.current_original_image = image_path
            return True
        return False

    def get_columns_for_current_image(self):
        # (省略 - 元コードと同じ)
        if self.current_original_image is None or self.df is None:
            return pd.DataFrame()
        # current_original_image が NaN でないことを確認してから比較
        if pd.isna(self.current_original_image):
            return pd.DataFrame()
        return self.df[self.df["original_image"] == self.current_original_image].copy()


    def get_column_data(self, column_image_path_relative):
        # (省略 - 元コードと同じ)
        if self.df is None: return None
        matches = self.df[self.df["column_image"] == column_image_path_relative]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        else:
            # print(f"Warning: Column data not found for relative path: {column_image_path_relative}") # 頻繁に出る可能性があるのでコメントアウト
            return None

    def get_column_abs_path(self, column_image_path_relative):
        # (省略 - 元コードと同じ)
        return self.processed_dir / column_image_path_relative

    def get_original_image_abs_path(self, original_image_path_str):
        # (省略 - 元コードと同じ、デバッグプリント除去)
        if not original_image_path_str or not isinstance(original_image_path_str, str): return None
        potential_path = Path(original_image_path_str)
        if potential_path.is_absolute():
            return potential_path # 存在チェックは呼び出し元で行う想定に変更
        project_root = self.base_path.parent if self.base_path.parent else Path.cwd()
        path_from_root = project_root / potential_path
        if path_from_root.exists(): return path_from_root
        path_from_base = self.base_path / potential_path
        if path_from_base.exists(): return path_from_base
        path_from_cwd = Path.cwd() / potential_path
        if path_from_cwd.exists(): return path_from_cwd
        # print(f"Warning: Could not determine absolute path for original image: '{original_image_path_str}'. Returning best guess: {path_from_root}")
        return path_from_root # 見つからなくても推測パスを返す

    def _recalculate_column_bounds(self, char_boxes_in_orig):
        # (省略 - 元コードと同じ)
        if not char_boxes_in_orig: return [0, 0, 0, 0]
        try:
            all_coords = np.array(char_boxes_in_orig)
            if all_coords.shape[1] != 4: return [0, 0, 0, 0] # 不正な形式
            x1 = np.min(all_coords[:, 0])
            y1 = np.min(all_coords[:, 1])
            x2 = np.max(all_coords[:, 2])
            y2 = np.max(all_coords[:, 3])
            return [int(x1), int(y1), int(x2), int(y2)] # 整数で返す
        except Exception:
            return [0, 0, 0, 0] # 計算失敗

    def _get_char_boxes_in_original(self, column_data):
        # (省略 - 元コードと同じ、型チェック追加)
        col_box = column_data.get("box_in_original")
        char_boxes_col = column_data.get("char_boxes_in_column")
        if not isinstance(col_box, list) or len(col_box) < 2 or not isinstance(char_boxes_col, list):
            print(f"Warning: Invalid format for box_in_original or char_boxes_in_column in get_char_boxes_in_original for {column_data.get('column_image')}")
            return []
        col_x1, col_y1 = col_box[0], col_box[1]
        char_boxes_in_orig = []
        for box in char_boxes_col:
            if isinstance(box, list) and len(box) == 4:
                cx1, cy1, cx2, cy2 = box
                orig_x1 = cx1 + col_x1
                orig_y1 = cy1 + col_y1
                orig_x2 = cx2 + col_x1
                orig_y2 = cy2 + col_y1
                char_boxes_in_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
            else:
                # 不正な形式のボックスはスキップ
                 print(f"Warning: Skipping invalid char box format {box} in {column_data.get('column_image')}")
        return char_boxes_in_orig

    def merge_columns(self, column_paths_to_merge):
        # (省略 - ファイル削除をバックアップ移動に変更、エラーハンドリング強化)
        if len(column_paths_to_merge) < 2:
            messagebox.showerror("エラー", "結合するには少なくとも2つの列を選択してください。")
            return False

        columns_data = []
        original_image_path = None
        for path_rel in column_paths_to_merge:
            data = self.get_column_data(path_rel)
            if data:
                columns_data.append(data)
                current_orig_img = data.get("original_image")
                if current_orig_img is None:
                    messagebox.showerror("エラー", f"列データに元画像パスがありません: {path_rel}")
                    return False
                if original_image_path is None:
                    original_image_path = current_orig_img
                elif original_image_path != current_orig_img:
                    messagebox.showerror("エラー", "異なる元画像の列は結合できません。")
                    return False
            else:
                messagebox.showerror("エラー", f"列データが見つかりません: {path_rel}")
                return False

        all_char_boxes_in_orig = []
        all_unicode_ids = []
        parse_error = False
        for data in columns_data:
            char_boxes_orig = self._get_char_boxes_in_original(data)
            uids = data.get("unicode_ids")
            if not isinstance(uids, list) or len(char_boxes_orig) != len(uids):
                messagebox.showerror("エラー", f"文字ボックスとUnicode IDの数または形式が一致しません: {data.get('column_image')}")
                parse_error = True
                break
            all_char_boxes_in_orig.extend(char_boxes_orig)
            all_unicode_ids.extend(uids)
        if parse_error: return False

        if not all_char_boxes_in_orig:
            messagebox.showinfo("情報", "結合対象の列に文字が含まれていません。")
            return False

        # Y座標でソート
        try:
            sorted_indices = np.argsort([box[1] for box in all_char_boxes_in_orig])
            all_char_boxes_in_orig = [all_char_boxes_in_orig[i] for i in sorted_indices]
            all_unicode_ids = [all_unicode_ids[i] for i in sorted_indices]
        except IndexError:
             messagebox.showerror("エラー", "文字座標のソート中にエラーが発生しました。データ形式を確認してください。")
             return False

        # ★変更: 新しい列データ生成を _recreate_column_from_chars に任せる
        base_rel_path_str = column_paths_to_merge[0] # 新ファイル名のベース
        suffix = "_merged"
        new_column_data, new_column_rel_path = self._recreate_column_from_chars(
            all_char_boxes_in_orig, all_unicode_ids, original_image_path, base_rel_path_str, suffix
        )

        if new_column_data is None:
            # _recreate_column_from_chars 内でエラーメッセージ表示されるはず
            return False

        # --- DataFrameの更新 (エラー発生しないように最後に実行) ---
        try:
            indices_to_drop = self.df[self.df["column_image"].isin(column_paths_to_merge)].index
            self.df = self.df.drop(indices_to_drop).reset_index(drop=True) # インデックス再構築

            new_row_df = pd.DataFrame([new_column_data])
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        except Exception as e:
             messagebox.showerror("エラー", f"DataFrameの更新中にエラーが発生しました: {e}")
             # ★変更: 失敗した場合、生成した新しいファイルを削除すべきか？ -> ここではしないでおく
             return False

        # --- ★変更: 元の列画像ファイルを backup ディレクトリに移動 ---
        backup_paths = []
        move_errors = []
        for path_rel in column_paths_to_merge:
            src_abs_path = self.get_column_abs_path(path_rel)
            # backup ディレクトリ内に元の相対パス構造を再現
            backup_dest_path = self.backup_dir / path_rel
            backup_dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if src_abs_path.exists():
                    shutil.move(str(src_abs_path), str(backup_dest_path))
                    backup_paths.append(str(backup_dest_path))
                    # 元のディレクトリが空になったら削除 (オプション)
                    # if not any(src_abs_path.parent.iterdir()):
                    #     try: shutil.rmtree(src_abs_path.parent)
                    #     except OSError: pass # 無視
            except Exception as e:
                move_errors.append(f"Could not move {path_rel} to backup: {e}")

        if move_errors:
             print("Warning: Errors occurred during backup file moving:\n" + "\n".join(move_errors))
             # ここで処理を中断すべきか？ -> 続行する

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Columns {column_paths_to_merge} merged into: {new_column_rel_path}")
        if backup_paths:
            print(f"Moved original column images to backup: {len(backup_paths)} files.")
        return True

    def split_column(self, column_path_to_split, split_index):
        # (省略 - エラーハンドリング強化)
        column_data = self.get_column_data(column_path_to_split)
        if not column_data:
            messagebox.showerror("エラー", f"列データが見つかりません: {column_path_to_split}")
            return False

        char_boxes_col = column_data.get("char_boxes_in_column")
        unicode_ids = column_data.get("unicode_ids")
        original_image_path = column_data.get("original_image")
        col_box_orig_crop = column_data.get("box_in_original") # マージン込み

        if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list) \
           or not isinstance(original_image_path, str) or not isinstance(col_box_orig_crop, list):
            messagebox.showerror("エラー", f"列データの形式が無効です: {column_path_to_split}")
            return False

        if not 0 < split_index < len(unicode_ids):
            messagebox.showerror("エラー", "無効な分割位置です。")
            return False

        # 元画像上の絶対座標を取得
        try:
            crop_x1, crop_y1, _, _ = col_box_orig_crop
            char_boxes_orig = []
            for box in char_boxes_col:
                orig_x1 = box[0] + crop_x1; orig_y1 = box[1] + crop_y1
                orig_x2 = box[2] + crop_x1; orig_y2 = box[3] + crop_y1
                char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
        except Exception as e:
             messagebox.showerror("エラー", f"文字座標の計算中にエラーが発生しました: {e}")
             return False

        # 分割
        chars_orig1 = char_boxes_orig[:split_index]
        ids1 = unicode_ids[:split_index]
        chars_orig2 = char_boxes_orig[split_index:]
        ids2 = unicode_ids[split_index:]

        if not chars_orig1 or not chars_orig2:
            messagebox.showerror("エラー", "分割後の列が空になります。")
            return False

        # --- 新しい列を生成 ---
        new_data1, new_path1 = self._recreate_column_from_chars(
            chars_orig1, ids1, original_image_path, column_path_to_split, "_splitA"
        )
        # new_data1 が None の場合、ここで return False する前に path1 を削除する必要はない
        # _recreate_column_from_chars が失敗した場合、ファイルは作られないか、作られてもパスは返さない想定
        if new_data1 is None: return False

        new_data2, new_path2 = self._recreate_column_from_chars(
            chars_orig2, ids2, original_image_path, column_path_to_split, "_splitB"
        )
        if new_data2 is None:
            # 失敗した場合、成功した最初の列のファイルを削除するロールバック処理が必要
            if new_path1:
                try: os.remove(self.processed_dir / new_path1)
                except OSError as e: print(f"Warning: Could not remove partially created file {new_path1}: {e}")
            return False

        # --- DataFrameの更新 (最後に実行) ---
        try:
            index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            new_rows_df = pd.DataFrame([new_data1, new_data2])
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)
        except Exception as e:
            messagebox.showerror("エラー", f"DataFrameの更新中にエラーが発生しました: {e}")
            # ロールバック: 作成したファイルを削除
            if new_path1:
                try:
                    os.remove(self.processed_dir / new_path1)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_path1}: {e}")
            if new_path2:
                try:
                    os.remove(self.processed_dir / new_path2)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_path2}: {e}")
            return False

        # --- 元の列画像ファイルを削除 (オプションだが、現在の仕様) ---
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted old column image: {abs_path_to_delete}")
        except Exception as e:
            print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Column {column_path_to_split} split into {new_path1} and {new_path2}")
        return True

    def split_column_by_selection(self, column_path_to_split, selected_char_indices):
        # (省略 - 1点分割と同様のエラーハンドリングと実行順序)
        column_data = self.get_column_data(column_path_to_split)
        if not column_data: messagebox.showerror("エラー", f"列データが見つかりません: {column_path_to_split}"); return False

        char_boxes_col = column_data.get("char_boxes_in_column")
        unicode_ids = column_data.get("unicode_ids")
        original_image_path = column_data.get("original_image")
        col_box_orig_crop = column_data.get("box_in_original")

        if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list) \
           or not isinstance(original_image_path, str) or not isinstance(col_box_orig_crop, list):
            messagebox.showerror("エラー", f"列データの形式が無効です: {column_path_to_split}"); return False

        if not selected_char_indices: messagebox.showerror("エラー", "分割する文字が選択されていません。"); return False
        if len(selected_char_indices) == len(unicode_ids): messagebox.showerror("エラー", "全ての文字が選択されています。分割できません。"); return False

        # 元画像座標取得
        try:
            crop_x1, crop_y1, _, _ = col_box_orig_crop
            char_boxes_orig = []
            for box in char_boxes_col:
                orig_x1 = box[0] + crop_x1; orig_y1 = box[1] + crop_y1
                orig_x2 = box[2] + crop_x1; orig_y2 = box[3] + crop_y1
                char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
        except Exception as e: messagebox.showerror("エラー", f"文字座標の計算中にエラーが発生しました: {e}"); return False

        # 分割とソート
        selected_indices_set = set(selected_char_indices)
        chars_orig_selected, ids_selected = [], []
        chars_orig_other, ids_other = [], []
        for i, (box_orig, uid) in enumerate(zip(char_boxes_orig, unicode_ids)):
            if i in selected_indices_set:
                chars_orig_selected.append(box_orig); ids_selected.append(uid)
            else:
                chars_orig_other.append(box_orig); ids_other.append(uid)
        try:
            if chars_orig_selected:
                sorted_indices_sel = np.argsort([box[1] for box in chars_orig_selected])
                chars_orig_selected = [chars_orig_selected[i] for i in sorted_indices_sel]
                ids_selected = [ids_selected[i] for i in sorted_indices_sel]
            if chars_orig_other:
                sorted_indices_oth = np.argsort([box[1] for box in chars_orig_other])
                chars_orig_other = [chars_orig_other[i] for i in sorted_indices_oth]
                ids_other = [ids_other[i] for i in sorted_indices_oth]
        except IndexError: messagebox.showerror("エラー", "文字座標のソート中にエラーが発生しました。"); return False

        # --- 新しい列を生成 ---
        new_data_sel, new_path_sel = self._recreate_column_from_chars(
            chars_orig_selected, ids_selected, original_image_path, column_path_to_split, "_selA"
        )
        if new_data_sel is None: return False

        new_data_oth, new_path_oth = self._recreate_column_from_chars(
            chars_orig_other, ids_other, original_image_path, column_path_to_split, "_selB"
        )
        if new_data_oth is None:
            if new_path_sel:
                try:
                    os.remove(self.processed_dir / new_path_sel)
                except OSError:
                    print(f"Warning: Could not remove partially created file {new_path_sel}")
            return False

        # --- DataFrameの更新 (最後に実行) ---
        try:
            index_to_drop = self.df[self.df["column_image"] == column_path_to_split].index
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            new_rows_df = pd.DataFrame([new_data_sel, new_data_oth])
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)
        except Exception as e:
            messagebox.showerror("エラー", f"DataFrameの更新中にエラーが発生しました: {e}")
            if new_path_sel:
                try:
                    os.remove(self.processed_dir / new_path_sel)
                except OSError:
                    pass
            if new_path_oth:
                try:
                    os.remove(self.processed_dir / new_path_oth)
                except OSError:
                    pass
            return False

        # --- 元の列画像ファイルを削除 ---
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_split)
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted old column image: {abs_path_to_delete}")
        except Exception as e:
            print(f"Warning: Could not delete old column image {column_path_to_split}: {e}")

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Column {column_path_to_split} split by selection into {new_path_sel} and {new_path_oth}")
        return True

    def move_characters(self, src_column_path, target_column_path, char_indices_to_move):
        # (省略 - 1点分割と同様のエラーハンドリングと実行順序)
        if src_column_path == target_column_path: messagebox.showerror("エラー", "同じ列には移動できません。"); return False
        if not char_indices_to_move: messagebox.showinfo("情報", "移動する文字が選択されていません。"); return False

        src_data = self.get_column_data(src_column_path)
        tgt_data = self.get_column_data(target_column_path)
        if not src_data or not tgt_data: messagebox.showerror("エラー", "移動元または移動先の列データが見つかりません。"); return False

        src_orig_img = src_data.get("original_image")
        tgt_orig_img = tgt_data.get("original_image")
        if not src_orig_img or src_orig_img != tgt_orig_img: messagebox.showerror("エラー", "異なる元画像の列間、または元画像情報がない列では移動できません。"); return False
        original_image_path = src_orig_img # 共通の元画像パス

        # データ形式チェックと座標計算
        try:
            src_chars_col = src_data["char_boxes_in_column"]; src_ids = src_data["unicode_ids"]
            src_crop_x1, src_crop_y1, _, _ = src_data["box_in_original"]
            tgt_chars_col = tgt_data["char_boxes_in_column"]; tgt_ids = tgt_data["unicode_ids"]
            tgt_crop_x1, tgt_crop_y1, _, _ = tgt_data["box_in_original"]

            if not isinstance(src_chars_col, list) or not isinstance(src_ids, list) or \
               not isinstance(tgt_chars_col, list) or not isinstance(tgt_ids, list):
                raise ValueError("Invalid data format in source or target column.")

            moved_chars_orig, moved_ids = [], []
            remaining_src_chars_orig, remaining_src_ids = [], []
            src_indices_set = set(char_indices_to_move)

            for i, (box_col, uid) in enumerate(zip(src_chars_col, src_ids)):
                char_orig = [box_col[0] + src_crop_x1, box_col[1] + src_crop_y1,
                             box_col[2] + src_crop_x1, box_col[3] + src_crop_y1]
                if i in src_indices_set: moved_chars_orig.append(char_orig); moved_ids.append(uid)
                else: remaining_src_chars_orig.append(char_orig); remaining_src_ids.append(uid)

            tgt_chars_orig = []
            for box_col in tgt_chars_col:
                tgt_chars_orig.append([box_col[0] + tgt_crop_x1, box_col[1] + tgt_crop_y1,
                                       box_col[2] + tgt_crop_x1, box_col[3] + tgt_crop_y1])

            # 結合とソート
            combined_tgt_chars_orig = tgt_chars_orig + moved_chars_orig
            combined_tgt_ids = tgt_ids + moved_ids
            sorted_indices = np.argsort([box[1] for box in combined_tgt_chars_orig])
            final_tgt_chars_orig = [combined_tgt_chars_orig[i] for i in sorted_indices]
            final_tgt_ids = [combined_tgt_ids[i] for i in sorted_indices]

        except (TypeError, IndexError, KeyError, ValueError) as e:
            messagebox.showerror("エラー", f"データ処理中にエラーが発生しました: {e}"); return False

        # --- 列再生成 ---
        # 移動元 (空でも _recreate は None を返すのでOK)
        new_src_data, new_src_path = self._recreate_column_from_chars(
            remaining_src_chars_orig, remaining_src_ids, original_image_path, src_column_path, "_move_src"
        )
        # new_src_data が None でも致命的ではない場合がある (元が空になっただけ) が、
        # _recreate が None を返すのは通常エラーなので、ここで中断する方が安全
        if remaining_src_chars_orig and new_src_data is None:
            messagebox.showerror("エラー", "移動元の列の再生成に失敗しました。"); return False

        # 移動先
        new_tgt_data, new_tgt_path = self._recreate_column_from_chars(
            final_tgt_chars_orig, final_tgt_ids, original_image_path, target_column_path, "_move_tgt"
        )
        if new_tgt_data is None:
             messagebox.showerror("エラー", "移動先の列の再生成に失敗しました。")
             # ロールバック: 移動元のファイル(もし作られていたら)を削除
             if new_src_path:
                try:
                    os.remove(self.processed_dir / new_src_path)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_src_path}: {e}")
             return False

        # --- DataFrame更新 (最後に実行) ---
        try:
            indices_to_drop = self.df[
                (self.df["column_image"] == src_column_path) | (self.df["column_image"] == target_column_path)
            ].index
            self.df = self.df.drop(indices_to_drop).reset_index(drop=True)

            new_rows = []
            if new_src_data: new_rows.append(new_src_data) # 移動元が空でなければ追加
            new_rows.append(new_tgt_data) # 移動先は必ず追加

            new_rows_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)
        except Exception as e:
            messagebox.showerror("エラー", f"DataFrameの更新中にエラーが発生しました: {e}")
            # ロールバック: 作成したファイルを削除
            if new_src_path:
                try:
                    os.remove(self.processed_dir / new_src_path)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_src_path}: {e}")
            if new_tgt_path:
                try:
                    os.remove(self.processed_dir / new_tgt_path)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_tgt_path}: {e}")
            return False

        # --- 古い画像削除 ---
        delete_errors = []
        try:
            src_abs = self.get_column_abs_path(src_column_path)
            if src_abs.exists(): os.remove(src_abs); print(f"Deleted old column image: {src_column_path}")
        except Exception as e: delete_errors.append(f"Could not delete {src_column_path}: {e}")
        try:
            tgt_abs = self.get_column_abs_path(target_column_path)
            if tgt_abs.exists(): os.remove(tgt_abs); print(f"Deleted old column image: {target_column_path}")
        except Exception as e: delete_errors.append(f"Could not delete {target_column_path}: {e}")
        if delete_errors: print("Warning: Errors during old file deletion:\n" + "\n".join(delete_errors))

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Characters moved from {src_column_path} to {new_tgt_path or 'None'}")
        return True

    def delete_column(self, column_path_to_delete):
        # (省略 - 実行順序)
        column_data = self.get_column_data(column_path_to_delete)
        if not column_data:
            messagebox.showerror("エラー", f"削除対象の列データが見つかりません: {column_path_to_delete}")
            return False
        original_image_path = column_data.get("original_image")
        if not original_image_path:
             messagebox.showerror("エラー", f"列データに元画像パスがありません: {column_path_to_delete}")
             return False # original_image_path がないと changed_image_paths に記録できない

        # --- DataFrameから削除 (先に実行) ---
        try:
            index_to_drop = self.df[self.df["column_image"] == column_path_to_delete].index
            if index_to_drop.empty:
                 messagebox.showinfo("情報", f"列 {column_path_to_delete} は既に削除されているようです。")
                 return True # 実質成功
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
        except Exception as e:
            messagebox.showerror("エラー", f"DataFrameからの削除中にエラーが発生しました: {e}")
            return False

        # --- 画像ファイルを削除 ---
        try:
            abs_path_to_delete = self.get_column_abs_path(column_path_to_delete)
            dir_to_check = abs_path_to_delete.parent
            if abs_path_to_delete.exists():
                os.remove(abs_path_to_delete)
                print(f"Deleted column image: {abs_path_to_delete}")
                # ディレクトリが空になったら削除 (オプション)
                if dir_to_check.exists() and not any(dir_to_check.iterdir()):
                    try:
                        shutil.rmtree(dir_to_check)
                        print(f"Deleted empty directory: {dir_to_check}")
                    except OSError as e_rmdir:
                         print(f"Warning: Could not delete empty directory {dir_to_check}: {e_rmdir}")
        except Exception as e:
            # ファイル削除失敗は警告にとどめる（DataFrameからは消えているため）
            print(f"Warning: Could not delete column image file or dir {column_path_to_delete}: {e}")

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Deleted column: {column_path_to_delete}")
        return True

    def delete_character(self, column_path, char_index_to_delete):
        # (省略 - 実行順序、エラーハンドリング)
        col_data = self.get_column_data(column_path)
        if not col_data: messagebox.showerror("Error", f"Column data not found: {column_path}"); return False

        char_boxes_col = col_data.get("char_boxes_in_column")
        unicode_ids = col_data.get("unicode_ids")
        original_image_path = col_data.get("original_image")
        col_box_orig_crop = col_data.get("box_in_original")

        if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list) \
           or not isinstance(original_image_path, str) or not isinstance(col_box_orig_crop, list):
            messagebox.showerror("エラー", f"列データの形式が無効です: {column_path}"); return False

        if not 0 <= char_index_to_delete < len(unicode_ids):
            messagebox.showerror("Error", "Invalid character index to delete."); return False

        # 削除対象を除いたリストを作成
        new_char_boxes_col = [box for i, box in enumerate(char_boxes_col) if i != char_index_to_delete]
        new_unicode_ids = [uid for i, uid in enumerate(unicode_ids) if i != char_index_to_delete]

        # --- 全ての文字が削除された場合 -> 列ごと削除 ---
        if not new_unicode_ids:
            print(f"All characters deleted from {column_path}. Deleting column.")
            # delete_column を呼び出すと、その中で DataFrame 更新とファイル削除が行われる
            return self.delete_column(column_path)

        # --- 文字が残っている場合 -> 列を再生成 ---
        try:
            crop_x1_orig, crop_y1_orig, _, _ = col_box_orig_crop
            new_char_boxes_orig = []
            for box_col in new_char_boxes_col:
                orig_x1 = box_col[0] + crop_x1_orig; orig_y1 = box_col[1] + crop_y1_orig
                orig_x2 = box_col[2] + crop_x1_orig; orig_y2 = box_col[3] + crop_y1_orig
                new_char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
        except Exception as e:
            messagebox.showerror("エラー", f"文字座標の計算中にエラーが発生しました: {e}"); return False

        # 列を再生成
        new_col_data, new_col_path = self._recreate_column_from_chars(
            new_char_boxes_orig, new_unicode_ids, original_image_path, column_path, "_chardel"
        )
        if new_col_data is None:
            messagebox.showerror("Error", "Failed to recreate column after deleting character."); return False

        # --- DataFrame Update (最後に実行) ---
        try:
            index_to_drop = self.df[self.df["column_image"] == column_path].index
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            new_row_df = pd.DataFrame([new_col_data])
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        except Exception as e:
            messagebox.showerror("Error", f"DataFrame update failed after char delete: {e}")
            if new_col_path:
                try:
                    os.remove(self.processed_dir / new_col_path)
                except OSError as e:
                    print(f"Warning: Could not remove new column image {new_col_path}: {e}")
            return False

        # --- Delete old image ---
        try:
            old_abs_path = self.get_column_abs_path(column_path)
            if old_abs_path.exists():
                os.remove(old_abs_path)
                print(f"Deleted old column image: {column_path}")
        except Exception as e:
            print(f"Warning: could not delete old image {column_path}: {e}")

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Character {char_index_to_delete} deleted from {column_path}, recreated as {new_col_path}")
        return True

    # ★新規追加: 文字追加メソッド
    def add_character(self, column_path, new_char_box_in_col, new_unicode_id):
        """
        指定された列に新しい文字を追加し、列を再生成する。
        new_char_box_in_col: [x1, y1, x2, y2] (列画像内の相対座標)
        """
        col_data = self.get_column_data(column_path)
        if not col_data: messagebox.showerror("Error", f"Column data not found: {column_path}"); return False

        original_image_path = col_data.get("original_image")
        col_box_orig_crop = col_data.get("box_in_original") # マージン込み座標
        char_boxes_col = col_data.get("char_boxes_in_column", [])
        unicode_ids = col_data.get("unicode_ids", [])

        if not isinstance(char_boxes_col, list) or not isinstance(unicode_ids, list) \
           or not isinstance(original_image_path, str) or not isinstance(col_box_orig_crop, list):
            messagebox.showerror("エラー", f"列データの形式が無効です: {column_path}"); return False
        if not isinstance(new_char_box_in_col, list) or len(new_char_box_in_col) != 4:
             messagebox.showerror("エラー", "追加する文字の座標形式が無効です。"); return False
        if not isinstance(new_unicode_id, str) or not re.match(r'^U\+[0-9A-Fa-f]+$', new_unicode_id):
             messagebox.showerror("エラー", "追加する文字のUnicode ID形式が無効です (例: U+XXXX)。"); return False

        # --- 元画像上の絶対座標に変換 ---
        try:
            crop_x1_orig, crop_y1_orig, _, _ = col_box_orig_crop
            # 既存の文字
            existing_char_boxes_orig = []
            for box_col in char_boxes_col:
                orig_x1 = box_col[0] + crop_x1_orig; orig_y1 = box_col[1] + crop_y1_orig
                orig_x2 = box_col[2] + crop_x1_orig; orig_y2 = box_col[3] + crop_y1_orig
                existing_char_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
            # 新しい文字
            nx1, ny1, nx2, ny2 = new_char_box_in_col
            new_char_box_orig = [nx1 + crop_x1_orig, ny1 + crop_y1_orig,
                                 nx2 + crop_x1_orig, ny2 + crop_y1_orig]
        except Exception as e:
            messagebox.showerror("エラー", f"文字座標の計算中にエラーが発生しました: {e}"); return False

        # --- 新しい文字を追加し、Y座標でソート ---
        all_char_boxes_orig = existing_char_boxes_orig + [new_char_box_orig]
        all_unicode_ids = unicode_ids + [new_unicode_id]
        try:
            sorted_indices = np.argsort([box[1] for box in all_char_boxes_orig])
            final_char_boxes_orig = [all_char_boxes_orig[i] for i in sorted_indices]
            final_unicode_ids = [all_unicode_ids[i] for i in sorted_indices]
        except IndexError: messagebox.showerror("エラー", "文字座標のソート中にエラーが発生しました。"); return False

        # --- 列を再生成 ---
        new_col_data, new_col_path = self._recreate_column_from_chars(
            final_char_boxes_orig, final_unicode_ids, original_image_path, column_path, "_charadd"
        )
        if new_col_data is None:
            messagebox.showerror("Error", "Failed to recreate column after adding character."); return False

        # --- DataFrame Update (最後に実行) ---
        try:
            index_to_drop = self.df[self.df["column_image"] == column_path].index
            self.df = self.df.drop(index_to_drop).reset_index(drop=True)
            new_row_df = pd.DataFrame([new_col_data])
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        except Exception as e:
            messagebox.showerror("Error", f"DataFrame update failed after char add: {e}")
            if new_col_path:
                try:
                    os.remove(self.processed_dir / new_col_path)
                except OSError as e:
                    print(f"Warning: Could not remove partially created file {new_col_path}: {e}")
            return False

        # --- Delete old image ---
        try:
            old_abs_path = self.get_column_abs_path(column_path)
            if old_abs_path.exists():
                os.remove(old_abs_path)
                print(f"Deleted old column image: {column_path}")
        except Exception as e:
            print(f"Warning: could not delete old image {column_path}: {e}")

        # 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"Character '{new_unicode_id}' added to {column_path}, recreated as {new_col_path}")
        return True

    # ★新規追加: 列追加メソッド
    def add_column(self, original_image_path, new_box_in_original):
        """
        指定された元画像に新しい列を追加する。
        new_box_in_original: [x1, y1, x2, y2] (元画像上の絶対座標、マージンなし)
        """
        if not isinstance(new_box_in_original, list) or len(new_box_in_original) != 4:
            messagebox.showerror("エラー", "新しい列の座標形式が無効です。", parent=None) # GUIがない場合もあるので parent=None
            return False

        nc_x1, nc_y1, nc_x2, nc_y2 = new_box_in_original

        # 1. 元画像を開く
        orig_img_abs_path = self.get_original_image_abs_path(original_image_path)
        if not orig_img_abs_path or not orig_img_abs_path.exists():
             messagebox.showerror("エラー", f"元画像ファイルが見つかりません:\n{orig_img_abs_path}", parent=None)
             print(f"Error: Original image not found: {orig_img_abs_path}")
             return False
        try:
            orig_img = Image.open(orig_img_abs_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("エラー", f"元画像の読み込みに失敗しました:\n{orig_img_abs_path}\n{e}", parent=None)
            print(f"Error opening original image {original_image_path}: {e}")
            return False

        # 2. マージン追加と切り抜き座標計算
        margin = 5 # TODO: 設定可能にするか検討
        crop_x1 = max(0, int(nc_x1 - margin))
        crop_y1 = max(0, int(nc_y1 - margin))
        crop_x2 = min(orig_img.width, int(nc_x2 + margin))
        crop_y2 = min(orig_img.height, int(nc_y2 + margin))

        # 幅か高さが0以下の場合はエラー
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            print(f"Warning: Invalid crop region calculated for new column ([{crop_x1},{crop_y1},{crop_x2},{crop_y2}]). Cannot create column image.")
            messagebox.showerror("エラー", f"列画像の切り抜き領域が無効になりました。\n座標: [{crop_x1},{crop_y1},{crop_x2},{crop_y2}]", parent=None)
            return False

        # 3. 列画像切り出し
        try:
            new_col_img_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        except Exception as e:
            messagebox.showerror("エラー", f"画像の切り出し中にエラーが発生しました:\n{e}", parent=None)
            print(f"Error cropping image for new column: {e}")
            return False

        # 4. 新しい列画像のパス決定と保存
        try:
            # 保存先ディレクトリを決定 (書籍ID/ページID)
            orig_path_obj = Path(original_image_path)
            # 例: '100241706/images/100241706_00001.jpg'
            book_id = orig_path_obj.parent.parent.stem # '100241706'
            page_id = orig_path_obj.stem       # '100241706_00001'
            save_dir = self.column_images_base_dir / book_id / page_id # data/processed/column_images/書籍ID/ページID
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
            # 新しいファイル名 (例: newcol_2025....jpg)
            new_filename = f"newcol_{timestamp}.jpg"
            new_col_abs_path = save_dir / new_filename # data/processed/column_images/書籍ID/ページID/newcol_....jpg

            # 相対パス (processed/からのパス)
            new_col_rel_path = f"column_images/{book_id}/{page_id}/{new_filename}"

            new_col_img_pil.save(new_col_abs_path, "JPEG")
            print(f"New column image saved to: {new_col_abs_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"新しい列画像の保存中にエラーが発生しました:\n{new_col_abs_path}\n{e}", parent=None)
            print(f"Error saving new column image {new_col_abs_path}: {e}")
            return False

        # 5. 新しい行データ作成 (文字情報は空)
        new_data = {
            "column_image": str(new_col_rel_path).replace("\\", "/"),
            "original_image": original_image_path,
            "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2], # マージン込みの座標
            "char_boxes_in_column": [], # 新規列なので空
            "unicode_ids": [],          # 新規列なので空
            # 他の列があればデフォルト値などを設定
        }

        # 6. DataFrameに追加
        try:
            new_row_df = pd.DataFrame([new_data])
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        except Exception as e:
             messagebox.showerror("エラー", f"DataFrameへの追加中にエラーが発生しました: {e}", parent=None)
             # ロールバック: 作成したファイルを削除
             try: os.remove(new_col_abs_path)
             except OSError as e_rm: print(f"Warning: Could not remove created file {new_col_abs_path}: {e_rm}")
             return False

        # 7. 変更を記録
        self.changed_image_paths.add(original_image_path)
        print(f"New column added: {new_col_rel_path}")
        return True


    def _recreate_column_from_chars(self, chars_in_orig, u_ids, original_img_path, base_rel_path_str, suffix):
        # (省略 - エラーハンドリング強化)
        if not chars_in_orig:
            # 空のリストが渡された場合、エラーではなく (None, None) を返す
            # (例: 文字移動で移動元が空になった場合など)
            return None, None

        # 1. 新しい列のバウンディングボックス (マージンなし)
        new_col_bounds_no_margin = self._recalculate_column_bounds(chars_in_orig)
        nc_x1, nc_y1, nc_x2, nc_y2 = new_col_bounds_no_margin

        # 2. マージン追加と切り抜き座標計算
        orig_img_abs_path = self.get_original_image_abs_path(original_img_path)
        if not orig_img_abs_path or not orig_img_abs_path.exists():
             # エラーメッセージをより具体的に
             messagebox.showerror("エラー", f"元画像ファイルが見つかりません:\n{orig_img_abs_path}")
             print(f"Error: Original image not found: {orig_img_abs_path}")
             return None, None
        try:
            orig_img = Image.open(orig_img_abs_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("エラー", f"元画像の読み込みに失敗しました:\n{orig_img_abs_path}\n{e}")
            print(f"Error opening original image {original_img_path}: {e}")
            return None, None

        margin = 5 # TODO: 設定可能にするか検討
        crop_x1 = max(0, int(nc_x1 - margin))
        crop_y1 = max(0, int(nc_y1 - margin))
        crop_x2 = min(orig_img.width, int(nc_x2 + margin))
        crop_y2 = min(orig_img.height, int(nc_y2 + margin))

        # 幅か高さが0以下の場合はエラー
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            print(f"Warning: Invalid crop region calculated for {base_rel_path_str}{suffix} ([{crop_x1},{crop_y1},{crop_x2},{crop_y2}]). Cannot create column image.")
            messagebox.showerror("エラー", f"列画像の切り抜き領域が無効になりました。\n座標: [{crop_x1},{crop_y1},{crop_x2},{crop_y2}]")
            return None, None

        # 3. 列画像切り出し
        try:
            new_col_img_pil = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        except Exception as e:
            messagebox.showerror("エラー", f"画像の切り出し中にエラーが発生しました:\n{e}")
            print(f"Error cropping image for {base_rel_path_str}{suffix}: {e}")
            return None, None

        # 4. 新しい列画像のパス決定と保存
        try:
            base_path = Path(base_rel_path_str)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
            # 元のファイル名から拡張子を除去してサフィックスとタイムスタンプを追加
            new_filename = f"{base_path.stem}{suffix}_{timestamp}.jpg"
            # 保存先ディレクトリを元のパスから取得
            new_col_rel_path = base_path.parent / new_filename
            new_col_abs_path = self.processed_dir / new_col_rel_path
            new_col_abs_path.parent.mkdir(parents=True, exist_ok=True)
            new_col_img_pil.save(new_col_abs_path, "JPEG")
        except Exception as e:
            messagebox.showerror("エラー", f"新しい列画像の保存中にエラーが発生しました:\n{new_col_abs_path}\n{e}")
            print(f"Error saving new column image {new_col_abs_path}: {e}")
            # 保存に失敗した場合、ファイルが中途半端に残る可能性がある -> ここでは削除しない
            return None, None

        # 5. 列画像内の文字座標 (相対座標、マージン考慮)
        char_boxes_in_cropped = []
        try:
            for box_orig in chars_in_orig:
                # 座標は整数に丸める (floatでも問題ない場合が多いが、整数が無難)
                rel_x1 = int(box_orig[0] - crop_x1)
                rel_y1 = int(box_orig[1] - crop_y1)
                rel_x2 = int(box_orig[2] - crop_x1)
                rel_y2 = int(box_orig[3] - crop_y1)
                char_boxes_in_cropped.append([rel_x1, rel_y1, rel_x2, rel_y2])
        except IndexError:
            messagebox.showerror("エラー", "文字座標の計算中にインデックスエラーが発生しました。")
            # 作成したファイルを削除するロールバック処理
            try:
                os.remove(new_col_abs_path)
            except OSError as e:
                print(f"Warning: Could not delete new column image {new_col_abs_path}: {e}")
            return None, None

        # 6. 新しい行データ作成
        new_data = {
            "column_image": str(new_col_rel_path).replace("\\", "/"),
            "original_image": original_img_path,
            "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],
            "char_boxes_in_column": char_boxes_in_cropped,
            "unicode_ids": u_ids,
            # TODO: f1_score_below_1.0.csv に他の列があれば、それらも適切に設定する必要がある
            # 例: "score": 1.0 (手動編集なのでスコアは最大？) など
        }
        return new_data, new_col_rel_path


    # ★変更: 変更があったページのアノテーションを別のCSVファイルに「追記」する
    def save_changes(self):
        """
        現在のセッションで変更があったページについて、最新のアノテーションを
        別のCSVファイル (self.modified_csv_path) に追記する。
        追記前に、ファイル内に既に存在する該当ページの古いデータを削除する。
        """
        if not self.changed_image_paths:
            print("保存すべき変更はありません。")
            return True

        changed_paths_list = list(self.changed_image_paths)
        if not changed_paths_list:
            print("変更リストは空です。")
            return True

        print(f"変更があった {len(changed_paths_list)} ページのアノテーションを '{self.modified_csv_path}' に追記します...")

        try:
            # --- メモリ上のDataFrameから今回変更があったページの最新データを抽出 ---
            df_to_append = self.df[self.df['original_image'].isin(changed_paths_list)].copy()

            if df_to_append.empty:
                print(f"Warning: DataFrame内に今回変更があったページ ({changed_paths_list}) のデータが見つかりませんでした。")
                self.changed_image_paths.clear()
                return True

            # --- DataFrameをCSV保存用に準備 (文字列変換) ---
            def safe_stringify(value): # (前回と同じ safe_stringify 関数)
                def nan_to_none_recursive(item):
                    if isinstance(item, list): return [nan_to_none_recursive(sub_item) for sub_item in item]
                    elif isinstance(item, tuple): return tuple(nan_to_none_recursive(sub_item) for sub_item in item)
                    elif isinstance(item, dict): return {k: nan_to_none_recursive(v) for k, v in item.items()}
                    elif isinstance(item, float) and np.isnan(item): return None
                    elif pd.isna(item): return None
                    else: return item
                if isinstance(value, np.ndarray):
                    try:
                        py_list = value.tolist()
                        cleaned_list = nan_to_none_recursive(py_list)
                        return str(cleaned_list)
                    except Exception: return '[]'
                elif isinstance(value, (list, tuple)):
                    cleaned_list = nan_to_none_recursive(list(value))
                    return str(cleaned_list)
                elif isinstance(value, dict):
                    cleaned_dict = nan_to_none_recursive(value)
                    return str(cleaned_dict)
                elif pd.isna(value): return '[]'
                else: return str(value)

            cols_to_stringify = ["box_in_original", "char_boxes_in_column", "unicode_ids"]
            for col in cols_to_stringify:
                if col in df_to_append.columns:
                    try:
                        df_to_append[col] = df_to_append[col].apply(safe_stringify)
                    except Exception as e:
                        print(f"Error applying string conversion to column '{col}'. Saving might fail. Error: {e}")
                        messagebox.showerror("内部エラー", f"列 '{col}' のデータ変換中にエラーが発生しました。\n保存に失敗する可能性があります。\nError: {e}")
                        return False

            # --- 追記処理 ---
            file_exists = self.modified_csv_path.exists()
            temp_file_path = self.modified_csv_path.with_suffix(self.modified_csv_path.suffix + '.tmp') # 一時ファイル

            if file_exists:
                # 既存ファイルを読み込み、追記対象ページのデータを削除
                try:
                    df_existing = pd.read_csv(self.modified_csv_path) # コンバーターなしで読み込む(文字列比較のため)
                    # original_image 列が文字列でない可能性を考慮
                    df_existing['original_image'] = df_existing['original_image'].astype(str)
                    changed_paths_str = [str(p) for p in changed_paths_list]

                    # 追記対象外のデータのみをフィルタリング
                    df_to_keep = df_existing[~df_existing['original_image'].isin(changed_paths_str)]

                    # ヘッダー付きで一時ファイルに書き込み (mode='w')
                    df_to_keep.to_csv(temp_file_path, index=False, encoding='utf-8', mode='w')

                    # 追記するデータを一時ファイルに追記 (mode='a', header=False)
                    df_to_append.to_csv(temp_file_path, index=False, encoding='utf-8', mode='a', header=False)

                    # 元のファイルを削除し、一時ファイル名を変更
                    os.remove(self.modified_csv_path)
                    os.rename(temp_file_path, self.modified_csv_path)

                except Exception as read_write_e:
                    print(f"既存CSVの読み書き中にエラーが発生しました: {read_write_e}")
                    # エラー時は追記だけ試みる (重複データが残る可能性あり)
                    print("フォールバック: 既存データを削除せずに追記します。")
                    df_to_append.to_csv(self.modified_csv_path, index=False, encoding='utf-8', mode='a', header=not file_exists)
                    # エラーが発生したことをユーザーに通知
                    messagebox.showwarning("保存警告", f"既存データの更新中にエラーが発生しました。\nデータを追記しましたが、古いデータが残っている可能性があります。\nError: {read_write_e}")
                    # 追記は試みたので changed_image_paths はクリアする
                    self.changed_image_paths.clear()
                    return True # 部分成功として扱う

            else:
                # ファイルが存在しない場合は、ヘッダー付きで新規作成 (mode='w')
                df_to_append.to_csv(self.modified_csv_path, index=False, encoding='utf-8', mode='w', header=True)

            # 保存が成功したら変更フラグをクリア
            self.changed_image_paths.clear()
            print(f"変更されたページのアノテーションをCSVファイルに追記/更新しました: {self.modified_csv_path}")
            return True

        except PermissionError as e:
            error_msg = f"CSVファイルへの書き込み権限がありません:\n{self.modified_csv_path}\n\nファイルが開かれていないか確認してください。\nError: {e}"
            print(error_msg)
            messagebox.showerror("保存エラー", error_msg)
            return False
        except IOError as e:
            error_msg = f"CSVファイルへの書き込み中にエラーが発生しました:\n{self.modified_csv_path}\n\nディスク容量などを確認してください。\nError: {e}"
            print(error_msg)
            messagebox.showerror("保存エラー", error_msg)
            return False
        except Exception as e:
            error_msg = f"CSVファイルへの保存中に予期せぬエラーが発生しました:\n{self.modified_csv_path}\n\nError: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            messagebox.showerror("保存エラー", error_msg)
            return False


    # 元のCSV保存メソッド (参考用、未使用)
    def _save_to_original_csv(self):
        # if not self.changes_made: # 古いフラグ
        if not self.changed_image_paths:
            return True
        try:
            # listやdictを文字列に変換して保存 (read_csvで再度パースできるように)
            # df_to_save = self.df.copy()
            # for col in ["box_in_original", "char_boxes_in_column", "unicode_ids"]:
            #     if col in df_to_save.columns:
            #         # NaNを空リストなどに変換してから文字列化
            #         df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else str([] if pd.isna(x) else x) )
            # df_to_save.to_csv(self.csv_path, index=False)

            # 現在は DataFrame の内容は Python オブジェクトのままなので、そのまま to_csv
            self.df.to_csv(self.csv_path, index=False)
            self.changed_image_paths.clear() # 保存したら変更済みセットをクリア
            print(f"変更を元のCSVに保存しました: {self.csv_path}")
            return True
        except Exception as e:
            messagebox.showerror("保存エラー", f"CSVファイルへの保存中にエラーが発生しました:\n{e}")
            print(f"Error saving CSV: {e}")
            return False

    def discard_changes(self):
        """未保存の変更を破棄し、元のCSVからデータを再読み込みする"""
        if self.changed_image_paths:
            print("Discarding unsaved changes...")
            current_orig_image = self.current_original_image # 現在表示中の画像を覚えておく
            try:
                self.load_data() # 元のCSVから再読み込み
                # 表示中の画像に戻す (リストになければ最初に戻る)
                if current_orig_image and current_orig_image in self.original_images:
                    self.current_original_image = current_orig_image
                elif self.original_images:
                    self.current_original_image = self.original_images[0]
                else:
                    self.current_original_image = None
                self.changed_image_paths.clear()
                print("Changes discarded and data reloaded.")
                return True
            except Exception as e:
                messagebox.showerror("エラー", f"データの再読み込み中にエラーが発生しました: {e}")
                print(f"Error reloading data after discarding changes: {e}")
                # エラーが起きた場合、変更は破棄されたことにするかどうか？ -> する
                self.changed_image_paths.clear()
                return False
        return True # 変更がなければTrue


class AnnotatorApp:
    """GUIアプリケーション本体 (変更あり)"""
    def __init__(self, root, base_data_dir):
        self.root = root
        self.root.title("縦書き列アノテーション修正ツール")
        self.root.geometry("1200x800")

        try:
            self.data_manager = DataManager(base_data_dir)
        except (FileNotFoundError, IOError) as e:
             messagebox.showerror("起動エラー", f"{e}")
             root.destroy()
             return # DataManager初期化失敗時は起動しない

        # --- メインフレーム ---
        # (省略 - 元コードと同じ)
        main_frame = ttk.Frame(root, padding="5")
        main_frame.pack(expand=True, fill="both")
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=2)

        # --- 上部: ナビゲーション ---
        # (省略 - 元コードと同じ)
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        self.prev_button = ttk.Button(nav_frame, text="<< 前の画像", command=self.prev_image)
        self.prev_button.pack(side="left", padx=5)
        self.image_label = ttk.Label(nav_frame, text="画像: ", width=60, anchor="w")
        self.image_label.pack(side="left", padx=5, fill="x", expand=True)
        self.next_button = ttk.Button(nav_frame, text="次の画像 >>", command=self.next_image)
        self.next_button.pack(side="left", padx=5)
        self.page_info_label = ttk.Label(nav_frame, text="- / -", width=10, anchor="e")
        self.page_info_label.pack(side="right", padx=5)
        self.save_button = ttk.Button(nav_frame, text="変更を保存", command=self.save_all_changes)
        self.save_button.pack(side="right", padx=5)


        # --- 左: 元画像表示 ---
        # (省略 - 元コードと同じ)
        orig_frame = ttk.LabelFrame(main_frame, text="元画像と列範囲")
        orig_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        orig_frame.rowconfigure(0, weight=1)
        orig_frame.columnconfigure(0, weight=1)
        self.orig_canvas = ImageCanvas(orig_frame, bg="lightgrey")
        self.orig_canvas.grid(row=0, column=0, sticky="nsew")

        # --- 中央: 列リストと操作 ---
        list_op_frame = ttk.Frame(main_frame)
        list_op_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        list_op_frame.rowconfigure(1, weight=1)
        list_op_frame.columnconfigure(0, weight=1)

        op_buttons_frame = ttk.Frame(list_op_frame)
        op_buttons_frame.grid(row=0, column=0, sticky="ew", pady=(5,2)) # 上下の pady を調整

        # ボタンを2行に分ける (スペース確保のため)
        op_buttons_frame1 = ttk.Frame(op_buttons_frame)
        op_buttons_frame1.pack(fill="x")
        op_buttons_frame2 = ttk.Frame(op_buttons_frame)
        op_buttons_frame2.pack(fill="x", pady=(2,0))

        self.merge_button = ttk.Button(op_buttons_frame1, text="結合", command=self.merge_selected_columns)
        self.merge_button.pack(side="left", padx=2, fill="x", expand=True)
        self.split_button = ttk.Button(op_buttons_frame1, text="1点分割", command=self.split_selected_column)
        self.split_button.pack(side="left", padx=2, fill="x", expand=True)
        self.split_selection_button = ttk.Button(op_buttons_frame1, text="選択分割", command=self.split_column_by_selection)
        self.split_selection_button.pack(side="left", padx=2, fill="x", expand=True)

        self.move_char_button = ttk.Button(op_buttons_frame2, text="文字移動", command=self.initiate_move_character)
        self.move_char_button.pack(side="left", padx=2, fill="x", expand=True)
        self.delete_col_button = ttk.Button(op_buttons_frame2, text="列削除", command=self.delete_selected_column)
        self.delete_col_button.pack(side="left", padx=2, fill="x", expand=True)
        self.delete_char_button = ttk.Button(op_buttons_frame2, text="文字削除", command=self.delete_selected_character)
        self.delete_char_button.pack(side="left", padx=2, fill="x", expand=True)
        # ★変更: 文字追加ボタンは削除 (右クリックで代替)
        # self.add_char_button = ttk.Button(op_buttons_frame2, text="文字追加", command=self.initiate_add_character_dialog)
        # self.add_char_button.pack(side="left", padx=2, fill="x", expand=True)


        list_frame = ttk.LabelFrame(list_op_frame, text="現在の画像の列一覧")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(2,0))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.column_listbox = tk.Listbox(list_frame, selectmode="extended")
        self.column_listbox.grid(row=0, column=0, sticky="nsew")
        self.column_listbox.bind("<<ListboxSelect>>", self.on_column_select)
        list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.column_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.column_listbox.config(yscrollcommand=list_scrollbar.set)

        # --- 右: 選択列詳細 ---
        detail_frame = ttk.LabelFrame(main_frame, text="選択された列の詳細 (右クリックで文字追加)") # ラベル変更
        detail_frame.grid(row=1, column=2, sticky="nsew", padx=5)
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        self.detail_canvas = ImageCanvas(detail_frame, bg="lightgrey")
        self.detail_canvas.grid(row=0, column=0, sticky="nsew")

        # --- 初期化 ---
        self.selected_column_paths = []
        self.current_detail_column_path = None
        self.moving_characters_info = None

        # --- ▼▼▼ 文字追加モード用状態変数 (リセット) ▼▼▼ ---
        self.add_char_mode = None
        self.first_click_pos = None
        self.pending_unicode_id = None
        # --- ▲▲▲ 文字追加モード用状態変数 ▲▲▲ ---

        # --- ▼▼▼ 列追加モード用状態変数 ▼▼▼ ---
        self.add_column_mode = None # None, 'waiting_first_click', 'waiting_second_click'
        self.first_click_pos_orig = None # (canvas_x, canvas_y) - 元画像Canvas用
        # --- ▲▲▲ 列追加モード用状態変数 ▲▲▲ ---

        # Canvasクリックのコールバックを設定
        self.orig_canvas.master.on_canvas_click = self.handle_orig_canvas_click
        self.detail_canvas.master.on_canvas_click = self.handle_detail_canvas_click
        # ★変更: 右クリックコールバック設定
        self.orig_canvas.master.on_canvas_right_click = self.initiate_add_column # 元画像右クリック
        self.detail_canvas.master.on_canvas_right_click = self.initiate_add_character # 詳細右クリック

        self.load_current_image_data()

        # ウィンドウを閉じる際の処理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- キーボードショートカット (アプリケーション全体にバインド) ---
        self.root.bind_all("<KeyPress>", self.handle_key_press)

    def handle_key_press(self, event):
        # (省略 - 元コードと同じ + Escキー処理追加)
        if event.widget.winfo_class() in ('Entry', 'Text'): return # 入力中は無視

        # --- ▼▼▼ Escキーで各種モードキャンセル ▼▼▼ ---
        if event.keysym == "Escape":
            cancelled = False
            if self.add_char_mode:
                print("文字追加モードをキャンセルしました (Escキー)。")
                self.add_char_mode = None
                self.first_click_pos = None
                self.pending_unicode_id = None
                cancelled = True
            if self.add_column_mode: # 列追加モードのキャンセルを追加
                print("列追加モードをキャンセルしました (Escキー)。")
                self.add_column_mode = None
                self.first_click_pos_orig = None
                cancelled = True
            if self.moving_characters_info:
                print("文字移動モードをキャンセルしました (Escキー)。")
                self.moving_characters_info = None
                cancelled = True

            if cancelled:
                self.update_button_states() # カーソルを戻す
                messagebox.showinfo("キャンセル", "操作をキャンセルしました。", parent=self.root)
                return # 他のキー処理は行わない
        # --- ▲▲▲ Escキーで各種モードキャンセル ▲▲▲ ---

        if event.keysym == "Left": self.prev_image()
        elif event.keysym == "Right": self.next_image()
        elif event.keysym == "Return":
            if len(self.selected_column_paths) >= 2:
                self.merge_selected_columns()
            # 詳細エリアにフォーカスがあれば、文字追加確定なども考えられるが、今は結合のみ

    def load_current_image_data(self):
        # (省略 - 元コードと同じ、ページ情報更新など)
        self.orig_canvas.clear_canvas()
        self.column_listbox.delete(0, tk.END)
        self.detail_canvas.clear_canvas()
        self.selected_column_paths = []
        self.current_detail_column_path = None
        self.moving_characters_info = None
        # self.adding_character_info = None # 古い形式は削除

        # --- ▼▼▼ 各種モード用状態変数 (リセット) ▼▼▼ ---
        self.add_char_mode = None
        self.first_click_pos = None
        self.pending_unicode_id = None
        self.add_column_mode = None # 列追加モードもリセット
        self.first_click_pos_orig = None
        # --- ▲▲▲ 各種モード用状態変数 ▲▲▲ ---

        current_image_path = self.data_manager.current_original_image
        all_images = self.data_manager.get_original_images()
        total_pages = len(all_images)

        if not current_image_path or not all_images:
            self.image_label.config(text="画像: (データなし)")
            self.page_info_label.config(text="0 / 0")
            self.update_button_states() # ボタン状態も更新
            return
        else:
            try:
                current_page_index = all_images.index(current_image_path)
                current_page_num = current_page_index + 1
                self.page_info_label.config(text=f"{current_page_num} / {total_pages}")
            except ValueError:
                self.page_info_label.config(text=f"? / {total_pages}")

            label_path = Path(current_image_path)
            display_name = "/".join(label_path.parts[-4:])
            self.image_label.config(text=f"画像: {display_name}")

        orig_abs_path = self.data_manager.get_original_image_abs_path(current_image_path)
        if orig_abs_path and orig_abs_path.exists():
            self.orig_canvas.load_image(orig_abs_path)
        else:
            print(f"Error: Original image file not found or path is incorrect: {orig_abs_path}")
            self.orig_canvas.clear_canvas()
            self.orig_canvas.create_text(self.orig_canvas.winfo_width() // 2 if self.orig_canvas.winfo_width() > 1 else 300, 20,
                                         text=f"元画像が見つかりません:\n{orig_abs_path}", fill="red", anchor="n", width=self.orig_canvas.winfo_width()-20 if self.orig_canvas.winfo_width() > 20 else 580)

        columns_df = self.data_manager.get_columns_for_current_image()
        if not columns_df.empty:
            # 列をソート (box_in_original がリストでない場合を考慮)
            def get_sort_key(box):
                 if isinstance(box, list) and len(box) > 0 and isinstance(box[0], (int, float)):
                     return box[0]
                 return float('inf')
            columns_df["sort_key"] = columns_df["box_in_original"].apply(get_sort_key)
            columns_df = columns_df.sort_values("sort_key")

            colors = ["orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
            for i, (_, row) in enumerate(columns_df.iterrows()):
                col_rel_path = row["column_image"]
                display_text = Path(col_rel_path).name
                self.column_listbox.insert(tk.END, display_text)
                self.column_listbox.itemconfig(tk.END, {"fg": "black"})

                col_box = row["box_in_original"]
                if isinstance(col_box, list) and len(col_box) == 4 and all(isinstance(n, (int, float)) for n in col_box):
                    self.orig_canvas.add_box(
                        tag=col_rel_path, x1=col_box[0], y1=col_box[1], x2=col_box[2], y2=col_box[3],
                        color=colors[i % len(colors)], width=1
                    )
                else:
                    print(f"Warning: Invalid column box format for {col_rel_path}: {col_box}. Skipping drawing on original image.")

        self.update_button_states()

        if self.column_listbox.size() > 0:
            self.column_listbox.selection_clear(0, tk.END)
            self.column_listbox.selection_set(0)
            self.column_listbox.activate(0) # アクティブ要素も設定
            self.column_listbox.event_generate("<<ListboxSelect>>")

        self.root.update_idletasks()
        # フォーカス設定は状況によって挙動が変わるので、一旦削除。必要なら戻す。
        # self.column_listbox.focus_force()

    def on_column_select(self, event=None):
        # (省略 - ソートキー取得部分を安全に)
        selected_indices = self.column_listbox.curselection()
        if not selected_indices:
            self.selected_column_paths = []
            self.detail_canvas.clear_canvas()
            self.current_detail_column_path = None
        else:
            all_columns_df = self.data_manager.get_columns_for_current_image()
            if all_columns_df.empty: # データがない場合は何もしない
                 self.selected_column_paths = []
                 self.detail_canvas.clear_canvas()
                 self.current_detail_column_path = None
            else:
                def get_sort_key(box):
                    if isinstance(box, list) and len(box) > 0 and isinstance(box[0], (int, float)): return box[0]
                    return float('inf')
                all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(get_sort_key)
                # インデックスをリセットして、リストボックスのインデックスと一致させる
                all_columns_df = all_columns_df.sort_values("sort_key").reset_index(drop=True)

                # インデックスが範囲外にならないようにチェック
                valid_indices = [idx for idx in selected_indices if 0 <= idx < len(all_columns_df)]
                if not valid_indices:
                    self.selected_column_paths = []
                    self.detail_canvas.clear_canvas()
                    self.current_detail_column_path = None
                else:
                    self.selected_column_paths = [all_columns_df.iloc[idx]["column_image"] for idx in valid_indices]
                    last_selected_idx = valid_indices[-1]
                    self.current_detail_column_path = all_columns_df.iloc[last_selected_idx]["column_image"]
                    self.load_detail_column(self.current_detail_column_path)

        self.highlight_selected_columns_on_orig()
        self.update_button_states()

    def load_detail_column(self, column_rel_path):
        # (省略 - 元コードと同じ、エラーチェック強化)
        self.detail_canvas.clear_canvas()
        column_data = self.data_manager.get_column_data(column_rel_path)
        if not column_data:
            print(f"Error: Cannot load detail, column data not found for {column_rel_path}")
            self.detail_canvas.create_text(10, 10, text=f"列データが見つかりません:\n{column_rel_path}", fill="red", anchor="nw")
            return

        col_abs_path = self.data_manager.get_column_abs_path(column_rel_path)
        if col_abs_path.exists():
            self.detail_canvas.load_image(col_abs_path)
            char_boxes = column_data.get("char_boxes_in_column")
            uids = column_data.get("unicode_ids")

            if isinstance(char_boxes, list) and isinstance(uids, list) and len(char_boxes) == len(uids):
                for i, (box, uid) in enumerate(zip(char_boxes, uids)):
                    if isinstance(box, list) and len(box) == 4:
                        x1, y1, x2, y2 = box
                        char_tag = f"char_{i}"
                        char_text = unicode_to_char(uid) if uid else "?" # Unicode変換失敗時は?
                        self.detail_canvas.add_box(
                            tag=char_tag, x1=x1, y1=y1, x2=x2, y2=y2, color="green", width=1, text=char_text
                        )
                    else:
                        print(f"Warning: Invalid char box format for index {i} in {column_rel_path}: {box}")
            elif char_boxes or uids: # どちらか一方でもあるのに形式が違う場合
                print(f"Warning: Mismatch or invalid format for char_boxes/unicode_ids in {column_rel_path}")
        else:
            print(f"Error: Column image file not found: {col_abs_path}")
            self.detail_canvas.create_text(10, 10, text=f"列画像が見つかりません:\n{col_abs_path}", fill="red", anchor="nw", width=self.detail_canvas.winfo_width()-20 if self.detail_canvas.winfo_width() > 20 else 380)

    def highlight_selected_columns_on_orig(self):
        # (省略 - 元コードと同じ)
        if not hasattr(self.orig_canvas, "boxes"): return
        all_col_tags_on_orig = {box["tag"] for box in self.orig_canvas.boxes if not box["tag"].startswith("char_")}
        self.orig_canvas.selected_box_tags.clear()
        for path in self.selected_column_paths:
            if path in all_col_tags_on_orig:
                self.orig_canvas.selected_box_tags.add(path)
        self.orig_canvas.redraw_boxes()

    def update_button_states(self):
        # (省略 - 文字追加ボタン削除)
        num_selected_cols = len(self.selected_column_paths)
        num_selected_chars = len(self.detail_canvas.get_selected_tags()) if self.current_detail_column_path else 0

        self.merge_button.config(state=tk.NORMAL if num_selected_cols >= 2 else tk.DISABLED)
        self.split_button.config(state=tk.NORMAL if num_selected_cols == 1 and num_selected_chars == 1 and self.current_detail_column_path else tk.DISABLED)

        col_data = self.data_manager.get_column_data(self.current_detail_column_path) if self.current_detail_column_path else None
        total_chars = len(col_data.get("unicode_ids", [])) if col_data else 0
        self.split_selection_button.config(state=tk.NORMAL if num_selected_cols == 1 and self.current_detail_column_path and 0 < num_selected_chars < total_chars else tk.DISABLED)

        self.move_char_button.config(state=tk.NORMAL if num_selected_chars > 0 and self.current_detail_column_path else tk.DISABLED)
        self.delete_col_button.config(state=tk.NORMAL if num_selected_cols > 0 else tk.DISABLED)
        self.delete_char_button.config(state=tk.NORMAL if num_selected_chars > 0 and self.current_detail_column_path else tk.DISABLED)
        # 文字追加ボタンはなくなった
        # self.add_char_button.config(state=tk.NORMAL if self.current_detail_column_path else tk.DISABLED)

        # --- ▼▼▼ モードに応じたカーソル変更 ▼▼▼ ---
        if self.moving_characters_info:
            self.root.config(cursor="crosshair")
        elif self.add_char_mode == 'waiting_first_click' or self.add_char_mode == 'waiting_second_click':
            self.root.config(cursor="plus") # 文字追加のクリック待ち中は十字カーソル
        elif self.add_column_mode == 'waiting_first_click' or self.add_column_mode == 'waiting_second_click': # 列追加モードのカーソル
            self.root.config(cursor="tcross") # 列追加のクリック待ち中は太い十字カーソル
        else:
            self.root.config(cursor="") # 通常カーソル
        # --- ▲▲▲ モードに応じたカーソル変更 ▲▲▲ ---

    # --- ボタンアクション ---
    def merge_selected_columns(self):
        # (省略 - 元コードと同じ)
        if len(self.selected_column_paths) < 2: return
        success = self.data_manager.merge_columns(self.selected_column_paths)
        if success: self.load_current_image_data()

    def split_selected_column(self):
        # (省略 - 元コードと同じ)
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path: return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if len(selected_char_tags) != 1:
            messagebox.showinfo("情報", "分割点を指定するため、詳細表示で文字を1つだけ選択してください。")
            return
        try:
            split_char_index = int(selected_char_tags[0].split("_")[1])
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字のタグからインデックスを取得できませんでした。")
            return
        success = self.data_manager.split_column(self.current_detail_column_path, split_char_index)
        if success:
            self.load_current_image_data()

    def split_column_by_selection(self):
        # (省略 - 元コードと同じ)
        if len(self.selected_column_paths) != 1 or not self.current_detail_column_path:
            messagebox.showerror("エラー", "選択分割を行うには、列リストで列を1つだけ選択してください。")
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            messagebox.showerror("エラー", "分割する文字が選択されていません。")
            return
        try:
            selected_char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            selected_char_indices.sort()
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字タグの解析に失敗しました。")
            return
        col_data = self.data_manager.get_column_data(self.current_detail_column_path)
        if col_data and len(selected_char_indices) == len(col_data.get("unicode_ids", [])):
            messagebox.showerror("エラー", "全ての文字が選択されています。分割できません。")
            return
        confirm = messagebox.askyesno("選択分割確認", f"列 '{Path(self.current_detail_column_path).name}' を、選択された {len(selected_char_indices)} 文字と残りの文字の2つに分割しますか？")
        if confirm:
            success = self.data_manager.split_column_by_selection(self.current_detail_column_path, selected_char_indices)
            if success:
                self.load_current_image_data()

    def initiate_move_character(self):
        # (省略 - 元コードと同じ)
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return
        try:
            char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            char_indices.sort()
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字タグの解析に失敗しました。")
            return
        self.moving_characters_info = {"src_path": self.current_detail_column_path, "char_indices": char_indices}
        messagebox.showinfo("文字移動", f"{len(char_indices)}文字を選択しました。\n移動先の列を「元画像エリア」でクリックしてください。")
        self.update_button_states()

    def handle_orig_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        # (省略 - 元コードと同じ + 列追加モードのクリック処理)
        canvas_x = canvas.canvasx(canvas.winfo_pointerx() - canvas.winfo_rootx())
        canvas_y = canvas.canvasy(canvas.winfo_pointery() - canvas.winfo_rooty())

        # --- ▼▼▼ 列追加モード中の左クリック処理 ▼▼▼ ---
        if self.add_column_mode == 'waiting_first_click':
            self.first_click_pos_orig = (canvas_x, canvas_y)
            self.add_column_mode = 'waiting_second_click'
            print(f"列追加: 1点目クリック @ ({canvas_x:.1f}, {canvas_y:.1f})")
            messagebox.showinfo("列追加", "新しい列の範囲の【右下】をクリックしてください。", parent=self.root)
            return # 通常の列選択は行わない

        elif self.add_column_mode == 'waiting_second_click':
            if not self.first_click_pos_orig:
                print("エラー: 1点目の座標が記録されていません。列追加モードをリセットします。")
                self.add_column_mode = None
                self.update_button_states()
                return

            x1_canvas, y1_canvas = self.first_click_pos_orig
            x2_canvas, y2_canvas = canvas_x, canvas_y
            print(f"列追加: 2点目クリック @ ({x2_canvas:.1f}, {y2_canvas:.1f})")

            # スケールを考慮して元画像上の絶対座標に変換
            scale = canvas.scale
            x1_orig = x1_canvas / scale
            y1_orig = y1_canvas / scale
            x2_orig = x2_canvas / scale
            y2_orig = y2_canvas / scale

            # 座標の順序を正規化 (x1 < x2, y1 < y2)
            final_x1 = min(x1_orig, x2_orig)
            final_y1 = min(y1_orig, y2_orig)
            final_x2 = max(x1_orig, x2_orig)
            final_y2 = max(y1_orig, y2_orig)

            # 整数座標に変換
            new_box_in_original = [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]

            # 小さすぎるボックスは警告
            min_box_dim = 5 # 最小幅/高さ (元画像上)
            if (final_x2 - final_x1 < min_box_dim) or (final_y2 - final_y1 < min_box_dim):
                 if not messagebox.askyesno("確認", f"作成された列範囲が非常に小さいです ({new_box_in_original[2]-new_box_in_original[0]}x{new_box_in_original[3]-new_box_in_original[1]}px)。\nこのまま追加しますか？", parent=self.root):
                     messagebox.showinfo("やり直し", "再度、列範囲の【右下】をクリックしてください。", parent=self.root)
                     return # 処理中断

            print(f"新しい列を範囲 {new_box_in_original} で追加します。")

            # --- ▼▼▼ DataManagerに処理を依頼 (要実装: add_column) ▼▼▼ ---
            current_orig_image = self.data_manager.current_original_image
            if not current_orig_image:
                 messagebox.showerror("エラー", "現在の元画像情報がありません。", parent=self.root)
                 self.add_column_mode = None
                 self.first_click_pos_orig = None
                 self.update_button_states()
                 return

            success = self.data_manager.add_column(current_orig_image, new_box_in_original)
            # --- ▲▲▲ DataManagerに処理を依頼 ▲▲▲ ---


            # モードと状態をリセット
            self.add_column_mode = None
            self.first_click_pos_orig = None
            self.update_button_states() # カーソルを戻す

            if success:
                print("列追加成功。データを再読み込みします。")
                self.load_current_image_data() # 成功したら再描画
            else:
                # DataManager側でエラーメッセージ表示済みのはず (実装後)
                print("列追加に失敗しました。")
            return # 通常の列選択は行わない
        # --- ▲▲▲ 列追加モード中の左クリック処理 ▲▲▲ ---


        # --- ▼▼▼ 通常のクリック処理 (文字移動 or 列選択) ▼▼▼ ---
        if self.moving_characters_info and clicked_tags:
            target_col_path = None
            for tag in clicked_tags:
                # タグが列パスかどうか簡易判定 ( "/" や "\" が含まれるか、 ".jpg" などで終わるか)
                # TODO: より確実な判定方法 (例: DataManager に問い合わせる)
                if ("/" in tag or "\\" in tag) and Path(tag).suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
                    target_col_path = tag
                    break
            if target_col_path:
                print(f"移動先候補: {target_col_path}")
                src_path = self.moving_characters_info["src_path"]
                indices = self.moving_characters_info["char_indices"]
                if src_path == target_col_path:
                    messagebox.showinfo("情報", "同じ列には移動できません。")
                else:
                    confirm = messagebox.askyesnocancel("文字移動確認", f"{len(indices)}個の文字を\nFrom: {Path(src_path).name}\nTo:   {Path(target_col_path).name}\nに移動しますか？")
                    if confirm is True: # Yes
                        success = self.data_manager.move_characters(src_path, target_col_path, indices)
                        if success:
                            self.load_current_image_data()
                    elif confirm is None: # Cancel
                        print("文字移動をキャンセルしました。")
            # 移動モード解除 (成功、失敗、キャンセル問わず)
            self.moving_characters_info = None
            self.update_button_states()
            print("文字移動モード終了")

        elif not self.moving_characters_info:
            # 通常のクリック（列選択とリストボックスの同期）
            selected_tags_on_orig = canvas.get_selected_tags()
            all_columns_df = self.data_manager.get_columns_for_current_image()
            if not all_columns_df.empty:
                def get_sort_key(box):
                    if isinstance(box, list) and len(box) > 0 and isinstance(box[0], (int, float)):
                        return box[0]
                    return float('inf')
                all_columns_df["sort_key"] = all_columns_df["box_in_original"].apply(get_sort_key)
                all_columns_df = all_columns_df.sort_values("sort_key").reset_index(drop=True)
                all_paths = all_columns_df["column_image"].tolist()

                # リストボックスの選択をCanvasと同期
                self.column_listbox.selection_clear(0, tk.END)
                indices_to_select = []
                for tag in selected_tags_on_orig:
                    if tag in all_paths:
                        try:
                            idx = all_paths.index(tag)
                            indices_to_select.append(idx)
                        except ValueError:
                            pass
                for idx in indices_to_select:
                    self.column_listbox.selection_set(idx)

                # 選択状態が変わった可能性があるのでイベント発行
                if set(indices_to_select) != set(self.column_listbox.curselection()):
                     self.column_listbox.event_generate("<<ListboxSelect>>")
                # もしCanvasクリックで何も選択されなくなった場合もイベント発行
                elif not selected_tags_on_orig and self.selected_column_paths:
                     self.column_listbox.event_generate("<<ListboxSelect>>")
                # Listbox選択を更新
                self.selected_column_paths = [all_paths[i] for i in indices_to_select]

            self.update_button_states()

    def handle_detail_canvas_click(self, canvas, clicked_tags, ctrl_pressed):
        # (省略 - 元コードと同じ + 文字追加モードのクリック処理)
        canvas_x = canvas.canvasx(canvas.winfo_pointerx() - canvas.winfo_rootx())
        canvas_y = canvas.canvasy(canvas.winfo_pointery() - canvas.winfo_rooty())

        # --- ▼▼▼ 文字追加モード中の左クリック処理 ▼▼▼ ---
        if self.add_char_mode == 'waiting_first_click':
            self.first_click_pos = (canvas_x, canvas_y)
            self.add_char_mode = 'waiting_second_click'
            print(f"文字追加: 1点目クリック @ ({canvas_x:.1f}, {canvas_y:.1f})")
            messagebox.showinfo("文字追加", "バウンディングボックスの【右下】をクリックしてください。", parent=self.root)
            # update_button_states はカーソル更新のために呼ぶ必要はない (モードが変わっただけ)
            return # 通常の文字選択は行わない

        elif self.add_char_mode == 'waiting_second_click':
            if not self.first_click_pos: # 念のためチェック
                print("エラー: 1点目の座標が記録されていません。文字追加モードをリセットします。")
                self.add_char_mode = None
                self.update_button_states()
                return

            x1_canvas, y1_canvas = self.first_click_pos
            x2_canvas, y2_canvas = canvas_x, canvas_y
            print(f"文字追加: 2点目クリック @ ({x2_canvas:.1f}, {y2_canvas:.1f})")

            # スケールを考慮して列画像内の座標に変換
            scale = canvas.scale
            x1_unscaled = x1_canvas / scale
            y1_unscaled = y1_canvas / scale
            x2_unscaled = x2_canvas / scale
            y2_unscaled = y2_canvas / scale

            # 座標の順序を正規化 (x1 < x2, y1 < y2)
            final_x1 = min(x1_unscaled, x2_unscaled)
            final_y1 = min(y1_unscaled, y2_unscaled)
            final_x2 = max(x1_unscaled, x2_unscaled)
            final_y2 = max(y1_unscaled, y2_unscaled)

            # 整数座標に変換
            new_char_box_in_col = [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]

            # 小さすぎるボックスは警告 (任意)
            min_box_dim = 3 # 最小幅/高さ (非スケール時)
            if (final_x2 - final_x1 < min_box_dim) or (final_y2 - final_y1 < min_box_dim):
                 if not messagebox.askyesno("確認", f"作成されたボックスサイズが非常に小さいです ({new_char_box_in_col[2]-new_char_box_in_col[0]}x{new_char_box_in_col[3]-new_char_box_in_col[1]}px)。\nこのまま追加しますか？", parent=self.root):
                     # キャンセルする場合、モードを1段階戻すか、完全にキャンセルするか選択
                     # ここでは1段階戻す (再度右下をクリックさせる)
                     messagebox.showinfo("やり直し", "再度バウンディングボックスの【右下】をクリックしてください。", parent=self.root)
                     return # 処理中断

            print(f"文字 '{self.pending_unicode_id}' をボックス {new_char_box_in_col} で列 {self.current_detail_column_path} に追加します。")

            # DataManagerに処理を依頼
            success = self.data_manager.add_character(self.current_detail_column_path, new_char_box_in_col, self.pending_unicode_id)

            # モードと状態をリセット
            self.add_char_mode = None
            self.first_click_pos = None
            self.pending_unicode_id = None
            self.update_button_states() # カーソルを戻す

            if success:
                print("文字追加成功。データを再読み込みします。")
                self.load_current_image_data() # 成功したら再描画
            else:
                # DataManager側でエラーメッセージ表示済みのはず
                print("文字追加に失敗しました。")
            return # 通常の文字選択は行わない
        # --- ▲▲▲ 文字追加モード中の左クリック処理 ▲▲▲ ---

        # --- ▼▼▼ 通常の文字選択処理 ▼▼▼ ---
        # print(f"Detail canvas clicked (normal mode). Tags: {clicked_tags}")
        self.update_button_states()
        # --- ▲▲▲ 通常の文字選択処理 ▲▲▲ ---

    # ★変更: 文字追加開始メソッド (詳細エリア右クリック) - ボックス描画対応
    def initiate_add_character(self, canvas, canvas_x, canvas_y):
        """詳細表示エリアが右クリックされたときに文字追加プロセスを開始、またはキャンセル"""
        # --- ▼▼▼ 文字追加モード中の右クリックはキャンセル ▼▼▼ ---
        if self.add_char_mode:
            print("文字追加モードをキャンセルしました (右クリック)。")
            self.add_char_mode = None
            self.first_click_pos = None
            self.pending_unicode_id = None
            self.update_button_states() # カーソルを戻す
            messagebox.showinfo("キャンセル", "文字追加モードをキャンセルしました。", parent=self.root)
            return
        # --- ▲▲▲ 文字追加モード中の右クリックはキャンセル ▲▲▲ ---

        if not self.current_detail_column_path:
            messagebox.showinfo("情報", "文字を追加する列を選択してください。", parent=self.root)
            return

        # すでに文字移動モードの場合は無視
        if self.moving_characters_info:
            return

        # --- ▼▼▼ 文字追加モード開始 ▼▼▼ ---
        self.add_char_mode = 'waiting_unicode'
        print(f"文字追加モード開始 (右クリック位置: {canvas_x:.1f}, {canvas_y:.1f})")

        # Unicode ID を入力させる
        unicode_id = simpledialog.askstring("文字追加", "追加する文字のUnicode IDを入力してください (例: U+4E00):\n(Escキーまたは右クリックでキャンセル)", parent=self.root)

        # Unicode ID 入力中にキャンセルされたかチェック
        if self.add_char_mode != 'waiting_unicode': # Escキーなどでキャンセルされた場合
             print("Unicode入力中にキャンセルされました。")
             return

        if not unicode_id:
            print("文字追加をキャンセルしました (Unicode入力なし)。")
            self.add_char_mode = None # モードリセット
            self.update_button_states()
            return

        # Unicode ID の形式チェックと文字変換
        unicode_id = unicode_id.strip().upper()
        char = unicode_to_char(unicode_id)
        if char is None:
            messagebox.showerror("入力エラー", f"無効なUnicode ID形式です: {unicode_id}\n'U+'に続けて16進数を入力してください。", parent=self.root)
            self.add_char_mode = None # モードリセット
            self.update_button_states()
            return

        # Unicode ID 正常 -> 次のステップへ
        self.pending_unicode_id = unicode_id
        self.add_char_mode = 'waiting_first_click'
        self.update_button_states() # カーソル変更
        messagebox.showinfo("文字追加", f"文字 '{char}' ({unicode_id}) を追加します。\nバウンディングボックスの【左上】をクリックしてください。", parent=self.root)
        # --- ▲▲▲ 文字追加モード開始 ▲▲▲ ---

        # ★注意: ここでは DataManager.add_character は呼び出さない。
        # ボックス座標は左クリックで決定する。


    # ★新規追加: 列追加開始メソッド (元画像エリア右クリック)
    def initiate_add_column(self, canvas, canvas_x, canvas_y):
        """元画像表示エリアが右クリックされたときに列追加プロセスを開始、またはキャンセル"""
        # --- ▼▼▼ 各種モード中の右クリックはキャンセル ▼▼▼ ---
        cancelled = False
        if self.add_char_mode:
            print("文字追加モードをキャンセルしました (右クリック)。")
            self.add_char_mode = None; self.first_click_pos = None; self.pending_unicode_id = None; cancelled = True
        if self.add_column_mode: # 列追加モード中の右クリックもキャンセル
            print("列追加モードをキャンセルしました (右クリック)。")
            self.add_column_mode = None; self.first_click_pos_orig = None; cancelled = True
        if self.moving_characters_info:
             print("文字移動モードをキャンセルしました (右クリック)。")
             self.moving_characters_info = None; cancelled = True

        if cancelled:
            self.update_button_states() # カーソルを戻す
            messagebox.showinfo("キャンセル", "操作をキャンセルしました。", parent=self.root)
            return
        # --- ▲▲▲ 各種モード中の右クリックはキャンセル ▲▲▲ ---

        if not self.data_manager.current_original_image:
            messagebox.showinfo("情報", "元画像が読み込まれていません。", parent=self.root)
            return

        # --- ▼▼▼ 列追加モード開始 ▼▼▼ ---
        self.add_column_mode = 'waiting_first_click'
        self.update_button_states() # カーソル変更
        print(f"列追加モード開始 (右クリック位置: {canvas_x:.1f}, {canvas_y:.1f})")
        messagebox.showinfo("列追加", "新しい列の範囲の【左上】をクリックしてください。\n(Escキーまたは右クリックでキャンセル)", parent=self.root)
        # --- ▲▲▲ 列追加モード開始 ▲▲▲ ---


    def delete_selected_column(self):
        # (省略 - 元コードと同じ)
        if not self.selected_column_paths:
            return
        confirm = messagebox.askyesno("列削除確認", f"{len(self.selected_column_paths)}個の列を削除しますか？\nこの操作は元に戻せません（バックアップも削除されます）。")
        if confirm:
            deleted_count = 0
            # 削除中にリストが変わる可能性があるのでコピー
            paths_to_delete = list(self.selected_column_paths)
            for path in paths_to_delete:
                # get_column_data で存在を確認してから削除
                if self.data_manager.get_column_data(path):
                    success = self.data_manager.delete_column(path)
                    if success:
                        deleted_count += 1
                else:
                    print(f"Skipping deletion of already removed or non-existent column: {path}")

            if deleted_count > 0:
                # load_current_image_data は選択状態もリセットするので、ここで呼ぶ
                self.load_current_image_data()
                messagebox.showinfo("削除完了", f"{deleted_count}個の列を削除しました。")
            else:
                messagebox.showinfo("情報", "削除対象の列が見つからなかったか、削除されませんでした。")


    def delete_selected_character(self):
        # (省略 - 元コードと同じ、複数削除の扱い改善)
        if not self.current_detail_column_path:
            return
        selected_char_tags = self.detail_canvas.get_selected_tags()
        if not selected_char_tags:
            return

        try:
            char_indices = [int(tag.split("_")[1]) for tag in selected_char_tags]
            # 削除はインデックスが大きい方から行う
            char_indices.sort(reverse=True)
        except (IndexError, ValueError):
            messagebox.showerror("エラー", "選択された文字タグの解析に失敗しました。")
            return

        confirm = messagebox.askyesnocancel("文字削除確認", f"{len(char_indices)}個の文字を列 '{Path(self.current_detail_column_path).name}' から削除しますか？")

        if confirm is True: # Yes
            deleted_count = 0
            # 削除処理中に列パスが変わる可能性があるため、最初に保持
            target_column_path = self.current_detail_column_path
            needs_reload = False
            last_error_msg = None

            for index_to_delete in char_indices:
                # 各削除の前に、対象列がまだ存在するか確認
                current_col_data = self.data_manager.get_column_data(target_column_path)
                if not current_col_data:
                    last_error_msg = f"途中で列 {target_column_path} が見つからなくなりました。"
                    break # 列が存在しない場合は中断

                # インデックスが有効範囲内か再確認 (前の削除でインデックスが変わるため)
                current_len = len(current_col_data.get("unicode_ids", []))
                if index_to_delete >= current_len:
                    # 通常ここには来ないはずだが、念のためスキップ
                    print(f"Warning: Index {index_to_delete} is out of bounds after previous deletions. Skipping.")
                    continue

                success = self.data_manager.delete_character(target_column_path, index_to_delete)
                if success:
                    deleted_count += 1
                    # 文字削除で列が再生成された場合、次のループのために列パスを更新する必要があるかもしれない
                    # が、DataManager の実装では常に再生成される。
                    # 確実なのは、一度成功したらリロードすること。
                    needs_reload = True
                    break # 複数選択からの削除は1文字ずつ確認しながら行うか、リロード前提にする。ここではリロード前提。
                else:
                    # DataManager側でエラーメッセージが出ているはず
                    last_error_msg = f"文字インデックス {index_to_delete} の削除中にエラーが発生しました。"
                    needs_reload = False # エラー発生時はリロードしない
                    break

            if needs_reload:
                self.load_current_image_data() # 表示更新
                messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。(画面更新)")
            elif deleted_count > 0:
                # リロードなしで完了した場合 (ありえないはずだが念のため)
                messagebox.showinfo("削除完了", f"{deleted_count}個の文字を削除しました。")
            elif last_error_msg:
                messagebox.showerror("削除エラー", last_error_msg)
            else:
                messagebox.showinfo("情報","文字は削除されませんでした。")

    # --- ナビゲーション ---
    def next_image(self):
        # (省略 - 元コードと同じ、変更保存処理を check_unsaved_changes に任せる)
        if not self.data_manager.original_images:
            return
        # 画像移動前に変更をチェック
        if not self.check_unsaved_changes():
            return # キャンセルされた場合

        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        next_idx = (current_idx + 1) % len(self.data_manager.original_images)
        self.data_manager.set_current_original_image(self.data_manager.original_images[next_idx])
        self.load_current_image_data()

    def prev_image(self):
        # (省略 - 元コードと同じ、変更保存処理を check_unsaved_changes に任せる)
        if not self.data_manager.original_images:
            return
        # 画像移動前に変更をチェック
        if not self.check_unsaved_changes():
            return # キャンセルされた場合

        current_idx = self.data_manager.original_images.index(self.data_manager.current_original_image)
        prev_idx = (current_idx - 1 + len(self.data_manager.original_images)) % len(self.data_manager.original_images)
        self.data_manager.set_current_original_image(self.data_manager.original_images[prev_idx])
        self.load_current_image_data()

    def save_all_changes(self):
        """全ての変更をページ別ファイルに保存"""
        if self.data_manager.changes_made:
            if self.data_manager.save_changes():
                messagebox.showinfo("保存完了", "変更がページ別ファイルに保存されました。")
            # 保存失敗時は DataManager 内でメッセージ表示されるはず
        else:
            messagebox.showinfo("情報", "保存すべき変更はありません。")

    # ★変更: 変更の破棄も選択できるように
    def check_unsaved_changes(self):
        """未保存の変更があるか確認し、ユーザーに尋ねる"""
        if self.data_manager.changes_made:
            response = messagebox.askyesnocancel(
                "未保存の変更",
                "未保存の変更があります。保存しますか？\n"
                "「はい」: 変更を保存して続行\n"
                "「いいえ」: 変更を破棄して続行\n"
                "「キャンセル」: 操作を中断",
                icon=messagebox.WARNING
            )
            if response is True:  # Yes (保存)
                return self.data_manager.save_changes()
            elif response is False:  # No (破棄)
                if self.data_manager.discard_changes():
                    self.load_current_image_data() # 破棄してリロード
                    return True
                else:
                    return False # 破棄失敗
            else:  # Cancel
                return False # 操作中断
        return True  # 変更がない場合は True

    def on_closing(self):
        """ウィンドウを閉じる際の処理"""
        if self.check_unsaved_changes():
            self.data_manager.save_last_state() # 最後に表示した画像を保存
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    # --- データディレクトリの選択 ---
    script_dir = Path(__file__).parent.resolve() # 絶対パスに
    default_data_dir = script_dir / "data"
    if not default_data_dir.exists():
        default_data_dir = script_dir.parent / "data" # 親も探す

    initial_dir = str(default_data_dir) if default_data_dir.exists() else str(script_dir)

    data_dir = filedialog.askdirectory(
        title="データディレクトリを選択してください (例: 'data' フォルダ)", initialdir=initial_dir
    )

    if not data_dir:
        print("データディレクトリが選択されませんでした。終了します。")
        root.destroy()
    else:
        data_path = Path(data_dir)
        # 元のCSVファイルの存在チェック
        csv_path = CSV_PATH
        if not csv_path.exists():
            # 警告を出すが起動は試みる
            messagebox.showwarning(
                "ファイル確認",
                f"元のCSVファイルが見つかりません:\n{csv_path}\n\n"
                "ツールは起動しますが、既存のアノテーションは読み込めません。",
            )

        try:
            # アプリケーションインスタンス生成前にルートウィンドウが存在するか確認
            if root.winfo_exists():
                app = AnnotatorApp(root, data_path)
                # AnnotatorApp の初期化で失敗した場合 (例: DataManager でエラー) は、
                # root.destroy() が呼ばれているはずなので、mainloop は呼ばれない。
                if app and root.winfo_exists(): # appが正常に生成され、windowがまだ存在する場合のみ mainloop
                    root.mainloop()
            else:
                print("Root window was destroyed before creating AnnotatorApp.")

        except Exception as e:
            # AnnotatorApp の初期化中やそれ以前の予期せぬエラー
            import traceback
            error_msg = f"予期せぬエラーが発生し、起動できませんでした:\n{e}\n\n{traceback.format_exc()}"
            print(error_msg)
            # messagebox が使える状態か不明なので、print のみ
            # try:
            #    if root.winfo_exists(): messagebox.showerror("起動エラー", error_msg)
            # except: pass # messagebox表示失敗は無視
            try:
                if root.winfo_exists():
                    root.destroy()
            except:
                pass # destroy 失敗も無視
