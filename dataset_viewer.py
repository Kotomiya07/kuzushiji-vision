# app.py
import io
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

# --- 設定 ---
# データセットが保存されている親ディレクトリを指定
DATA_DIR = Path("data/line_dataset")
DEFAULT_SPLIT = "train"
# ----------------

def get_shards_for_split(split: str) -> list[str]:
    """指定されたsplitに存在するシャードファイルの一覧を取得する"""
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return []
    return sorted([f.name for f in split_dir.glob("*.parquet")])


def load_data_from_shard(split: str, shard: str) -> pd.DataFrame:
    """シャードファイルを読み込み、DataFrameとして返す"""
    if not split or not shard:
        return pd.DataFrame()
    
    shard_path = DATA_DIR / split / shard
    if not shard_path.exists():
        gr.Warning(f"ファイルが見つかりません: {shard_path}")
        return pd.DataFrame()
    
    try:
        return pd.read_parquet(shard_path)
    except Exception as e:
        gr.Error(f"ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame()


def get_paginated_gallery_data(df: pd.DataFrame, page: int, page_size: int) -> list[tuple[Image.Image, str]]:
    """DataFrameから指定されたページのデータを抽出し、Gallery形式に変換する"""
    if df.empty:
        return []
    
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_df = df.iloc[start_index:end_index]
    
    gallery_data = []
    for _, row in paginated_df.iterrows():
        try:
            image = Image.open(io.BytesIO(row["image"]))
            text = str(row["text"])
            gallery_data.append((image, text))
        except Exception:
            # 破損した画像データなどをスキップ
            continue
            
    return gallery_data


# --- Gradioアプリケーションの構築 ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 現在読み込んでいるデータを保持するための非表示コンポーネント
    dataframe_state = gr.State(pd.DataFrame())
    
    gr.Markdown("# くずし字 行データセットビューア")
    gr.Markdown("データセットの分割（split）とシャードを選択して、画像とテキストのペアを確認します。")

    with gr.Row():
        # --- 左パネル: 操作部 ---
        with gr.Column(scale=1):
            with gr.Accordion("コントロールパネル", open=True):
                split_radio = gr.Radio(
                    label="データ分割を選択",
                    choices=["train", "val", "test"],
                    value=DEFAULT_SPLIT,
                    interactive=True
                )
                
                shard_dropdown = gr.Dropdown(
                    label="シャードファイルを選択",
                    choices=get_shards_for_split(DEFAULT_SPLIT),
                    interactive=True
                )
                
                load_button = gr.Button("データを読み込む", variant="primary")
            
            with gr.Accordion("表示設定", open=True):
                page_size_slider = gr.Slider(
                    label="1ページあたりの表示件数",
                    minimum=10,
                    maximum=100,
                    value=20,
                    step=10,
                    interactive=True
                )
                
                page_slider = gr.Slider(
                    label="ページ番号",
                    minimum=1,
                    maximum=1, # データ読み込み後に更新
                    value=1,
                    step=1,
                    interactive=True
                )
            
            with gr.Accordion("選択アイテム詳細", open=True):
                selected_image = gr.Image(label="選択した画像", show_label=False)
                selected_text = gr.Textbox(label="対応するテキスト", lines=5, interactive=False)
        
        # --- 右パネル: データ表示部 ---
        with gr.Column(scale=3):
            status_text = gr.Textbox(label="ステータス", interactive=False)
            gallery = gr.Gallery(
                label="データ一覧",
                show_label=True,
                columns=5,
                object_fit="contain",
                height="auto"
            )

    # --- イベントハンドラ ---

    def handle_split_change(split):
        """splitが変更されたら、シャードの選択肢を更新する"""
        shards = get_shards_for_split(split)
        return gr.Dropdown(choices=shards, value=shards[0] if shards else None)

    split_radio.change(
        fn=handle_split_change,
        inputs=split_radio,
        outputs=shard_dropdown
    )

    def handle_load_data(split, shard, page_size):
        """データを読み込み、最初のページを表示し、ページネーションを更新する"""
        df = load_data_from_shard(split, shard)
        if df.empty:
            total_rows = 0
            max_pages = 1
            status = "データの読み込みに失敗したか、データが空です。"
            gallery_data = []
        else:
            total_rows = len(df)
            max_pages = (total_rows + page_size - 1) // page_size
            status = f"'{split}/{shard}' から {total_rows} 件のデータを読み込みました。"
            gallery_data = get_paginated_gallery_data(df, 1, page_size)
        
        return (
            df,
            gr.Slider(maximum=max_pages, value=1, interactive=True),
            status,
            gallery_data
        )

    load_button.click(
        fn=handle_load_data,
        inputs=[split_radio, shard_dropdown, page_size_slider],
        outputs=[dataframe_state, page_slider, status_text, gallery]
    )

    def handle_pagination_change(df, page, page_size):
        """ページや表示件数が変更されたら、表示を更新する"""
        return get_paginated_gallery_data(df, page, page_size)

    page_slider.change(
        fn=handle_pagination_change,
        inputs=[dataframe_state, page_slider, page_size_slider],
        outputs=gallery
    )
    page_size_slider.change(
        fn=handle_load_data, # ページサイズ変更時はデータ再計算が必要なためloadを再実行
        inputs=[split_radio, shard_dropdown, page_size_slider],
        outputs=[dataframe_state, page_slider, status_text, gallery]
    )
    
    def handle_gallery_select(evt: gr.SelectData, df, page, page_size):
        """ギャラリーで画像が選択されたら、詳細ビューを更新する"""
        if df.empty:
            return None, ""
        
        # 選択された画像のインデックスを計算
        # evt.indexは現在のページでのインデックス
        absolute_index = (page - 1) * page_size + evt.index
        
        selected_row = df.iloc[absolute_index]
        image = Image.open(io.BytesIO(selected_row["image"]))
        text = selected_row["text"]
        
        return image, text

    gallery.select(
        fn=handle_gallery_select,
        inputs=[dataframe_state, page_slider, page_size_slider],
        outputs=[selected_image, selected_text]
    )


if __name__ == "__main__":
    demo.launch()
    
