"""列検出モデルをテストするGUIアプリケーション

このスクリプトは、train_yolov12_column.pyで学習した列検出モデルをテストするためのGradioアプリケーションです。
画像をアップロードして列を検出し、結果を可視化します。
"""

import base64
import io
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from yolov12.ultralytics import YOLO

# --- ユーザー設定 (★ご自身の環境に合わせて変更してください) ---
# デフォルトのモデルパス（最新の実験ディレクトリから自動検出を試みる）
DEFAULT_MODEL_PATH = None  # Noneの場合は最新の実験ディレクトリから自動検出
CONF_THRESHOLD = 0.25
# --- 設定ここまで ---

# --- グローバル変数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None


def find_latest_model() -> Path | None:
    """最新の学習済みモデルを検索"""
    experiments_dir = Path("experiments/yolov12_column")
    if not experiments_dir.exists():
        return None

    # タイムスタンプディレクトリを検索
    model_dirs = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            weights_path = exp_dir / "weights" / "best.pt"
            if weights_path.exists():
                model_dirs.append((exp_dir.name, weights_path))

    if not model_dirs:
        return None

    # 最新のディレクトリを返す（タイムスタンプ順）
    model_dirs.sort(key=lambda x: x[0], reverse=True)
    return model_dirs[0][1]


def load_model(model_path: str | None = None):
    """アプリケーション起動時にモデルをロードする"""
    global MODEL

    # モデルパスの決定
    if model_path is None or model_path == "":
        if DEFAULT_MODEL_PATH and Path(DEFAULT_MODEL_PATH).exists():
            model_path = DEFAULT_MODEL_PATH
        else:
            # 最新のモデルを自動検出
            auto_path = find_latest_model()
            if auto_path:
                model_path = str(auto_path)
            else:
                gr.Warning("モデルファイルが見つかりません。experiments/yolov12_column/ 以下に学習済みモデルがあることを確認してください。")
                MODEL = None
                return

    if not Path(model_path).exists():
        gr.Warning(f"モデルファイルが見つかりません: {model_path}")
        MODEL = None
        return

    try:
        MODEL = YOLO(model_path).to(DEVICE)
        gr.Info(f"列検出モデルが正常にロードされました: {model_path}")
    except Exception as e:
        gr.Error(f"モデルのロード中にエラー: {e}")
        MODEL = None


def perform_yolo_detection(image: Image.Image, confidence_threshold: float) -> list[dict]:
    """YOLOモデルを使用して列を検出"""
    if MODEL is None:
        return []

    try:
        results = MODEL.predict(image, conf=confidence_threshold, verbose=False)
        detections = []

        if results and results[0].boxes:
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = box
                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                    }
                )
        return detections
    except Exception as e:
        print(f"検出エラー: {e}")
        return []


def draw_boxes(image: Image.Image, detections: list[dict], color: str = "red") -> Image.Image:
    """検出結果を画像に描画"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font = ImageFont.truetype("assets/fonts/fonts-japanese-gothic.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]

        # 矩形を描画
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 信頼度と列番号を表示
        label_text = f"Col{i+1}: {confidence:.2f}"
        text_x, text_y = x2 + 5, y1
        draw.text((text_x, text_y), label_text, fill="black", font=font)

    return draw_image


def visualize_json_data(json_string: str, base_image: Image.Image) -> Image.Image | None:
    """JSONデータを受け取り、画像のサイズに合わせてスケーリングして矩形を描画する"""
    if not json_string.strip() or not base_image:
        return None

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        gr.Warning("JSONの形式が正しくありません。")
        return None

    draw_image = base_image.copy()
    image_w, image_h = base_image.size
    draw = ImageDraw.Draw(draw_image)

    try:
        font_size = max(15, int(image_h / 60))
        font = ImageFont.truetype("assets/fonts/fonts-japanese-gothic.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # 新しいJSON形式から画像サイズを抽出
    json_w, json_h = None, None
    if "imginfo" in data and isinstance(data["imginfo"], dict):
        json_w = data["imginfo"].get("img_width")
        json_h = data["imginfo"].get("img_height")

    scale_x, scale_y = (image_w / json_w, image_h / json_h) if json_w and json_h else (1.0, 1.0)
    if scale_x == 1.0 and scale_y == 1.0:
        gr.Info("JSONに'imginfo'と'img_width'/'img_height'キーがないか、画像サイズが一致しているため、アノテーションのスケーリングは行われません。")

    # "contents"キーの存在を確認
    if "contents" not in data or not isinstance(data["contents"], list):
        gr.Warning("JSONに'contents'キーがないか、リストではありません。")
        return None

    # "contents"内の各矩形情報を処理
    for i, content in enumerate(data["contents"]):
        if not isinstance(content, list) or len(content) < 4:
            continue

        # 列検出の場合は、x1, y1, x2, y2の4要素のみを想定
        if len(content) >= 4:
            x1, y1, x2, y2 = content[:4]

            if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
                continue

            # スケーリングを適用
            x, y, w, h = x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y

            # 矩形を描画
            draw.rectangle([x, y, x + w, y + h], outline="green", width=3)

            # 列番号を表示
            label_text = f"Col{i+1}"
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:  # 古いPillowバージョン用のフォールバック
                text_w, text_h = draw.textsize(label_text, font=font)

            draw_x = x + (w - text_w) // 2
            draw_y = y + 5
            draw.text((draw_x, draw_y), label_text, fill="blue", font=font)

    return draw_image


def resize_image(image: Image.Image, height: int = 1280) -> Image.Image:
    """画像を指定した高さにリサイズ（アスペクト比を保持）"""
    if image is None:
        return None
    return image.resize((int(height * image.width / image.height), height), Image.Resampling.LANCZOS)


def predict(
    uploaded_image: Image.Image,
    json_string: str,
    confidence_threshold: float,
    model_path: str | None,
):
    """推論とJSON可視化を同時に実行する"""
    # モデルがロードされていない場合は再試行
    if MODEL is None:
        if model_path:
            load_model(model_path)
        if MODEL is None:
            gr.Warning("モデルがロードされていません。モデルパスを指定してください。")
            return None, None, None, gr.update(visible=False), None, gr.update(visible=False)

    if uploaded_image is None:
        gr.Info("画像をアップロードしてください。")
        return None, None, None, gr.update(visible=False), None, gr.update(visible=False)

    image = uploaded_image.convert("RGB")

    # 列検出を実行
    detections = perform_yolo_detection(image, confidence_threshold)

    # 検出結果を可視化
    pred_img = draw_boxes(image.copy(), detections, color="red")

    # JSON可視化
    json_vis_img = visualize_json_data(json_string, image.copy())
    json_zoom_visible = gr.update(visible=True) if json_vis_img is not None else gr.update(visible=False)

    # 検出結果のサマリー
    summary_text = f"検出された列数: {len(detections)}\n\n"
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        summary_text += f"列{i+1}: 信頼度={confidence:.3f}, 座標=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n"

    return (
        resize_image(pred_img),
        resize_image(json_vis_img) if json_vis_img else None,
        pred_img,
        gr.update(visible=True),
        json_vis_img,
        json_zoom_visible,
        summary_text,
    )


def open_in_new_tab(img):
    """画像を新しいタブで開くためのデータURIを生成"""
    if not isinstance(img, Image.Image):
        return ""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def create_ui():
    """GradioのUIを構築する"""
    open_tab_js = """
    (uri) => {
        if (!uri) { return; }
        const newWindow = window.open('', '_blank');
        if (newWindow) {
            newWindow.document.write(`<html><head><title>拡大表示</title><style>body { margin: 0; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; min-height: 100vh; } img { max-width: 100%; max-height: 100vh; }</style></head><body><img src="${uri}"></body></html>`);
            newWindow.document.close();
        } else { alert('ポップアップがブロックされました。'); }
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# YOLOv12 列検出モデル テストツール")
        gr.Markdown("画像をアップロードし、列検出モデルで列を検出します。")

        full_pred_img_state = gr.State()
        full_json_img_state = gr.State()
        data_uri_state = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(type="pil", label="画像をアップロード")
                json_input = gr.Textbox(
                    label="可視化するJSONデータを貼り付け（オプション）",
                    lines=5,
                    placeholder='{"contents": [[x1, y1, x2, y2], ...], "imginfo": {"img_width": 1000, "img_height": 2000}}',
                )
                run_button = gr.Button("推論実行", variant="primary")

            with gr.Column(scale=1):
                with gr.Accordion("推論設定", open=True):
                    model_path_input = gr.Textbox(
                        label="モデルパス",
                        placeholder="experiments/yolov12_column/YYYYMMDD_HHMMSS/weights/best.pt",
                        value="",
                        info="空欄の場合は最新のモデルを自動検出します",
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=CONF_THRESHOLD,
                        step=0.01,
                        label="信頼度しきい値",
                        info="値が低いほど、より多くの列が検出されます",
                    )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 予測結果（AI）")
                output_pred = gr.Image(label="Prediction (Red)", type="pil")
                zoom_button = gr.Button("🔍 予測結果を拡大", visible=False)

            with gr.Column(scale=1):
                gr.Markdown("### 可視化結果（JSON）")
                output_json_vis = gr.Image(label="Visualization from JSON (Green)", type="pil")
                json_zoom_button = gr.Button("🔍 可視化結果を拡大", visible=False)

            with gr.Column(scale=1):
                gr.Markdown("### 検出結果サマリー")
                output_summary = gr.Textbox(label="検出された列の情報", lines=15, interactive=False)

        # モデルロード
        demo.load(load_model, inputs=[model_path_input], outputs=[])

        # 推論実行
        run_button.click(
            fn=predict,
            inputs=[image_input, json_input, confidence_slider, model_path_input],
            outputs=[
                output_pred,
                output_json_vis,
                full_pred_img_state,
                zoom_button,
                full_json_img_state,
                json_zoom_button,
                output_summary,
            ],
        )

        # 拡大表示
        zoom_button.click(fn=open_in_new_tab, inputs=[full_pred_img_state], outputs=[data_uri_state]).then(
            None, inputs=[data_uri_state], js=open_tab_js
        )
        json_zoom_button.click(fn=open_in_new_tab, inputs=[full_json_img_state], outputs=[data_uri_state]).then(
            None, inputs=[data_uri_state], js=open_tab_js
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch()

