"""
YOLO推論Webアプリケーション

FastAPI + htmxを使用したインタラクティブな文字検出推論アプリ
"""

import io
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from ultralytics import YOLO

# アプリ設定
UPLOAD_DIR = Path("outputs/yolo_app_uploads")
RESULT_DIR = Path("outputs/yolo_app_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# デフォルトモデルパス
DEFAULT_MODEL = "experiments/character_detection/yolo12x_14split_new/weights/best.pt"

app = FastAPI(title="YOLO推論アプリ")

# テンプレートと静的ファイル
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

# モデルのグローバルキャッシュ
_model = None


def get_model():
    """モデルを取得（遅延ロード）"""
    global _model
    if _model is None:
        print(f"モデルをロード中: {DEFAULT_MODEL}")
        _model = YOLO(DEFAULT_MODEL)
    return _model


def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """バウンディングボックスを描画"""
    img = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        # スコアに応じて色を変える（高い = 緑、低い = 赤）
        color = (0, int(255 * score), int(255 * (1 - score)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # スコア表示
        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """メインページ"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """画像アップロード"""
    # ファイル保存
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{file_id}{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 画像サイズ取得
    img = Image.open(filepath)
    width, height = img.size

    return {
        "success": True,
        "image_id": file_id,
        "filename": filename,
        "width": width,
        "height": height
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    image_id: str = Form(...),
    filename: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7)
):
    """全体画像で推論"""
    filepath = UPLOAD_DIR / filename

    if not filepath.exists():
        return HTMLResponse("<p>画像が見つかりません</p>")

    # モデルで推論
    model = get_model()
    results = model.predict(str(filepath), conf=conf, iou=iou, verbose=False)[0]

    # 結果画像を生成
    image = cv2.imread(str(filepath))
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        result_image = draw_boxes(image, boxes, scores)
        n_detections = len(boxes)
    else:
        result_image = image
        n_detections = 0

    # 結果画像保存（タイムスタンプでキャッシュ回避）
    timestamp = int(time.time() * 1000)
    result_filename = f"{image_id}_result_{timestamp}.jpg"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), result_image)

    # HTMLレスポンス
    html = f"""
    <div class="result-wrapper" data-count="{n_detections}">
        <img src="/results/{result_filename}" alt="推論結果" class="result-image" />
    </div>
    """
    return HTMLResponse(html)


@app.post("/predict-region", response_class=HTMLResponse)
async def predict_region(
    request: Request,
    image_id: str = Form(...),
    filename: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    """指定領域で推論"""
    filepath = UPLOAD_DIR / filename

    if not filepath.exists():
        return HTMLResponse("<p>画像が見つかりません</p>")

    # 画像読み込み・クロップ
    image = cv2.imread(str(filepath))
    h, w = image.shape[:2]

    # 座標を正規化
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return HTMLResponse("<p>選択範囲が小さすぎます</p>")

    cropped = image[y1:y2, x1:x2]

    # クロップ画像で推論
    model = get_model()
    results = model.predict(cropped, conf=conf, iou=iou, verbose=False)[0]

    # 結果を描画
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        result_cropped = draw_boxes(cropped, boxes, scores)
        n_detections = len(boxes)
    else:
        result_cropped = cropped
        n_detections = 0

    # 結果画像保存（タイムスタンプでキャッシュ回避）
    timestamp = int(time.time() * 1000)
    result_filename = f"{image_id}_region_{timestamp}.jpg"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), result_cropped)

    # HTMLレスポンス
    html = f"""
    <div class="result-wrapper" data-count="{n_detections}" data-region="{x2-x1}x{y2-y1}">
        <img src="/results/{result_filename}" alt="領域推論結果" class="result-image" />
    </div>
    """
    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
