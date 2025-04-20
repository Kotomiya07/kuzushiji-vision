#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from utils.util import EasyDict  # EasyDictをインポート


def load_config(config_path: str) -> EasyDict:  # 型ヒントをEasyDictに変更
    """設定ファイルを読み込む

    Args:
        config_path (str): 設定ファイルのパス

    Returns:
        EasyDict: 設定内容
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            # yaml.safe_loadの結果をEasyDictでラップ
            return EasyDict(yaml.safe_load(f))
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {str(e)}")
        sys.exit(1)


def process_image(image: Image.Image, model: YOLO, confidence_threshold: float = 0.5) -> Image.Image:
    """画像を処理し、行検出を行う

    Args:
        image (Image.Image): 入力画像
        model (YOLO): YOLOモデル
        confidence_threshold (float): 信頼度の閾値

    Returns:
        Image.Image: 処理結果の画像
    """
    try:
        # 元の画像のコピーを作成
        result_image = image.copy()

        # YOLOモデルで推論実行
        predictions = model(image, device="cpu")[0]  # CPUを明示的に指定

        # 結果の可視化
        draw = ImageDraw.Draw(result_image)

        # 予測結果の取得
        boxes = predictions.boxes.data.cpu().numpy()

        # 検出数をカウント
        valid_detections = boxes[boxes[:, 4] >= confidence_threshold]
        num_detections = len(valid_detections)

        # バウンディングボックスの描画
        for box in boxes:
            x1, y1, x2, y2, conf, _ = box

            if conf >= confidence_threshold:
                # バウンディングボックスを描画
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

                # 信頼度スコアを表示
                draw.text(
                    (x1, y1 - 15),
                    f"{conf:.2f}",
                    fill="blue",
                    font=ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", 80),
                )

        # 検出数を画像の左上に表示
        draw.text(
            (100, 50),
            f"Number of detected lines: {num_detections}",
            fill="blue",
            font=ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", 100),
        )

        return result_image

    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {str(e)}")
        raise gr.Error(f"画像処理に失敗しました: {str(e)}")


def create_gradio_interface(model_path: str):
    """Gradioインターフェースを作成する

    Args:
        model_path (str): モデルファイルのパス
    """
    try:
        # モデルの読み込み（CPU指定）
        model = YOLO(model_path)
        model.cpu()  # 明示的にCPUを指定
        print(f"モデルを読み込みました: {model_path}")

        def predict(image, confidence_threshold):
            if image is None:
                raise gr.Error("画像が指定されていません。")

            try:
                # PILイメージに変換
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)

                # 画像処理と推論
                result = process_image(image, model, confidence_threshold)

                return result

            except Exception as e:
                raise gr.Error(f"予測実行中にエラーが発生しました: {str(e)}")

        # サンプル用のディレクトリを作成
        examples_dir = Path("data/examples")
        examples_dir.mkdir(parents=True, exist_ok=True)

        # サンプル画像をコピー（存在しない場合のみ）
        sample_image_path = examples_dir / "sample.jpg"
        if not sample_image_path.exists():
            shutil.copy("/data/raw/dataset/100241706/images/100241706_00001.jpg", sample_image_path)

        # Gradioインターフェースの作成
        iface = gr.Interface(
            fn=predict,
            inputs=[
                gr.Image(type="pil", label="入力画像", height=512),
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="信頼度閾値",
                    info="検出結果をフィルタリングする閾値（0.0〜1.0）",
                ),
            ],
            outputs=[gr.Image(type="pil", label="検出結果", height=512)],
            title="くずし字行検出",
            description="画像をアップロードすると、くずし字の行を検出して表示します。",
            allow_flagging="never",
            examples=[[str(sample_image_path), 0.5]],
        )

        return iface

    except Exception as e:
        print(f"インターフェース作成中にエラーが発生しました: {str(e)}")
        sys.exit(1)


def main():
    try:
        parser = argparse.ArgumentParser(description="くずし字行検出可視化アプリケーション")
        parser.add_argument("--config", type=str, default="config/inference.yaml", help="設定ファイルのパス")
        parser.add_argument("--model_path", type=str, help="モデルファイルのパス（設定ファイルの値を上書きします）")
        parser.add_argument("--port", type=int, default=7860, help="Gradioインターフェースのポート番号")
        parser.add_argument("--share", action="store_true", help="Gradioインターフェースを外部に公開するかどうか")
        args = parser.parse_args()

        # 設定の読み込み
        config = load_config(args.config)
        print("設定ファイルを読み込みました")

        # モデルパスの設定 (属性アクセスに変更)
        model_path = args.model_path or config.line_extraction.weights_path
        print(f"使用するモデル: {model_path}")

        # モデルファイルの存在確認
        if not os.path.exists(model_path):
            print(f"エラー: モデルファイルが見つかりません: {model_path}")
            sys.exit(1)

        # Gradioインターフェースの作成と起動
        print("Gradioインターフェースを作成しています...")
        iface = create_gradio_interface(model_path)
        print("インターフェースを起動します...")
        if args.share:
            interface_url = iface.launch(share=True, server_port=args.port)
            print(f"インターフェースを公開しました: {interface_url}")
        else:
            iface.launch(
                server_port=args.port,
            )

    except Exception as e:
        print(f"アプリケーション実行中にエラーが発生しました: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
