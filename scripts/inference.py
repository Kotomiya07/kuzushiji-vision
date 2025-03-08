import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.column_extraction.model import ColumnDetectionModel
from models.character_detection.model import CharacterDetectionModel
from utils.visualization import visualize_detections

class KuzushijiRecognizer:
    """くずし字認識の推論パイプライン"""
    
    def __init__(
        self,
        column_model_config: str,
        column_model_weights: str,
        char_model_config: str,
        char_model_weights: str,
        device: torch.device = None
    ):
        """
        Args:
            column_model_config (str): 列検出モデルの設定ファイルパス
            column_model_weights (str): 列検出モデルの重みファイルパス
            char_model_config (str): 文字検出モデルの設定ファイルパス
            char_model_weights (str): 文字検出モデルの重みファイルパス
            device (torch.device, optional): 使用するデバイス. Defaults to None.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 列検出モデルの準備
        with open(column_model_config, 'r') as f:
            column_config = yaml.safe_load(f)
        
        self.column_model = ColumnDetectionModel(column_config)
        checkpoint = torch.load(column_model_weights, map_location=self.device)
        self.column_model.load_state_dict(checkpoint['model_state_dict'])
        self.column_model.to(self.device)
        self.column_model.eval()
        
        # 文字検出モデルの準備
        with open(char_model_config, 'r') as f:
            char_config = yaml.safe_load(f)
        
        self.char_model = CharacterDetectionModel(char_config)
        checkpoint = torch.load(char_model_weights, map_location=self.device)
        self.char_model.load_state_dict(checkpoint['model_state_dict'])
        self.char_model.to(self.device)
        self.char_model.eval()
        
        # 設定の保存
        self.column_config = column_config
        self.char_config = char_config
    
    @torch.no_grad()
    def process_image(
        self,
        image: Image.Image,
        visualize: bool = False
    ) -> Tuple[List[Dict], Image.Image]:
        """画像を処理して文字を検出

        Args:
            image (Image.Image): 入力画像
            visualize (bool, optional): 可視化するかどうか. Defaults to False.

        Returns:
            Tuple[List[Dict], Image.Image]: 
                - 検出結果のリスト（列ごと）
                - 可視化結果の画像（visualize=Trueの場合）
        """
        # 画像の前処理
        orig_w, orig_h = image.size
        image = image.convert('RGB')
        
        # 列の検出
        column_image = image.resize(
            (self.column_config['model']['input_size'][0],) * 2,
            Image.LANCZOS
        )
        column_tensor = torch.from_numpy(np.array(column_image)).permute(2, 0, 1)
        column_tensor = column_tensor.float() / 255.0
        column_tensor = column_tensor.unsqueeze(0).to(self.device)
        
        column_predictions = self.column_model(column_tensor)
        column_boxes = column_predictions['boxes'][0]
        column_scores = column_predictions['scores'][0]
        
        # スケールを元のサイズに戻す
        scale_w = orig_w / self.column_config['model']['input_size'][0]
        scale_h = orig_h / self.column_config['model']['input_size'][0]
        column_boxes[:, [0, 2]] *= scale_w
        column_boxes[:, [1, 3]] *= scale_h
        
        results = []
        for column_box, column_score in zip(column_boxes, column_scores):
            # 列領域の切り出し
            x1, y1, x2, y2 = map(int, column_box.tolist())
            column_image = image.crop((x1, y1, x2, y2))
            
            # 文字の検出
            char_image = column_image.resize(
                (self.char_config['model']['input_size'][0], None),
                Image.LANCZOS
            )
            char_tensor = torch.from_numpy(np.array(char_image)).permute(2, 0, 1)
            char_tensor = char_tensor.float() / 255.0
            char_tensor = char_tensor.unsqueeze(0).to(self.device)
            
            char_predictions = self.char_model(char_tensor)
            char_boxes = char_predictions['boxes'][0]
            char_scores = char_predictions['scores'][0]
            char_labels = char_predictions['labels'][0]
            
            # スケールを元のサイズに戻す
            scale_w = (x2 - x1) / self.char_config['model']['input_size'][0]
            scale_h = scale_w  # アスペクト比は保持
            char_boxes[:, [0, 2]] *= scale_w
            char_boxes[:, [1, 3]] *= scale_h
            
            # 列の座標系に変換
            char_boxes[:, [0, 2]] += x1
            char_boxes[:, [1, 3]] += y1
            
            # 結果の保存
            results.append({
                'column_box': column_box.tolist(),
                'column_score': column_score.item(),
                'char_boxes': char_boxes.tolist(),
                'char_scores': char_scores.tolist(),
                'char_labels': char_labels.tolist()
            })
        
        # 可視化
        if visualize:
            vis_image = visualize_detections(
                image,
                results,
                self.char_config['model']['label_map']
            )
            return results, vis_image
        
        return results, None

def main():
    # 設定の読み込み
    with open('config/inference.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 認識器の準備
    recognizer = KuzushijiRecognizer(
        column_model_config='config/model/column_extraction.yaml',
        column_model_weights=config['column_detection']['weights_path'],
        char_model_config='config/model/character_detection.yaml',
        char_model_weights=config['character_detection']['weights_path']
    )
    
    # 入力画像の処理
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    for image_path in tqdm(image_paths, desc='Processing images'):
        # 画像の読み込みと処理
        image = Image.open(image_path)
        results, vis_image = recognizer.process_image(
            image,
            visualize=config['visualization']['enabled']
        )
        
        # 結果の保存
        output_name = image_path.stem
        
        # 検出結果の保存
        with open(output_dir / f'{output_name}_results.yaml', 'w') as f:
            yaml.dump(results, f)
        
        # 可視化結果の保存
        if vis_image is not None:
            vis_image.save(output_dir / f'{output_name}_vis.png')

if __name__ == '__main__':
    main() 