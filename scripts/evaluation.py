import yaml
import torch
from tqdm import tqdm
from typing import Dict, Tuple

from models.column_extraction.model import ColumnDetectionModel
from models.character_detection.model import CharacterDetectionModel
from utils.dataset import ColumnDetectionDataset, CharacterDetectionDataset
from utils.metrics import compute_map, compute_character_accuracy

def evaluate_column_detection(
    model: ColumnDetectionModel,
    dataset: ColumnDetectionDataset,
    config: Dict
) -> Tuple[float, Dict[int, float]]:
    """列検出モデルの評価

    Args:
        model (ColumnDetectionModel): 評価するモデル
        dataset (ColumnDetectionDataset): 評価用データセット
        config (Dict): 評価設定

    Returns:
        Tuple[float, Dict[int, float]]: mAPとクラスごとのAP
    """
    model.eval()
    device = next(model.parameters()).device
    
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc='Evaluating column detection'):
            # データの準備
            images = batch['image'].to(device)
            gt_boxes = batch['boxes']
            
            # 推論
            predictions = model(images)
            pred_boxes = predictions['boxes']
            pred_scores = predictions['scores']
            
            # 結果の収集
            pred_boxes_list.extend(pred_boxes)
            pred_scores_list.extend(pred_scores)
            pred_labels_list.extend([
                torch.zeros_like(score, dtype=torch.long)
                for score in pred_scores
            ])
            gt_boxes_list.extend(gt_boxes)
            gt_labels_list.extend([
                torch.zeros(len(boxes), dtype=torch.long)
                for boxes in gt_boxes
            ])
    
    # mAPの計算
    mAP, class_ap = compute_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    return mAP, class_ap

def evaluate_character_detection(
    model: CharacterDetectionModel,
    dataset: CharacterDetectionDataset,
    config: Dict
) -> Tuple[float, Dict[int, float], float, Dict[int, float]]:
    """文字検出モデルの評価

    Args:
        model (CharacterDetectionModel): 評価するモデル
        dataset (CharacterDetectionDataset): 評価用データセット
        config (Dict): 評価設定

    Returns:
        Tuple[float, Dict[int, float], float, Dict[int, float]]: 
            mAP, クラスごとのAP, 文字認識精度, クラスごとの文字認識精度
    """
    model.eval()
    device = next(model.parameters()).device
    
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc='Evaluating character detection'):
            # データの準備
            images = batch['image'].to(device)
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            # 推論
            predictions = model(images)
            pred_boxes = predictions['boxes']
            pred_scores = predictions['scores']
            pred_labels = predictions['labels']
            
            # 結果の収集
            pred_boxes_list.extend(pred_boxes)
            pred_scores_list.extend(pred_scores)
            pred_labels_list.extend(pred_labels)
            gt_boxes_list.extend(gt_boxes)
            gt_labels_list.extend(gt_labels)
    
    # mAPの計算
    mAP, class_ap = compute_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    # 文字認識精度の計算
    accuracy, class_accuracy = compute_character_accuracy(
        pred_labels_list,
        gt_labels_list,
        pred_boxes_list,
        gt_boxes_list,
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    return mAP, class_ap, accuracy, class_accuracy

def main():
    # 設定の読み込み
    with open('config/evaluation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 列検出モデルの評価
    if config['evaluate_column_detection']:
        print('Evaluating column detection model...')
        
        # モデルの読み込み
        with open('config/model/column_extraction.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        
        model = ColumnDetectionModel(model_config)
        model.load_pretrained(config['column_detection']['weights_path'])
        model.to(device)
        
        # データセットの準備
        dataset = ColumnDetectionDataset(
            image_dir=config['column_detection']['image_dir'],
            annotation_file=config['column_detection']['annotation_file'],
            target_size=model_config['model']['input_size'][0]
        )
        
        # 評価の実行
        mAP, class_ap = evaluate_column_detection(model, dataset, config)
        
        print(f'Column Detection mAP: {mAP:.4f}')
        for class_id, ap in class_ap.items():
            print(f'Class {class_id} AP: {ap:.4f}')
    
    # 文字検出モデルの評価
    if config['evaluate_character_detection']:
        print('\nEvaluating character detection model...')
        
        # モデルの読み込み
        with open('config/model/character_detection.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        
        model = CharacterDetectionModel(model_config)
        model.load_pretrained(config['character_detection']['weights_path'])
        model.to(device)
        
        # データセットの準備
        dataset = CharacterDetectionDataset(
            column_image_dir=config['character_detection']['image_dir'],
            annotation_file=config['character_detection']['annotation_file'],
            target_width=model_config['model']['input_size'][0]
        )
        
        # 評価の実行
        mAP, class_ap, accuracy, class_accuracy = evaluate_character_detection(
            model, dataset, config
        )
        
        print(f'Character Detection mAP: {mAP:.4f}')
        print(f'Character Recognition Accuracy: {accuracy:.4f}')
        
        print('\nPer-class results:')
        for class_id in sorted(class_ap.keys()):
            ap = class_ap[class_id]
            acc = class_accuracy.get(class_id, 0.0)
            print(f'Class {class_id}:')
            print(f'  AP: {ap:.4f}')
            print(f'  Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main() 