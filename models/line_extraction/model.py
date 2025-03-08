import torch
import torch.nn as nn
from typing import Dict, Tuple
from ultralytics import YOLO

class ColumnDetectionModel(nn.Module):
    """YOLOv8をベースにした列検出モデル"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): モデルの設定
        """
        super().__init__()
        self.config = config
        
        # YOLOv8モデルの初期化
        self.model = YOLO('yolov8n.yaml')  # nanoモデルをベースに使用
        
        # モデル構造の調整
        self.model.model.head.nc = config['model']['num_classes']  # クラス数の調整
        
        # 推論時のパラメータ
        self.conf_threshold = config['model']['confidence_threshold']
        self.nms_threshold = config['model']['nms_threshold']
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """順伝播

        Args:
            images (torch.Tensor): 入力画像 [B, C, H, W]
            targets (Dict[str, torch.Tensor], optional): 
                学習時のターゲット情報
                - boxes: バウンディングボックス [B, N, 4]
                - labels: ラベル [B, N]
                Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: 
                学習時: 損失の辞書
                推論時: 検出結果の辞書
        """
        if self.training and targets is not None:
            # 学習時
            loss_dict = self.model(
                images,
                targets,
                augment=True
            )
            return loss_dict
        else:
            # 推論時
            predictions = self.model(
                images,
                augment=False,
                conf=self.conf_threshold,
                iou=self.nms_threshold
            )
            
            # 予測結果の整形
            boxes = []
            scores = []
            for pred in predictions:
                if len(pred) == 0:
                    # 検出なしの場合
                    boxes.append(torch.zeros((0, 4), device=images.device))
                    scores.append(torch.zeros(0, device=images.device))
                else:
                    boxes.append(pred.boxes.xyxy)
                    scores.append(pred.boxes.conf)
            
            return {
                'boxes': boxes,  # List[Tensor[N, 4]]
                'scores': scores  # List[Tensor[N]]
            }
    
    def load_pretrained(self, weights_path: str = None):
        """事前学習済みの重みを読み込む

        Args:
            weights_path (str, optional): 重みファイルのパス. 
                Noneの場合はCOCOの事前学習済み重みを使用. Defaults to None.
        """
        if weights_path is None:
            # COCOの事前学習済み重みを使用
            self.model = YOLO('yolov8n.pt')
            # クラス数の調整
            self.model.model.head.nc = self.config['model']['num_classes']
        else:
            # 指定された重みを読み込む
            self.model = YOLO(weights_path)
    
    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        score_threshold: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """単一画像に対する推論

        Args:
            image (torch.Tensor): 入力画像 [1, C, H, W]
            score_threshold (float, optional): スコアのしきい値.
                Noneの場合はモデルのデフォルト値を使用. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - boxes: 検出されたバウンディングボックス [N, 4]
                - scores: 検出スコア [N]
        """
        threshold = score_threshold or self.conf_threshold
        predictions = self.model(
            image,
            augment=False,
            conf=threshold,
            iou=self.nms_threshold
        )
        
        if len(predictions[0]) == 0:
            # 検出なしの場合
            return (
                torch.zeros((0, 4), device=image.device),
                torch.zeros(0, device=image.device)
            )
        
        return (
            predictions[0].boxes.xyxy,
            predictions[0].boxes.conf
        ) 