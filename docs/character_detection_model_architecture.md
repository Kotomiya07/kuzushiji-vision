# Character Detection Model Architecture

```mermaid
graph TD
    A["Input Image (B, C, H, W)"] --> B(ViT Backbone: ViTModel);
    B -- last_hidden_state --> C{"Features (B, N, D)"};
    C --> D[Detection Head];
    C --> E[Classification Head];
    D --> F["Detection Output (B, N, 5)"];
    E --> G["Classification Output (B, N, num_classes)"];

    subgraph Backbone
        B
    end

    subgraph Heads
        D("Linear -> ReLU -> Dropout -> Linear(5)")
        E("Linear -> ReLU -> Dropout -> Linear(num_classes)")
    end

    subgraph Training Path
        F & G --> H{Loss Calculation};
        H -- _compute_detection_loss --> I["Detection Loss (L1 + GIoU + Focal)"];
        H -- _compute_classification_loss --> J["Classification Loss (CrossEntropy)"];
        I & J --> K[Total Loss];
    end

    subgraph Inference Path
        F & G --> L{Post Processing};
        L -- _post_process (Filter + NMS) --> M["Output Boxes (List[Tensor])"];
        L -- _post_process (Filter + NMS) --> N["Output Scores (List[Tensor])"];
        L -- _post_process (Filter + NMS) --> O["Output Labels (List[Tensor])"];
    end
```

## Explanation

1.  **Input Image**: Model input image (Batch size B, Channels C, Height H, Width W).
2.  **ViT Backbone**: Uses `ViTModel` from the `transformers` library as the backbone to extract `Features` from the image.
3.  **Features**: Features output from the backbone (Batch size B, Number of patches N, Hidden dimension D).
4.  **Detection Head**: Head that predicts bounding boxes (x1, y1, x2, y2) and confidence from the features.
5.  **Classification Head**: Head that predicts the character class within each bounding box from the features.
6.  **Training Path**: During training, calculates the loss (Detection Loss and Classification Loss) using the outputs of the Detection Head and Classification Head, and the ground truth labels.
7.  **Inference Path**: During inference, post-processes the outputs of the Detection Head and Classification Head (filtering by confidence, Non-Maximum Suppression (NMS)) to output the final bounding boxes, scores, and labels.
