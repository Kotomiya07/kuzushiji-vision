# Character Detection Model Refinement Plan

This document outlines the planned refinements for the `models/character_detection/model.py` based on the initial review and discussion.

## Agreed Refinements

1.  **Modify Backbone Parameter Freezing:**
    *   **Current:** Freeze the first 100 parameters (`list(self.backbone.parameters())[:100]`).
    *   **Change:** Freeze the first **6 layers** of the ViT backbone. This is a more standard approach for transfer learning with Vision Transformers.
    *   **Rationale:** The original parameter count freezing was experimental. Layer-based freezing is more interpretable and common. Freezing the initial layers helps retain general features learned during pre-training.
    *   **Files:** `models/character_detection/model.py`

2.  **Make IoU Threshold Configurable:**
    *   **Current:** Dynamically adjust the IoU threshold in `_compute_detection_loss` based on the current epoch (`max(0.3, min(0.5, 0.3 + 0.2 * self.current_epoch / 50))`).
    *   **Change:** Introduce configuration options in `config/model/character_detection.yaml` to switch between a fixed IoU threshold and the dynamic adjustment.
        *   Add `dynamic_iou_threshold: false` (boolean, default).
        *   Add `fixed_iou_threshold: 0.5` (float, default).
        *   Add `dynamic_iou_params: { start: 0.3, end: 0.5, epochs: 50 }` (dict, optional, for dynamic calculation).
    *   Modify `_compute_detection_loss` to use either the fixed value or the dynamic calculation based on `config.model.dynamic_iou_threshold`.
    *   **Rationale:** The dynamic adjustment was experimental and its effectiveness wasn't confirmed. Providing a switch allows for easy comparison and defaults to the more standard fixed threshold (0.5).
    *   **Files:** `config/model/character_detection.yaml`, `models/character_detection/model.py`

3.  **Clean Up Configuration File:**
    *   **Current:** `config/model/character_detection.yaml` includes `mlp_ratio: 4`.
    *   **Change:** **Remove** the `mlp_ratio` setting.
    *   **Rationale:** The model code (`model.py`) calculates the `intermediate_size` directly using `hidden_size * 4` and does not actually reference `config.model.mlp_ratio`. Removing it simplifies the configuration.
    *   **Files:** `config/model/character_detection.yaml`

4.  **Correct Type Hint for `targets`:**
    *   **Current:** The `forward` method signature has `targets: dict[str, torch.Tensor]`.
    *   **Change:** Update the type hint to `targets: dict[str, list[torch.Tensor]]`.
    *   **Rationale:** The loss computation functions (`_compute_detection_loss`, `_compute_classification_loss`) iterate through the batch and expect `targets['boxes']` and `targets['labels']` to be lists of tensors (one tensor per image in the batch). The type hint should reflect this actual usage.
    *   **Files:** `models/character_detection/model.py`

## Implementation Plan

These changes will be implemented by switching to the `code` mode after user approval of this plan.
