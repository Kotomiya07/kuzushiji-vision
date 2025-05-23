# OCR Model with PyTorch Lightning

This project implements an Optical Character Recognition (OCR) model using PyTorch Lightning. The model is designed to predict text and character-level bounding boxes from images of single lines of text.

## Project Structure

```
.
├── data/column_dataset_padded/ # Expected data directory (not included in repo)
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── bounding_boxes/
│   ├── val/
│   └── test/
├── tests/                      # Unit tests
│   ├── test_data_loader.py
│   ├── test_ocr_model.py
│   └── test_train_script.py
├── config.yaml                 # Configuration file for training parameters
├── data_loader.py              # PyTorch Dataset and DataLoader implementation
├── ocr_model.py                # PyTorch Lightning OCR model definition
├── train.py                    # Main training script
└── README.md                   # This file
```

## Features

*   OCR model predicting both text and bounding boxes.
*   Built with PyTorch Lightning for structured and scalable training.
*   Configurable model architecture (ResNet backbone + RNN decoder).
*   Metrics: Character Error Rate (CER) for text, Intersection over Union (IoU) for bounding boxes.
*   Uses CTC loss for text prediction.
*   Configurable training via `config.yaml` and command-line arguments.
*   Includes unit tests for key components.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is not yet generated, but the core dependencies are:
    *   `torch`
    *   `torchvision`
    *   `pytorch-lightning`
    *   `Pillow` (PIL)
    *   `opencv-python` (optional, for image loading/processing if preferred over Pillow)
    *   `numpy`
    *   `PyYAML` (for config file)
    *   `editdistance` (for CER calculation)
    *   `pytest` (for running tests)

    You can install them manually:
    ```bash
    pip install torch torchvision pytorch-lightning Pillow PyYAML editdistance pytest numpy opencv-python
    ```
    (Note: Specify versions as needed, especially for torch/torchvision/pytorch-lightning compatibility.)

## Data Preparation

The model expects data in the following structure under the directory specified by `data_dir` (default: `./data/column_dataset_padded`):

*   `{split}/images/{BookID}/{image_name}.jpg` (or other image formats)
*   `{split}/labels/{BookID}/{image_name}.txt` (containing the text label)
*   `{split}/bounding_boxes/{BookID}/{image_name}.json` (containing a list of bounding boxes `[x_min, y_min, x_max, y_max]` for each character)

The filenames for image, label, and bounding_boxes must correspond.

**Example `bounding_boxes` JSON (`100241706_00004_2_line_5014.json`):**
```json
[
    [8, 8, 48, 59],
    [8, 59, 45, 97],
    [11, 99, 47, 139],
    [8, 143, 50, 192],
    [15, 218, 50, 263],
    [11, 277, 51, 323],
    [12, 339, 52, 382],
    [15, 394, 56, 445]
]
```

If you don't have the dataset, you can test the scripts using dummy data by running the training script with the `--create_dummy_if_missing` flag (or setting `create_dummy_if_missing: true` in `config.yaml`).

## Training

The main training script is `train.py`.

1.  **Configuration:**
    You can adjust hyperparameters and settings in `config.yaml` or by overriding them with command-line arguments.

2.  **Running Training:**
    ```bash
    python train.py --data_dir path/to/your/data --max_epochs 100 --batch_size 32 --learning_rate 0.0005
    ```
    Or, using a configuration file:
    ```bash
    python train.py --config_path config.yaml
    ```

    To use dummy data if the actual dataset is not present:
    ```bash
    python train.py --create_dummy_if_missing
    ```

    Key command-line arguments (see `python train.py --help` for all options):
    *   `--data_dir`: Path to the dataset.
    *   `--config_path`: Path to a YAML configuration file.
    *   `--learning_rate`: Learning rate.
    *   `--batch_size`: Batch size.
    *   `--max_epochs`: Number of training epochs.
    *   `--rnn_hidden_size`: Hidden size for the RNN.
    *   `--accelerator`: "cpu", "gpu", "auto".
    *   `--run_test`: Run evaluation on the test set after training.
    *   `--create_dummy_if_missing`: Generate a small dummy dataset if `data_dir` is empty/missing.

    Checkpoints will be saved in the directory specified by `checkpoint_dir` in `config.yaml` (default: `./checkpoints/`).

##Running Tests

To run the unit tests:
```bash
pytest tests/
```

## Future Improvements / TODO

*   Generate `requirements.txt`.
*   More sophisticated decoder options (e.g., Transformer).
*   Advanced data augmentation.
*   Hyperparameter tuning utilities.
*   Support for different image input modes (e.g., grayscale).
*   More detailed logging and visualization (e.g., with TensorBoard image logging).
```
