
# RetinexNet-PyTorch

This project is an implementation of the RetinexNet model in PyTorch, designed for low-light image enhancement. The model aims to enhance image quality by addressing issues in low-light environments.

## Features

- **Training and Testing**: The script supports both training (`train` phase) and testing (`test` phase) modes for the RetinexNet model.
- **Customizable Hyperparameters**: Options for batch size, learning rate, patch size, number of epochs, and more.
- **GPU Support**: Utilizes GPU if available, with configurable memory usage.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TuyamPandey/RetinexNet-Pytorch.git
   cd RetinexNet-Pytorch
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run:

```bash
python main.py --phase train
```

### Testing

To test the model with a pre-trained checkpoint, run:

```bash
python main.py --phase test --test_dir ./data/test/low
```

### Command-Line Options

- `--use_gpu`: Flag for GPU usage (1 for GPU, 0 for CPU).
- `--gpu_idx`: GPU index to use (default: "0").
- `--epoch`: Number of epochs for training.
- `--batch_size`: Number of samples in each batch.
- `--start_lr`: Initial learning rate.
- `--test_dir`: Directory for testing inputs.

## Directory Structure

- `checkpoint`: Directory for saving checkpoints.
- `data`: Contains training and testing data.
- `sample`: Directory for evaluating model outputs.
- `test_results`: Directory for saving testing results.

## Model

The `LowlightEnhance` model, defined in `model.py`, performs low-light enhancement by learning decomposition and relighting tasks.
