# ECS 170 Project

This project currently implements:

- **Stage 2:** a multi-layer perceptron (MLP) using PyTorch to classify handwritten digits from grayscale images.
- **Stage 3:** convolutional neural networks (CNNs) using PyTorch to classify images from three datasets: handwritten digits (MNIST), human faces (ORL), and colored objects (CIFAR-10).

## Repository Structure

```text
ECS170-project/
├── local_code/
│   ├── base_class/                         # Shared base classes
│   ├── stage_2_code/
│   │   ├── Dataset_Loader.py               # Loads training and testing data
│   │   ├── Method_MLP.py                   # Defines and trains the neural network
│   │   ├── Evaluate_Metrics.py             # Evaluates model predictions
│   │   ├── Plot_Learning_Curve.py          # Generates training visualizations
│   │   ├── Setting_Train_Test_Given.py     # Manages training and testing
│   │   └── Result_Saver.py                 # Saves model predictions
│   └── stage_3_code/
│       ├── Dataset_Loader.py               # Loads MNIST, ORL, or CIFAR-10
│       ├── Method_CNN.py                   # Shared CNN used by all three models
│       ├── Method_CNN_MNIST.py             # CNN settings for MNIST
│       ├── Method_CNN_ORL.py               # CNN settings for ORL
│       ├── Method_CNN_CIFAR.py             # CNN settings for CIFAR-10
│       ├── Evaluate_Metrics.py             # Evaluates model predictions
│       ├── Plot_Learning_Curve.py          # Generates training visualizations
│       ├── Setting_Train_Test_Given.py     # Manages training and testing
│       ├── Result_Saver.py                 # Saves model predictions
│       └── Result_Loader.py                # Loads saved predictions
├── script/
│   ├── stage_2_script/
│   │   ├── script_mlp.py                   # Runs the main MLP model
│   │   └── script_mlp_experiments.py       # Runs model configuration experiments
│   └── stage_3_script/
│       ├── script_cnn_mnist.py             # Runs the MNIST model
│       ├── script_cnn_orl.py               # Runs the ORL model
│       ├── script_cnn_cifar.py             # Runs the CIFAR-10 model
│       └── script_cnn_experiments.py       # Runs model configuration experiments
├── data/
│   ├── stage_2_data/
│   │   ├── train.csv                       # Training dataset
│   │   └── test.csv                        # Testing dataset
│   └── stage_3_data/
│       ├── MNIST                           # Handwritten digits
│       ├── ORL                             # Human faces
│       └── CIFAR                           # Colored objects
├── figures/
│   ├── stage_2_figures/                    # Figures used in the stage 2 report
│   └── stage_3_figures/                    # Figures used in the stage 3 report
└── result/
    ├── stage_2_result/                     # Model outputs and generated plots
    └── stage_3_result/                     # Model outputs and generated plots
```

## Stage 2 Files

### `local_code/stage_2_code/`

Contains the classes and functions used to build, train, and evaluate the stage 2 model.

- **`Dataset_Loader.py`** — Reads the training and testing CSV files and extracts image features and labels.
- **`Method_MLP.py`** — Defines the neural network architecture, training loop, optimizers, and prediction logic.
- **`Evaluate_Metrics.py`** — Computes classification evaluation metrics.
- **`Plot_Learning_Curve.py`** — Generates plots to visualize training progress.
- **`Setting_Train_Test_Given.py`** — Coordinates data loading, training, testing, and evaluation.
- **`Result_Saver.py`** — Saves model predictions and outputs.

### `script/stage_2_script/`

Contains the executable scripts used to run the stage 2 model.

- **`script_mlp.py`** — Runs the main model training and evaluation pipeline.
- **`script_mlp_experiments.py`** — Runs experiments with different neural network architectures, activation functions, dropout settings, optimizers, learning rates, and loss functions.

### `result/stage_2_result/`

Stores outputs generated during training and evaluation, including saved predictions, learning curves, and experiment summaries.

## Stage 3 Files

### `local_code/stage_3_code/`

Contains the classes and functions used to build, train, and evaluate the stage 3 models.

- **`Dataset_Loader.py`** — Loads the MNIST, ORL, or CIFAR-10 dataset file, scales pixels to the range 0–1, and arranges each image as channels × height × width for PyTorch. ORL images are gray-scale copies stored in three channels, so only one channel is kept, and ORL labels are shifted from 1–40 to 0–39.
- **`Method_CNN.py`** — Defines the shared CNN: convolutional blocks (convolution, optional batch normalization, ReLU, max pooling), then a hidden fully connected layer with dropout and an output layer. Also contains the training loop, optimizers, loss functions, and prediction logic.
- **`Method_CNN_MNIST.py`** — MNIST model: 2 convolutional blocks, 128-neuron hidden layer, 10 outputs.
- **`Method_CNN_ORL.py`** — ORL model: 3 convolutional blocks, 256-neuron hidden layer, 40 outputs.
- **`Method_CNN_CIFAR.py`** — CIFAR-10 model: 3 convolutional blocks with batch normalization, 256-neuron hidden layer, 10 outputs, and a learning rate that decreases over training.
- **`Evaluate_Metrics.py`** — Computes accuracy, precision, recall, and F1 score (macro and weighted averages).
- **`Plot_Learning_Curve.py`** — Generates loss and accuracy plots to visualize training progress.
- **`Setting_Train_Test_Given.py`** — Coordinates data loading, training, testing, and evaluation using the provided train/test split.
- **`Result_Saver.py`** — Saves model predictions and outputs.
- **`Result_Loader.py`** — Loads saved predictions.

### `script/stage_3_script/`

Contains the executable scripts used to run the stage 3 models.

- **`script_cnn_mnist.py`** — Trains and evaluates the MNIST model.
- **`script_cnn_orl.py`** — Trains and evaluates the ORL model.
- **`script_cnn_cifar.py`** — Trains and evaluates the CIFAR-10 model.
- **`script_cnn_experiments.py`** — Runs experiments with different model depths, kernel sizes, padding, stride, pooling, hidden layer sizes, and loss functions on all three datasets.

### `result/stage_3_result/`

Stores outputs generated during training and evaluation, including saved predictions, learning curves, training logs, and experiment summaries.

### Stage 3 Results

| Dataset | Training accuracy | Testing accuracy |
| --- | --- | --- |
| MNIST (handwritten digits) | 99.44% | 98.76% |
| ORL (faces, 40 people) | 100.00% | 97.50% |
| CIFAR-10 (colored objects) | 96.63% | 80.77% |

## Data

The datasets are not included in this repository because of GitHub's file size limits (`data/stage_1_data` only has the toy dataset from stage 1). See the "How to Run" section below for where to place them.

## Technologies

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib

## How to Run

1. Install the required packages:

   ```bash
   pip install torch numpy pandas scikit-learn matplotlib
   ```

2. Download the datasets from the course project description and place them in the following locations:

   ```text
   data/stage_2_data/train.csv
   data/stage_2_data/test.csv
   data/stage_3_data/MNIST
   data/stage_3_data/ORL
   data/stage_3_data/CIFAR
   ```

3. Run the scripts from their stage's script folder with the repository root on the Python path. PyCharm handles this automatically when the repository is opened as a project.

   **Stage 2:**

   ```bash
   cd script/stage_2_script
   PYTHONPATH=../.. python3 script_mlp.py
   PYTHONPATH=../.. python3 script_mlp_experiments.py
   ```

   - **`script_mlp.py`** — Runs the final model, calculates training and testing metrics, and generates plots.
   - **`script_mlp_experiments.py`** — Runs the 12 ablation experiments described in Section 3.7 of the stage 2 report.

   **Stage 3:**

   ```bash
   cd script/stage_3_script
   PYTHONPATH=../.. python3 script_cnn_orl.py          # about 20 seconds
   PYTHONPATH=../.. python3 script_cnn_mnist.py        # about 3 minutes
   PYTHONPATH=../.. python3 script_cnn_cifar.py        # about 8 minutes
   PYTHONPATH=../.. python3 script_cnn_experiments.py  # about 40 minutes
   ```

   - **`script_cnn_mnist.py`**, **`script_cnn_orl.py`**, **`script_cnn_cifar.py`** — Run the final model for each dataset, calculate training and testing metrics, and generate learning curves.
   - **`script_cnn_experiments.py`** — Runs the 30 ablation experiments described in Section 3.7 of the stage 3 report.

   Times are on an Apple M-series Mac using its GPU (MPS). The code uses an Apple or NVIDIA GPU automatically when one is available, otherwise the CPU (slower).

4. Results are saved to:

   ```text
   result/stage_2_result/
   result/stage_3_result/
   ```

   Outputs include model predictions, learning curves, loss curves, and ablation experiment summaries.

## Contributors

- Sruthi Sridhar
- Betty Kang
- Ashwin Charles
