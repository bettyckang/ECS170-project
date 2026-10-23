
# Digit Classification with PyTorch

A machine learning project developed for ECS 170 at UC Davis. This project implements a multi-layer perceptron (MLP) using PyTorch to classify handwritten digits from grayscale images.

## Repository Structure

```text
ECS170-project/
├── local_code/
│   ├── base_class/                   # Shared base classes
│   └── stage_2_code/
│       ├── Dataset_Loader.py         # Loads training and testing data
│       ├── Method_MLP.py            # Defines and trains the neural network
│       ├── Evaluate_Metrics.py      # Evaluates model predictions
│       ├── Plot_Learning_Curve.py   # Generates training visualizations
│       ├── Setting_Train_Test_Given.py # Manages training and testing
│       └── Result_Saver.py          # Saves model predictions
├── script/
│   └── stage_2_script/
│       ├── script_mlp.py             # Runs the main MLP model
│       └── script_mlp_experiments.py # Runs model configuration experiments
├── data/
│   └── stage_2_data/
│       ├── train.csv                 # Training dataset
│       └── test.csv                  # Testing dataset
└── result/
    └── stage_2_result/             # Model outputs and generated plots
```

## Folder Descriptions

### `local_code/`
Contains the classes and functions used to build, train, and evaluate the model.

- **`base_class/`** — Shared foundational classes used by the project.
- **`Dataset_Loader.py`** — Reads the training and testing CSV files and extracts image features and labels.
- **`Method_MLP.py`** — Defines the neural network architecture, training loop, optimizers, and prediction logic.
- **`Evaluate_Metrics.py`** — Computes classification evaluation metrics.
- **`Plot_Learning_Curve.py`** — Generates plots to visualize training progress.
- **`Setting_Train_Test_Given.py`** — Coordinates data loading, training, testing, and evaluation.
- **`Result_Saver.py`** — Saves model predictions and outputs.

### `script/`
Contains the executable scripts used to run the project.

- **`script_mlp.py`** — Runs the main model training and evaluation pipeline.
- **`script_mlp_experiments.py`** — Runs experiments with different neural network architectures, activation functions, dropout settings, optimizers, learning rates, and loss functions.

### `data/`
Contains the handwritten digit datasets used by the model.

- **`train.csv`** — Training images and corresponding digit labels.
- **`test.csv`** — Testing images and corresponding digit labels.

Each image consists of 784 pixel values representing a 28 × 28 grayscale image.

### `result/`
Stores outputs generated during training and evaluation, including saved predictions, learning curves, and experiment summaries.

## Technologies

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Contributors

- Sruthi Sridhar
- Betty Kang
- Ashwin Charles

**UC Davis — ECS 170: Artificial Intelligence | Fall 2026**
