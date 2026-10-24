# ECS 170 Project

This project currently implements stage 2, a multi-layer perceptron (MLP) using PyTorch to classify handwritten digits from grayscale images.

## Repository Structure

```text
ECS170-project/
├── local_code/
│   ├── base_class/                     # Shared base classes
│   └── stage_2_code/
│       ├── Dataset_Loader.py           # Loads training and testing data
│       ├── Method_MLP.py               # Defines and trains the neural network
│       ├── Evaluate_Metrics.py         # Evaluates model predictions
│       ├── Plot_Learning_Curve.py      # Generates training visualizations
│       ├── Setting_Train_Test_Given.py # Manages training and testing
│       └── Result_Saver.py             # Saves model predictions
├── script/
│   └── stage_2_script/
│       ├── script_mlp.py               # Runs the main MLP model
│       └── script_mlp_experiments.py   # Runs model configuration experiments
├── data/
│   └── stage_2_data/
│       ├── train.csv                   # Training dataset
│       └── test.csv                    # Testing dataset
└── result/
    └── stage_2_result/                 # Model outputs and generated plots
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

Contains the executable scripts used to run stage 2 model.

- **`script_mlp.py`** — Runs the main model training and evaluation pipeline.
- **`script_mlp_experiments.py`** — Runs experiments with different neural network architectures, activation functions, dropout settings, optimizers, learning rates, and loss functions.

### `data/`

Currently the `train.csv` and `test.csv` files are not in our data folder, only the toy dataset from stage 1 due to size limitations. See the "How to run" section below

### `result/stage_2_result/`

Stores outputs generated during training and evaluation, including saved predictions, learning curves, and experiment summaries.

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

2. Download the Stage 2 dataset from the course project description. The dataset is not included in this repository because `train.csv` exceeds GitHub's 100 MB file limit.

   Place the files in the following locations:

   ```text
   data/stage_2_data/train.csv
   data/stage_2_data/test.csv
   ```

3. Run the scripts from the `stage_2_script` folder with the repository root on the Python path. PyCharm handles this automatically when the repository is opened as a project.

   ```bash
   cd script/stage_2_script
   PYTHONPATH=../.. python3 script_mlp.py
   PYTHONPATH=../.. python3 script_mlp_experiments.py
   ```

   - **`script_mlp.py`** — Runs the final model, calculates training and testing metrics, and generates plots.
   - **`script_mlp_experiments.py`** — Runs the 12 ablation experiments described in Section 3.7 of the project report.

4. Results are saved to:

   ```text
   result/stage_2_result/
   ```

   Outputs include model predictions, learning curves, loss curves, and ablation experiment summaries.

## Contributors

- Sruthi Sridhar
- Betty Kang
- Ashwin Charles
