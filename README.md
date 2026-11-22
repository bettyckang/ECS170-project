# ECS 170 Project

This project currently implements:

- **Stage 2:** a multi-layer perceptron (MLP) using PyTorch to classify handwritten digits from grayscale images.
- **Stage 3:** convolutional neural networks (CNNs) using PyTorch to classify images from three datasets: handwritten digits (MNIST), human faces (ORL), and colored objects (CIFAR-10).
- **Stage 4:** recurrent neural networks (RNN, LSTM, and GRU) using PyTorch for two text tasks: classifying IMDb movie reviews as positive or negative, and generating the rest of a joke from its first three words.

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
│   ├── stage_3_code/
│   │   ├── Dataset_Loader.py               # Loads MNIST, ORL, or CIFAR-10
│   │   ├── Method_CNN.py                   # Shared CNN used by all three models
│   │   ├── Method_CNN_MNIST.py             # CNN settings for MNIST
│   │   ├── Method_CNN_ORL.py               # CNN settings for ORL
│   │   ├── Method_CNN_CIFAR.py             # CNN settings for CIFAR-10
│   │   ├── Evaluate_Metrics.py             # Evaluates model predictions
│   │   ├── Plot_Learning_Curve.py          # Generates training visualizations
│   │   ├── Setting_Train_Test_Given.py     # Manages training and testing
│   │   ├── Result_Saver.py                 # Saves model predictions
│   │   └── Result_Loader.py                # Loads saved predictions
│   └── stage_4_code/
│       ├── Text_Utils.py                   # Cleans text, builds vocabularies, loads GloVe
│       ├── Dataset_Loader_Classification.py # Loads the movie reviews
│       ├── Dataset_Loader_Generation.py    # Loads the jokes
│       ├── Method_RNN_Classification.py    # RNN / LSTM / GRU review classifier
│       ├── Method_RNN_Generation.py        # RNN / LSTM / GRU joke generator
│       ├── Evaluate_Metrics.py             # Evaluates review predictions
│       ├── Evaluate_Generation.py          # Compares generated jokes with training jokes
│       ├── Plot_Learning_Curve.py          # Generates training visualizations
│       ├── Setting_Train_Test_Given.py     # Manages training and testing (reviews)
│       ├── Setting_Train_Only.py           # Manages training and generating (jokes)
│       └── Result_Saver.py                 # Saves model predictions
├── script/
│   ├── stage_2_script/
│   │   ├── script_mlp.py                   # Runs the main MLP model
│   │   └── script_mlp_experiments.py       # Runs model configuration experiments
│   ├── stage_3_script/
│   │   ├── script_cnn_mnist.py             # Runs the MNIST model
│   │   ├── script_cnn_orl.py               # Runs the ORL model
│   │   ├── script_cnn_cifar.py             # Runs the CIFAR-10 model
│   │   └── script_cnn_experiments.py       # Runs model configuration experiments
│   └── stage_4_script/
│       ├── script_rnn_classification.py    # Runs the review classifiers
│       ├── script_rnn_generation.py        # Runs the joke generators
│       ├── script_rnn_experiments.py       # Runs model configuration experiments
│       ├── script_plot_convergence.py      # Draws the training loss figure
│       └── script_generate_joke.py         # Writes jokes from your own start words
├── data/
│   ├── stage_2_data/
│   │   ├── train.csv                       # Training dataset
│   │   └── test.csv                        # Testing dataset
│   ├── stage_3_data/
│   │   ├── MNIST                           # Handwritten digits
│   │   ├── ORL                             # Human faces
│   │   └── CIFAR                           # Colored objects
│   └── stage_4_data/
│       ├── text_classification/            # Movie reviews (train/test, pos/neg)
│       ├── text_generation/                # Jokes
│       └── glove/                          # GloVe word vectors
├── figures/
│   ├── stage_2_figures/                    # Figures used in the stage 2 report
│   ├── stage_3_figures/                    # Figures used in the stage 3 report
│   └── stage_4_figures/                    # Figures used in the stage 4 report
└── result/
    ├── stage_2_result/                     # Model outputs and generated plots
    ├── stage_3_result/                     # Model outputs and generated plots
    └── stage_4_result/                     # Model outputs, generated jokes, and plots
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

## Stage 4 Files

### `local_code/stage_4_code/`

Contains the classes and functions used to build, train, and evaluate the stage 4 models.

- **`Text_Utils.py`** — Cleans the movie reviews (lower-case, no HTML, punctuation, numbers, or stop words, but keeping negations like "not"), splits jokes into words and punctuation, builds the vocabularies, and loads the GloVe word vectors.
- **`Dataset_Loader_Classification.py`** — Loads the movie reviews as lists of word IDs, keeping the last 256 words of long reviews. Words seen fewer than 5 times become an unknown token.
- **`Dataset_Loader_Generation.py`** — Loads the jokes as lists of word IDs, with an end-of-joke token added to each joke.
- **`Method_RNN_Classification.py`** — Defines the review classifier: GloVe word vectors, one recurrent layer (RNN, LSTM, or GRU) with 128 units, and an output layer with 2 outputs (positive / negative). Also contains the training loop, optimizer, and prediction logic.
- **`Method_RNN_Generation.py`** — Defines the joke generator: GloVe word vectors, two recurrent layers (RNN, LSTM, or GRU) with 512 units, and an output layer that predicts the next word. Generates a joke from three start words by adding the most likely next word until the end-of-joke token. Can also save and load a trained model.
- **`Evaluate_Metrics.py`** — Computes accuracy, precision, recall, and F1 score.
- **`Evaluate_Generation.py`** — Computes next-word accuracy, precision, recall, and F1, plus how many generated jokes exactly match a training joke and how similar they are.
- **`Plot_Learning_Curve.py`** — Generates loss and accuracy plots to visualize training progress.
- **`Setting_Train_Test_Given.py`** — Coordinates data loading, training, testing, and evaluation using the provided train/test split (reviews).
- **`Setting_Train_Only.py`** — Coordinates data loading, training, generating, and evaluation for the jokes, which have no test split, so generated jokes are compared with the training jokes.
- **`Result_Saver.py`** — Saves model predictions and outputs.

### `script/stage_4_script/`

Contains the executable scripts used to run the stage 4 models.

- **`script_rnn_classification.py`** — Trains and evaluates the RNN, LSTM, and GRU review classifiers.
- **`script_rnn_generation.py`** — Trains the RNN, LSTM, and GRU joke generators, evaluates them, and writes example jokes.
- **`script_rnn_experiments.py`** — Runs experiments with different hidden sizes, numbers of layers, bidirectional reading, dropout, learning rate schedules, review lengths, and word vectors (GloVe, frozen GloVe, or random).
- **`script_plot_convergence.py`** — Draws the training loss figure for all six models (section 3.5 of the stage 4 report).
- **`script_generate_joke.py`** — Writes jokes from start words you type, using the saved LSTM generator. It needs no dataset, GloVe file, or retraining.

### `result/stage_4_result/`

Stores outputs generated during training and evaluation, including scores, learning curves, example jokes, training logs, experiment summaries, and the trained LSTM joke generator (`generation/LSTM_model.pt`).

### Stage 4 Results

| Model | Review training accuracy | Review testing accuracy | Joke next-word accuracy | Joke exact match |
| --- | --- | --- | --- | --- |
| RNN | 88.28% | 83.82% | 72.90% | 7.29% |
| LSTM | 95.72% | 87.03% | 96.52% | 99.15% |
| GRU | 97.51% | 88.01% | 95.10% | 76.43% |

Exact match is the % of three-word starts (823) where the generated joke is word for word a joke from the training data.

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
   data/stage_4_data/text_classification/
   data/stage_4_data/text_generation/
   ```

   Stage 4 also needs the GloVe word vectors. Download `glove.2024.wikigiga.100d.zip` (560 MB) from <https://nlp.stanford.edu/projects/glove/> and unzip it so the text file is at `data/stage_4_data/glove/glove.2024.wikigiga.100d.txt`. The first run saves the vectors it needs to a small cache file next to it, so later runs start faster.

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

   **Stage 4:**

   ```bash
   cd script/stage_4_script
   PYTHONPATH=../.. python3 script_rnn_classification.py   # all three models; or one: ... script_rnn_classification.py lstm
   PYTHONPATH=../.. python3 script_rnn_generation.py       # all three models; or one: ... script_rnn_generation.py lstm
   PYTHONPATH=../.. python3 script_rnn_experiments.py      # about 75 minutes
   PYTHONPATH=../.. python3 script_plot_convergence.py     # after the first two scripts
   PYTHONPATH=../.. python3 script_generate_joke.py "why did the" "my dog is"
   ```

   - **`script_rnn_classification.py`** and **`script_rnn_generation.py`** — Run the final RNN, LSTM, and GRU models, calculate the metrics, and generate learning curves. The generation script also prints example jokes.
   - **`script_rnn_experiments.py`** — Runs the ablation experiments described in Section 3.7 of the stage 4 report.
   - **`script_generate_joke.py`** — Works right away with the saved LSTM generator, without the dataset or GloVe.

   On an Apple M-series Mac, LSTMs train on the GPU (MPS), and the RNN and GRU train on the CPU because they ran faster there. The LSTM models take 2–8 minutes each; the CPU models take longer (the GRU generator took about 80 minutes). GPU training is not bit-for-bit repeatable, so reruns can differ from the reported numbers by a few tenths of a percent.

4. Results are saved to:

   ```text
   result/stage_2_result/
   result/stage_3_result/
   result/stage_4_result/
   ```

   Outputs include model predictions, learning curves, loss curves, and ablation experiment summaries.

## Contributors

- Sruthi Sridhar
- Betty Kang
- Ashwin Charles
