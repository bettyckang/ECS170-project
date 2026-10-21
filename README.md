# ECS 170 Fall 2026 Course Project: Stage 2 Report

## Team Information

**Team name:** [Enter team name]

| Student | Name | Student ID | Email |
| --- | --- | --- | --- |
| 1 | [Name] | [ID] | [Email] |
| 2 | [Name] | [ID] | [Email] |
| 3 | [Name] | [ID] | [Email] |
| 4 | [Name] | [ID] | [Email] |
| 5 | [Name] | [ID] | [Email] |

## Section 1: Task Description

We trained a multi-layer perceptron (MLP) in PyTorch that classifies 28×28 grayscale images of handwritten digits into 10 classes (0–9), reaching 98.44% test accuracy. Each image is given as 784 pixel intensities, and the model learns a mapping from those features to a digit label.

The work follows the course code template: we wrote a new dataset loader, a setting for the pre-partitioned train/test data, a 10-class MLP, and multi-class evaluation metrics. We then varied depth, dropout, activation, optimizer, loss and learning rate to see how each affects performance.

## Section 2: Model Description

Our model is a fully connected MLP with two hidden layers of 256 and 128 units, implemented as `Method_MLP` on top of the template's `method` base class.

![Baseline MLP architecture: 3 linear layers, 235,146 parameters](figures/architecture.png)

Each part of the model has a specific job:

- **Linear layers:** each unit computes a weighted sum of all its inputs plus a bias (w·x + b). These weights are what training learns: the pixels that matter for a digit get larger weights.
- **Two hidden layers (256, then 128 units):** one linear layer can only separate digits with straight-line boundaries. Stacking layers lets the network build complex patterns out of simpler ones, and shrinking the sizes toward 10 condenses the information step by step.
- **ReLU activation, max(0, x):** without it, any stack of linear layers collapses into a single linear layer. ReLU adds the bend that lets the network learn curved boundaries. It is cheap to compute and, unlike sigmoid or tanh, does not flatten out for large inputs, so gradients stay strong.
- **Dropout (20%):** on each training step, it switches off a random 20% of hidden units. The model then cannot lean on any single unit to memorize training images. Dropout is off at test time.
- **Output layer (10 units):** produces one score, called a logit, for each digit 0–9.
- **Softmax with cross-entropy (training):** softmax turns the 10 scores into probabilities that sum to 1. Cross-entropy is small when the correct digit gets high probability and large for confident mistakes, so minimizing it pushes the model toward the right answer.
- **Argmax (testing):** the digit with the highest score is the prediction.

The model has 235,146 trainable parameters: 200,960 in hidden layer 1, 32,896 in hidden layer 2 and 1,290 in the output layer.

## Section 3: Experiment Settings

### 3.1 Dataset Description

The instructor's Stage 2 dataset comes already split into 60,000 training images (86%) and 10,000 testing images (14%). Each line of `train.csv` and `test.csv` holds 785 integers: the label (0–9), then 784 pixel values from 0 to 255.

| Split | Images | Share | Used for |
| --- | --- | --- | --- |
| Training | 60,000 | 86% | Learning the weights; the model sees every image each epoch |
| Testing | 10,000 | 14% | Measuring accuracy on images the model has never seen |

Why it is split this way:

- **A separate test set:** a model can score near 100% on images it has memorized, so accuracy only means something on unseen images. Our model scores 99.48% on training images but 98.44% on test images; a held-out set exposes that gap.
- **Most images go to training:** handwriting varies widely between writers, and the model learns that variety only from examples. More training images means better accuracy on new handwriting.
- **10,000 test images is already enough:** each test image is worth 0.01 percentage points, and our accuracy has a standard error of only about 0.12 points. A larger test set would barely sharpen that, but would take examples away from training.
- **Balanced digits:** each digit makes up 9–11% of both sets, so the test checks every digit fairly.
- **A fixed split:** every team is scored on the same test images, so we used no further splitting or cross validation.

### 3.2 Detailed Experimental Setups

| Setting | Value | Why it matters |
| --- | --- | --- |
| Input scaling | Pixels ÷ 255, so inputs lie in [0, 1] | Keeps all inputs small and on one scale, so training is stable and converges faster |
| Layer dimensions | 784 → 256 → 128 → 10 | 784 inputs match the pixels and 10 outputs match the digits; two hidden layers reach 98%+ in under a minute |
| Activation | ReLU after each hidden layer | Lets the network learn non-linear patterns; beat tanh and sigmoid by about 0.2 points (3.7) |
| Dropout | 0.2 after each hidden layer, training only | Discourages memorizing training images; removing it cost 0.10 points (3.7) |
| Output | 10 raw scores (logits) | One score per digit; the highest score is the prediction |
| Loss | Cross-entropy | The standard loss for choosing one of several classes; it penalizes confident wrong answers most |
| Optimizer | Adam, learning rate 0.001 | Adapts the step size for each weight, so it converges quickly; a 10× larger rate lost about 1 point (3.7) |
| Batch size | 128 | 469 weight updates per epoch instead of 1, so it learns faster than full-batch and fits easily in memory |
| Epochs | 30, data reshuffled every epoch | Loss has flattened by about epoch 20 (3.5); reshuffling keeps the model from learning the data's order |
| Initialization | PyTorch default for `nn.Linear` (Kaiming-uniform) | Starts weights at small random values scaled to each layer's input size, so signals neither blow up nor die out |
| Random seed | 2 (NumPy and PyTorch) | Makes runs repeatable, so differences in 3.7 come from the settings, not luck |
| Hardware | Apple M-series CPU; one run on the Apple GPU (MPS) | CPU is enough: 30 epochs train in under a minute |

We removed the template's final softmax layer because `nn.CrossEntropyLoss` already applies log-softmax, and applying softmax twice weakens the gradients. We also switched from full-batch to mini-batch training, which suits a 60,000-instance training set. Only one dataset is used, so there is one set of settings.

### 3.3 Evaluation Metrics

We report Accuracy, Precision, Recall and F1 on the test set; each answers a different question. The examples use our final model's test results for digit 8, its weakest digit.

| Metric | Question it answers | Example from our model |
| --- | --- | --- |
| Accuracy | Of all test images, how many did the model get right? | 9,844 of 10,000 correct (98.44%); 156 mistakes |
| Precision | When the model says "8", how often is it right? | It predicted "8" 976 times; 953 were real 8s (97.64%) and 23 were false alarms |
| Recall | Of all the real 8s, how many did the model find? | Of 974 real 8s, it found 953 (97.84%) and missed 21 |
| F1 score | One score balancing the two: 2 × P × R / (P + R). It is low if either one is low | Digit 8: 0.9774, the lowest; digit 1: 0.9934, the highest |

How to read them:

- **Low precision means false alarms:** the model labels other digits as this one.
- **Low recall means misses:** the model fails to recognize this digit.
- **Macro vs. weighted:** precision, recall and F1 are computed per digit, then averaged over the 10 digits. **Macro** counts every digit equally, so one weak digit pulls it down. **Weighted** counts each digit by its number of test images. When the two are close, as in 3.6, no digit is being neglected.

### 3.4 Source Code

Source code: <https://github.com/bettyckang/ECS170-project/tree/stage-2>. The Stage 2 code lives in `local_code/stage_2_code/` (Dataset_Loader, Method_MLP, Setting_Train_Test_Given, Evaluate_Metrics, Plot_Learning_Curve) and `script/stage_2_script/` (`script_mlp.py` for the main run, `script_mlp_experiments.py` for the ablations).

### 3.5 Training Convergence Plot

![Training loss falls from 0.383 to 0.015 over 30 epochs](figures/loss_curve.png)

The loss falls 60% between epochs 1 and 2 and flattens after about epoch 20, so gradient descent with Adam converges smoothly. Training and test accuracy per epoch are in [figures/train_test_accuracy.png](figures/train_test_accuracy.png).

### 3.6 Model Performance

The baseline MLP reaches 98.44% accuracy on the 10,000 test images, and every averaged metric is within 0.02 points of it.

| Metric | Macro average | Weighted average |
| --- | --- | --- |
| Accuracy | 0.9844 | 0.9844 |
| Precision | 0.9842 | 0.9844 |
| Recall | 0.9842 | 0.9844 |
| F1 score | 0.9842 | 0.9844 |

Per class, F1 ranges from 0.977 for digit 8 to 0.993 for digit 1. Macro and weighted averages nearly match because the 10 classes are close to balanced (892 to 1,135 test images each).

### 3.7 Ablation Studies

Only the learning rate mattered much: raising Adam's learning rate tenfold to 0.01 cut test accuracy to 97.28%, while every other change stayed within 0.2 points of the 98.25% baseline. Each run changes one setting from the baseline and trains for 20 epochs with the same seed. In every run, weighted precision, recall and F1 and macro F1 are within 0.0002 of accuracy, so the chart shows accuracy alone.

![Test accuracy for 12 one-change-at-a-time runs](figures/ablation_accuracy.png)

- **Learning rate:** at 0.01, Adam's steps overshoot, and accuracy fell about 1 point.
- **Depth:** one hidden layer lost only 0.08 points and trained twice as fast (12.5 s vs 26.9 s). Three layers gained 0.01 points but took 43.1 s.
- **Dropout and activation:** removing dropout cost 0.10 points. ReLU beat tanh and sigmoid by about 0.2 points.
- **Optimizer:** RMSprop (98.35%) and AdamW (98.31%) edged out Adam; SGD with momentum trailed at 98.12%.
- **Loss:** NLL loss after log-softmax is mathematically the same as cross-entropy, and it gave an identical 98.25%. The multi-margin (hinge) loss tied for best at 98.35%.
- **GPU (optional 2-6):** the same baseline on the Apple GPU (MPS) took 46.4 s versus 26.9 s on the CPU, because transfer overhead outweighs the speedup for a model this small.

These runs use a single seed, so gaps under about 0.2 points may be noise. Our final model keeps the baseline settings with 30 epochs, which reached 98.44% (Section 3.6).

## Running the code

The Stage 2 data is not in the repo, because `train.csv` (110 MB) exceeds GitHub's 100 MB file limit. Download it from the Stage 2 "Dataset" link in the project description and unzip it so the files sit at `data/stage_2_data/train.csv` and `data/stage_2_data/test.csv`.

Run the scripts from their own folder with the repo root on the Python path (PyCharm does both when the repo is opened as the project):

```bash
cd script/stage_2_script
PYTHONPATH=../.. python3 script_mlp.py
PYTHONPATH=../.. python3 script_mlp_experiments.py
```

Results and plots are written to `result/stage_2_result/`.
