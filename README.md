# ECS 170 Fall 2026: Stage 2 Report

## Team Information

**Team name:** [Enter team name]

| Student | Name | Student ID | Email |
| --- | --- | --- | --- |
| 1 | Betty Kang | 922129603 | bckang@ucdavis.edu |
| 2 | Sruthi Sridhar | 923887258 | srusridhar@ucdavis.edu |
| 3 | Ashwin Charles | 922006548 | ajcharles@ucdavis.edu |

## Section 1: Task Description

We trained a PyTorch multi-layer perceptron (MLP) to classify 28×28 handwritten digit images (784 pixels) into 10 classes, reaching 98.44% test accuracy. Using the course code template, we built the data loader, model and metrics, then tested how six settings affect accuracy. We compare results in percentage **points**, the plain difference between two percentages: 98.25% vs 98.05% is a gap of 0.2 points.

## Section 2: Model Description

Our model (`Method_MLP`, built on the template's `method` class) is a fully connected network with two hidden layers.

![Baseline MLP architecture: 3 linear layers, 235,146 parameters](figures/architecture.png)

- **Linear layers** compute a weighted sum of their inputs (w·x + b); training learns these weights.
- **Two hidden layers (256, 128)** build complex patterns from simple ones; a single layer can only draw straight-line boundaries.
- **ReLU, an on/off switch:** each hidden unit keeps its value if it is positive and outputs 0 if it is negative. This simple bend lets stacked layers learn curved shapes like a 3 or an 8; without it, they would act like one straight-line layer.
- **Dropout (20%)** switches off random hidden units during training, so the model cannot memorize training images.
- **Output (10 scores):** in training, cross-entropy applies softmax and penalizes wrong digits; at test time, the highest score is the prediction.

## Section 3: Experiment Settings

### 3.1 Dataset Description

The instructor's dataset is pre-split into 60,000 training images (86%) and 10,000 test images (14%). Each line of `train.csv` and `test.csv` is a label (0–9) followed by 784 pixel values (0–255). We kept this fixed split, so every team is scored on the same test images, and used no cross validation. Why this split:

- **A separate test set:** accuracy only means something on unseen images. Ours scores 99.48% on training images but 98.44% on test images.
- **Most images for training:** handwriting varies widely between writers, and the model learns that variety only from examples.
- **10,000 test images is enough:** luck in which images land in the test set moves our accuracy by only about 0.12 points, and each digit makes up 9–11% of both sets.

### 3.2 Detailed Experimental Setups

| Setting | Value | Why it matters |
| --- | --- | --- |
| Input scaling | Pixels ÷ 255 | Keeps inputs in [0, 1] for stable training |
| Layers | 784 → 256 → 128 → 10 | Inputs match the pixels; outputs match the 10 digits |
| Activation | ReLU | Learns curved shapes; beat tanh and sigmoid by 0.2 points |
| Dropout | 0.2, training only | Limits memorizing; removing it cost 0.10 points |
| Loss | Cross-entropy | Applies softmax itself, so we dropped the template's softmax layer |
| Optimizer | Adam, learning rate 0.001 | Adapts each weight's step; a 10× higher rate lost 1 point |
| Batch size / epochs | 128 / 30, reshuffled | 469 updates per epoch, not 1; loss flattens by epoch 20 |
| Initialization | PyTorch default (Kaiming) | Scaled random start, so signals neither blow up nor vanish |
| Random seed | 2 | Repeatable runs, so 3.7 compares settings fairly |

### 3.3 Evaluation Metrics

Each metric answers a different question. Examples use digit 8, our weakest digit.

| Metric | Question it answers | Example from our model |
| --- | --- | --- |
| Accuracy | How many test images were classified correctly? | 9,844 of 10,000 (98.44%) |
| Precision | When it predicts "8", how often is it right? | 953 of 976 "8" predictions (97.64%) |
| Recall | Of the real 8s, how many did it find? | 953 of 974 real 8s (97.84%) |
| F1 score | Balances the two: 2PR / (P + R) | 0.9774 for digit 8, the lowest of the 10 |

Low precision means false alarms; low recall means misses. Per-digit scores are averaged two ways: **macro** treats every digit equally, so a weak digit shows up, and **weighted** counts each digit by its number of test images.

### 3.4 Source Code

[github.com/bettyckang/ECS170-project (stage-2 branch)](https://github.com/bettyckang/ECS170-project/tree/stage-2): Stage 2 code is in `local_code/stage_2_code/` and `script/stage_2_script/`.

### 3.5 Training Convergence Plot

![Training loss falls from 0.383 to 0.015 over 30 epochs](figures/loss_curve.png)

Loss falls 60% between epochs 1 and 2 and flattens after about epoch 20, so training converges smoothly.

### 3.6 Model Performance

The final model reaches **98.44% accuracy** on the 10,000 test images, above the required 95%.

| Metric | Macro average | Weighted average |
| --- | --- | --- |
| Precision | 0.9842 | 0.9844 |
| Recall | 0.9842 | 0.9844 |
| F1 score | 0.9842 | 0.9844 |

Per-digit F1 ranges from 0.977 (digit 8) to 0.993 (digit 1).

### 3.7 Ablation Studies

Each run changes one setting from the baseline (20 epochs, same seed). Only the learning rate mattered much: at 0.01, accuracy fell to 97.28%, while every other change stayed within 0.2 points of the 98.25% baseline. Precision, recall and F1 track accuracy within 0.0002, so the chart shows accuracy alone.

![Test accuracy for 12 one-change-at-a-time runs](figures/ablation_accuracy.png)

- **Depth:** one hidden layer lost 0.08 points but trained twice as fast; three layers gained only 0.01.
- **Dropout and activation:** removing dropout cost 0.10 points; tanh and sigmoid trailed ReLU by about 0.2.
- **Optimizer and loss:** RMSprop and the hinge loss tied for best (98.35%). NLL matched cross-entropy exactly, as expected, since they compute the same loss.
With one seed, gaps under about 0.2 points may be noise. Our final model keeps the baseline settings, trained for 30 epochs.

## Running the code

The Stage 2 data is not in the repo, because `train.csv` (110 MB) exceeds GitHub's 100 MB file limit. Download it from the Stage 2 "Dataset" link in the project description and unzip it so the files sit at `data/stage_2_data/train.csv` and `data/stage_2_data/test.csv`.

Run the scripts from their own folder with the repo root on the Python path (PyCharm does both when the repo is opened as the project):

```bash
cd script/stage_2_script
PYTHONPATH=../.. python3 script_mlp.py
PYTHONPATH=../.. python3 script_mlp_experiments.py
```

Results and plots are written to `result/stage_2_result/`. Per-epoch training and test accuracy for the baseline are also in [figures/train_test_accuracy.png](figures/train_test_accuracy.png).
