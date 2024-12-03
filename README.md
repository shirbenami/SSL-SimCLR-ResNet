# SimCLR-Based Training and Fine-Tuning on STL10 Dataset
This repository contains a deep learning project focused on implementing SimCLR (Simple Contrastive Learning of Representations) for self-supervised learning and fine-tuning it on the STL10 dataset. The project includes training, validation, and testing modules, as well as visualization and evaluation tools.

## Project Overview

This project explores self-supervised learning techniques using SimCLR and evaluates its impact on classification performance when fine-tuned on a labeled dataset. The key steps include:

#### 1. Self-Supervised Pretraining (SimCLR):
-Pretrained ResNet50 on the STL10 dataset's unlabeled images.
-Optimized using the InfoNCE loss function to learn useful feature representations.

#### 2. Fine-Tuning:
-Fine-tuned the pretrained ResNet50 on the STL10 dataset's labeled training set.
-Evaluated on the validation and test sets.

#### 3. Supervised Training Baseline:
-Trained ResNet50 from scratch on the STL10 labeled dataset as a baseline for comparison.

#### 4. Evaluation:
-Compared the performance of the SSL-trained model with the supervised baseline.
-Visualized performance metrics such as loss, accuracy, and confusion matrix.

## Dataset - STL10

* The STL10 dataset is designed for developing self-supervised learning techniques. It includes:
* Unlabeled Set: 100,000 images (for SSL training).
* Train Set: 5,000 labeled images across 10 classes (500 per class).
* Test Set: 8,000 labeled images across the same 10 classes.
* The dataset is loaded using PyTorch's torchvision.datasets.STL10 utility.

## Project Structure

```python

project_root/
├── dataset/
│   └── stl10_loader.py          # Responsible for loading and preprocessing the STL10 dataset.
│
├── loss_functions/
│   └── info_nce.py              # Implementation of the InfoNCE loss function used in SimCLR.
│
├── model/
│   └── resnet50.py              # Defines the ResNet50 architecture with optional modifications (e.g., fine-tuning or custom classification head).
│   ├── ssl_model.py             # Defines the ssl architecture
|
├── trainers/
│   ├── train.py                 # Implements the training loop for supervised models.
│   ├── validate.py              # Implements the validation loop for calculating validation loss and accuracy.
│   └── test.py                  # Implements the testing loop for evaluating the model on the test dataset.
│
├── ssl_trainers/
│   ├── train.py                 # Implements the training loop for ssl models.
│   ├── validate.py              # Implements the validation loop for calculating validation loss and accuracy.
│   └── test.py                  # Implements the testing loop for evaluating the model on the test dataset.
├── output/
│   ├── logs/                    # Stores loss and accuracy graphs, as well as confusion matrix images.
│   └── models/                  # Stores the trained model weights (.pth files) for SimCLR and fine-tuning.
│
├── simclr_train.py              # Main script for training SimCLR on the STL10 unlabeled dataset.
├── supervised_train.py          # Main script for supervised training on the STL10 labeled dataset.
├── fine_tuning.py               # Main script for fine-tuning a model using SSL-pretrained weights.
├── README.md                    # Project description, instructions, and results.
└── .gitignore                   # Specifies files and folders to exclude from version control.

```


## Installation
### Prerequisites

Python 3.8+
PyTorch with CUDA support (if using a GPU)

## Usage 
### 1. Self-Supervised Pretraining (SimCLR)

Run the SimCLR training on STL10's unlabeled data:
```python
python trainers/train_ssl.py
```

### 2. Fine-Tuning with SSL Weights
Fine-tune the pretrained model on the STL10 labeled training set:
```python
python fine_tuning.py
```

### 3. Supervised Baseline
Train ResNet50 from scratch as a baseline:
```python
python supervised_train.py
 ```

### 4. Visualizations and Evaluation

View loss and accuracy graphs in output/logs/.
Inspect the confusion matrix in output/logs/confusion_matrix.png.

## Results

## 1. Loss and Accuracy Over Epochs

### Supervised Baseline:
![train_val_graphs](https://github.com/user-attachments/assets/9899d5bc-7a86-4a1d-9702-a739833e97cd)
* Test Accuracy: 67.16%

### Fine-Tuned (SSL):
![fine_tuning_classification_graphs](https://github.com/user-attachments/assets/0fe6c2c5-f6f9-4b28-a919-c27c1fc43b9a)
* Test Accuracy: 83.65%

## 2. Confusion Matrix

### Supervised Baseline:

![confusion_matrix](https://github.com/user-attachments/assets/212e41ba-1576-4e99-9645-36c7ea260215)

### Fine-Tuned (SSL):

![fine_tuning_confusion_matrix](https://github.com/user-attachments/assets/ad9c6432-e57a-4eb4-a1be-edf6e703e9d7)

## Resources
- [SimCLR Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [STL10 Dataset](https://www.kaggle.com/datasets/jessicali9530/stl10?resource=download)
