# SimCLR-Based Training and Fine-Tuning on STL10 Dataset
This repository contains a deep learning project focused on implementing SimCLR (Simple Contrastive Learning of Representations) for self-supervised learning and fine-tuning it on the STL10 dataset. The project includes training, validation, and testing modules, as well as visualization and evaluation tools.

## Project Overview
![image4](https://github.com/user-attachments/assets/b552f562-9cd8-4543-93b0-986dc4dd0cb2)

This project explores self-supervised learning techniques using SimCLR and evaluates its impact on classification performance when fine-tuned on a labeled dataset. The key steps include:

1. **Supervised Baseline Training:**
   - Trained ResNet50 from scratch on the STL10 labeled dataset to establish a baseline for performance comparison.

2. **Self-Supervised Pretraining with SimCLR:**
   - Pretrained ResNet50 on the STL10 dataset using unlabeled images.
   - Applied the SimCLR method, which involves contrastive learning to enhance feature representation.
   - Positive pairs: Created by applying data augmentations (e.g., cropping, color jittering) to the same image to generate different views.
   - Negative pairs: Consist of augmented views from different images.
   - The model was trained to minimize the distance between positive pairs while maximizing the distance from negative pairs, optimizing using the InfoNCE loss function.

3. **Fine-Tuning:**
   - Fine-tuned the SSL-pretrained ResNet50 on the labeled portion of the STL10 dataset.

#### 4. Evaluation:
* Compared the performance of the SSL-trained model with the supervised baseline.
* Visualized performance metrics such as loss, accuracy, and confusion matrix.

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
![train_val_graphs_supervised_model_60epochs](https://github.com/user-attachments/assets/23b23c3b-d5ac-4422-93cd-3e1ee5c1dd56)

* Test Accuracy: 73.55%

### Fine-Tuned (SSL):
![fine_tuning_classification_graphs_60epochs](https://github.com/user-attachments/assets/f83fb656-e983-41b0-90e0-803fdc654cb0)

* Test Accuracy: 82.49%

## 2. Confusion Matrix

### Supervised Baseline:

![confusion_matrix_60epochs](https://github.com/user-attachments/assets/3a3dd871-be6d-44af-b031-4164b96c6549)

### Fine-Tuned (SSL):

![confusion_matrix_fine_tuning_60epochs](https://github.com/user-attachments/assets/8c530b73-d825-4af8-9b28-dc97de92fd2e)


## Resources
- [SimCLR Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [STL10 Dataset](https://www.kaggle.com/datasets/jessicali9530/stl10?resource=download)
- [SimCLR information](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr)
