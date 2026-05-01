# Deep Learning Projects (PyTorch & TensorFlow)

This repository contains implementations of deep learning models using TensorFlow and PyTorch, with a focus on image processing and classification tasks.

## Projects

### MNIST Digit Classification
Implemented handwritten digit recognition in both TensorFlow and PyTorch.

**TensorFlow version**
- Data loading and normalization
- Feedforward neural network using Keras
- Model training, evaluation, and prediction
- Achieved ~97% test accuracy

**PyTorch version**
- Data loading with torchvision and DataLoader
- Feedforward neural network using `torch.nn`
- Manual training and evaluation loop
- Achieved 97.56% test accuracy

### Image Processing Pipeline
- Image loading using PIL
- Conversion to NumPy arrays
- Resizing and normalization
- Tensor conversion for TensorFlow and PyTorch

## Tech Stack
- Python
- TensorFlow
- PyTorch
- NumPy
- Matplotlib
- torchvision

## Model Evaluation

- Accuracy: 98%
- Precision/Recall: ~98–99%
- Confusion Matrix included

## Failure Analysis

The model struggles with:
- low-quality images
- occluded animals
- unusual poses


## Structure

```text
deep-learning-projects/
├── src/
├── data/
├── requirements.txt
├── README.md
└── .gitignore