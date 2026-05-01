import sys
import numpy as np
import tensorflow as tf
import torch

print("=" * 40)
print("Environment Check")
print("=" * 40)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(torch.backends.mps.is_available())