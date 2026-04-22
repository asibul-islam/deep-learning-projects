from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = Image.open("../sample.avif")

# Resize image (VERY IMPORTANT)
img = img.resize((128, 128))

# Convert to numpy array
img_array = np.array(img)

print("Before normalization:")
print("Shape:", img_array.shape)
print("Max pixel:", img_array.max())
print("Min pixel:", img_array.min())

# Normalize (0–255 → 0–1)
img_array = img_array / 255.0

print("\nAfter normalization:")
print("Max pixel:", img_array.max())
print("Min pixel:", img_array.min())

# Show image
plt.imshow(img_array)
plt.title("Normalized Image")
plt.axis('off')
plt.show()

import torch
import tensorflow as tf

# Convert to PyTorch tensor
torch_tensor = torch.tensor(img_array).permute(2, 0, 1)
print("\nPyTorch tensor shape:", torch_tensor.shape)

# Convert to TensorFlow tensor
tf_tensor = tf.convert_to_tensor(img_array)
print("TensorFlow tensor shape:", tf_tensor.shape)