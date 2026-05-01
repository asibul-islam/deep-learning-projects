import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

val_dir = "../data/cats_dogs/val"

model = keras.models.load_model("../models/cats_dogs_transfer_mobilenet.keras")

class_names = ["cats", "dogs"]

val_data = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

wrong_images = []
wrong_preds = []
wrong_labels = []

for images, labels in val_data:
    preds = model.predict(images)

    pred_labels = (preds > 0.5).astype(int).flatten()
    true_labels = labels.numpy().astype(int).flatten()

    for i in range(len(images)):
        if pred_labels[i] != true_labels[i]:
            wrong_images.append(images[i])
            wrong_preds.append(pred_labels[i])
            wrong_labels.append(true_labels[i])

# Show some wrong predictions
plt.figure(figsize=(10, 6))

for i in range(min(6, len(wrong_images))):
    plt.subplot(2, 3, i + 1)
    plt.imshow(wrong_images[i].numpy().astype("uint8"))

    pred = class_names[wrong_preds[i]]
    actual = class_names[wrong_labels[i]]

    plt.title(f"P: {pred} | A: {actual}")
    plt.axis("off")

plt.tight_layout()
plt.show()