from tensorflow import keras
import matplotlib.pyplot as plt

val_dir = "../data/cats_dogs/val"

model = keras.models.load_model("../models/cats_dogs_transfer_mobilenet.keras")

class_names = ["cats", "dogs"]

val_data = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary",
    shuffle=True
)

for images, labels in val_data.take(1):
    preds = model.predict(images)

    plt.figure(figsize=(10, 6))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        pred_label = class_names[int(preds[i][0] > 0.5)]
        true_label = class_names[int(labels[i][0])]

        plt.title(f"P: {pred_label} | A: {true_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()