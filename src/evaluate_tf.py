import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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

y_true = []
y_pred = []

for images, labels in val_data:
    preds = model.predict(images)

    predicted_labels = (preds > 0.5).astype(int).flatten()

    y_pred.extend(predicted_labels)
    y_true.extend(labels.numpy().astype(int).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix - TensorFlow MobileNetV2")
plt.show()