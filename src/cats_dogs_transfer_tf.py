import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

train_dir = "../data/cats_dogs/train"
val_dir = "../data/cats_dogs/val"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_data = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_data = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = keras.Sequential([
    keras.Input(shape=(160, 160, 3)),

    data_augmentation,

    keras.layers.Rescaling(1./127.5, offset=-1),

    base_model,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

val_loss, val_acc = model.evaluate(val_data)
print("\nValidation accuracy:", val_acc)

model.save("../models/cats_dogs_transfer_mobilenet.keras")

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Transfer Learning Accuracy")
plt.show()