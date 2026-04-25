import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Original train shape:", x_train.shape)
print("Original test shape:", x_test.shape)

# Normalize pixel
x_train = x_train / 255.0
x_test = x_test / 255.0

# channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("New train shape:", x_train.shape)
print("New test shape:", x_test.shape)

# Show one image
index = 2245
plt.imshow(x_test[index].squeeze(), cmap="gray")
plt.title(f"Label: {y_test[index]}")
plt.axis("off")
plt.show()

# Build CNN model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model structure
model.summary()

# Train model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)

# Predict
predictions = model.predict(x_test)

print("\nPredicted:", predictions[index].argmax())
print("Actual:", y_test[index])