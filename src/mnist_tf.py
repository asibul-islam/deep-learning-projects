import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

index = 1234

# Show one test image
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Label: {y_test[index]}")
plt.axis('off')
plt.show()

# Build model
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)

# Predict
predictions = model.predict(x_test)

print("\nPredicted:", predictions[index].argmax())
print("Actual:", y_test[index])