import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

tf.keras.utils.disable_interactive_logging()

print("--- EXPERIMENT 6: FEEDFORWARD AND CONVOLUTIONAL NEURAL NETWORKS ---")

print("\n--- Part 1: Feedforward Neural Network ---")

(x_train_fnn, y_train_fnn), (x_test_fnn, y_test_fnn) = fashion_mnist.load_data()

print("Original FNN training data shape:", x_train_fnn.shape)
print("Original FNN test data shape:", x_test_fnn.shape)

x_train_fnn_flat = x_train_fnn.reshape(-1, 28 * 28)
x_test_fnn_flat = x_test_fnn.reshape(-1, 28 * 28)

x_train_fnn_norm = x_train_fnn_flat / 255.0
x_test_fnn_norm = x_test_fnn_flat / 255.0

print("Flattened FNN training data shape:", x_train_fnn_norm.shape)
print("Flattened FNN test data shape:", x_test_fnn_norm.shape)

model_fnn = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model_fnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- FNN Model Summary ---")
model_fnn.summary()

print("\n--- Training FNN Model ---")

history_fnn = model_fnn.fit(
    x_train_fnn_norm,
    y_train_fnn,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

print("\n--- Evaluating FNN Model ---")

loss_fnn, accuracy_fnn = model_fnn.evaluate(
    x_test_fnn_norm,
    y_test_fnn,
    verbose=0
)

print("FNN Test Loss:", round(loss_fnn, 4))
print("FNN Test Accuracy:", round(accuracy_fnn, 4))

y_pred_fnn = np.argmax(
    model_fnn.predict(x_test_fnn_norm, verbose=0),
    axis=-1
)

print("\n--- FNN Classification Report ---")
print(classification_report(y_test_fnn, y_pred_fnn))

cm_fnn = confusion_matrix(y_test_fnn, y_pred_fnn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_fnn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("FNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_fnn.history["accuracy"], label="Training Accuracy")
plt.plot(history_fnn.history["val_accuracy"], label="Validation Accuracy")
plt.title("FNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_fnn.history["loss"], label="Training Loss")
plt.plot(history_fnn.history["val_loss"], label="Validation Loss")
plt.title("FNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n--- Part 2: Convolutional Neural Network ---")

(x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn) = mnist.load_data()

print("Original CNN training data shape:", x_train_cnn.shape)
print("Original CNN test data shape:", x_test_cnn.shape)

x_train_cnn = x_train_cnn.reshape(
    x_train_cnn.shape[0], 28, 28, 1
)

x_test_cnn = x_test_cnn.reshape(
    x_test_cnn.shape[0], 28, 28, 1
)

x_train_cnn = x_train_cnn.astype("float32") / 255.0
x_test_cnn = x_test_cnn.astype("float32") / 255.0

print("Reshaped CNN training data shape:", x_train_cnn.shape)
print("Reshaped CNN test data shape:", x_test_cnn.shape)

model_cnn = keras.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])

model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- CNN Model Summary ---")
model_cnn.summary()

print("\n--- Training CNN Model ---")

history_cnn = model_cnn.fit(
    x_train_cnn,
    y_train_cnn,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

print("\n--- Evaluating CNN Model ---")

loss_cnn, accuracy_cnn = model_cnn.evaluate(
    x_test_cnn,
    y_test_cnn,
    verbose=0
)

print("CNN Test Loss:", round(loss_cnn, 4))
print("CNN Test Accuracy:", round(accuracy_cnn, 4))

y_pred_cnn = np.argmax(
    model_cnn.predict(x_test_cnn, verbose=0),
    axis=-1
)

print("\n--- CNN Classification Report ---")
print(classification_report(y_test_cnn, y_pred_cnn))

cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_cnn.history["accuracy"], label="Training Accuracy")
plt.plot(history_cnn.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history["loss"], label="Training Loss")
plt.plot(history_cnn.history["val_loss"], label="Validation Loss")
plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n--- Sample CNN Predictions ---")

class_names = [str(i) for i in range(10)]

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(
        x_test_cnn[i].reshape(28, 28),
        cmap=plt.cm.binary
    )

    true_label = y_test_cnn[i]
    predicted_label = y_pred_cnn[i]

    label_color = (
        "green"
        if true_label == predicted_label
        else "red"
    )

    plt.xlabel(
        f"True: {class_names[true_label]}\n"
        f"Pred: {class_names[predicted_label]}",
        color=label_color
    )

plt.suptitle(
    "Sample CNN Predictions",
    y=1.02,
    fontsize=16
)

plt.tight_layout()
plt.show()

print("\n--- RESULTS ---")
print("FNN Test Accuracy:", round(accuracy_fnn, 4))
print("CNN Test Accuracy:", round(accuracy_cnn, 4))
