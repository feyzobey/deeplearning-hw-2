import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate
from tensorflow.keras import Input, layers, models
from tensorflow.keras.datasets import mnist

# download mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Eğitim seti boyutu:", x_train.shape)
print("Test seti boyutu:", x_test.shape)

# visualize first 10 digits
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")
plt.show()


def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)


def augment_dataset(images, labels):
    aug_images = []
    aug_digits = []
    aug_rotations = []

    no_rotate = [0, 6, 8, 9]
    rotation_angles = [0, 90, 180, 270]

    for img, label in zip(images, labels):
        if label in no_rotate:
            aug_images.append(img)
            aug_digits.append(label)
            aug_rotations.append(0)
        else:
            for angle in rotation_angles:
                aug_images.append(rotate_image(img, angle))
                aug_digits.append(label)
                aug_rotations.append(angle)
    return np.array(aug_images), np.array(aug_digits), np.array(aug_rotations)


# augment dataset (train & test)
x_train_aug, y_train_digit, y_train_rotation = augment_dataset(x_train, y_train)
x_test_aug, y_test_digit, y_test_rotation = augment_dataset(x_test, y_test)

print("Augmente eğitim seti:", x_train_aug.shape)
print("Augmente test seti:", x_test_aug.shape)


# rotation label to categorical index
def rotation_to_index(rotation):
    return {0: 0, 90: 1, 180: 2, 270: 3}[rotation]


y_train_rotation_idx = np.array([rotation_to_index(r) for r in y_train_rotation])
y_test_rotation_idx = np.array([rotation_to_index(r) for r in y_test_rotation])

# normalize inputs
x_train_aug = x_train_aug.astype("float32") / 255.0
x_test_aug = x_test_aug.astype("float32") / 255.0

x_train_aug = np.expand_dims(x_train_aug, -1)
x_test_aug = np.expand_dims(x_test_aug, -1)

print("x_train_aug shape:", x_train_aug.shape)
print("y_train_digit shape:", y_train_digit.shape)
print("y_train_rotation_idx shape:", y_train_rotation_idx.shape)

# visualize some examples
idxs = np.random.choice(len(x_train_aug), 8, replace=False)
plt.figure(figsize=(12, 2))
for i, idx in enumerate(idxs):
    plt.subplot(1, 8, i + 1)
    plt.imshow(x_train_aug[idx].reshape(28, 28), cmap="gray")
    plt.title(f"{y_train_digit[idx]}/{y_train_rotation[idx]}")
    plt.axis("off")
plt.show()

# model architecture
inputs = Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)

out_digit = layers.Dense(10, activation="softmax", name="digit")(x)
out_rotation = layers.Dense(4, activation="softmax", name="rotation")(x)

model = models.Model(inputs=inputs, outputs=[out_digit, out_rotation])
model.compile(optimizer="adam", loss={"digit": "sparse_categorical_crossentropy", "rotation": "sparse_categorical_crossentropy"}, metrics={"digit": "accuracy", "rotation": "accuracy"})
model.summary()

# model training
history = model.fit(
    x_train_aug, {"digit": y_train_digit, "rotation": y_train_rotation_idx}, validation_data=(x_test_aug, {"digit": y_test_digit, "rotation": y_test_rotation_idx}), epochs=5, batch_size=128
)

# visualize results
plt.figure(figsize=(8, 5))
plt.plot(history.history["digit_accuracy"], label="Digit Accuracy")
plt.plot(history.history["val_digit_accuracy"], label="Val Digit Accuracy")
plt.plot(history.history["rotation_accuracy"], label="Rotation Accuracy")
plt.plot(history.history["val_rotation_accuracy"], label="Val Rotation Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# evaluate on test set
results = model.evaluate(x_test_aug, {"digit": y_test_digit, "rotation": y_test_rotation_idx}, batch_size=128)
print(dict(zip(model.metrics_names, results)))

pred_digits, pred_rotations = model.predict(x_test_aug[:8])
plt.figure(figsize=(12, 2))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(x_test_aug[i].reshape(28, 28), cmap="gray")
    plt.title(f"Actual: {y_test_digit[i]}\nRotation: {y_test_rotation[i]}")
    plt.xlabel(f"Predicted: {np.argmax(pred_digits[i])}\nRotation: {np.argmax(pred_rotations[i])*90}")
    plt.axis("off")
plt.show()
