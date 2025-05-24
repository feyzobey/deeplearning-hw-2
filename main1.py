import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from tensorflow.keras import Input, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# visualize first 10 digits
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(str(y_train[i]))
    plt.axis("off")
plt.show()


def augment_dataset(images, labels):
    new_images = []
    new_labels = []
    rotation_labels = []
    for i in range(len(images)):
        digit = labels[i]
        image = images[i]
        if digit in [0, 6, 8, 9]:
            new_images.append(image)
            new_labels.append(digit)
            rotation_labels.append(0)
        else:
            for degree in [0, 90, 180, 270]:
                new_images.append(rotate(image, degree, reshape=False))
                new_labels.append(digit)
                rotation_labels.append(degree)
    return np.array(new_images), np.array(new_labels), np.array(rotation_labels)


# augment dataset
X_train, Y_train, Y_rotation = augment_dataset(x_train, y_train)
X_test, Y_test, Y_rotation_test = augment_dataset(x_test, y_test)


# label conversion for rotations, not sure if it's best way but works for us
def rot_to_cls(deg):
    if deg == 0:
        return 0
    if deg == 90:
        return 1
    if deg == 180:
        return 2
    return 3


Y_rot_cls = np.array([rot_to_cls(r) for r in Y_rotation])
Y_rot_cls_test = np.array([rot_to_cls(r) for r in Y_rotation_test])

# normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)

# model architecture
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))

inputs = Input(shape=(28, 28, 1))
m = Conv2D(13, (4, 4), activation="relu")(inputs)
m = MaxPooling2D(pool_size=(2, 2))(m)
m = Conv2D(29, (3, 3), activation="relu")(m)
m = MaxPooling2D(pool_size=(2, 2))(m)
m = Dropout(0.21)(m)
m = Flatten()(m)
m = Dense(39, activation="relu")(m)
m = Dense(16, activation="relu")(m)

out_digit = Dense(10, activation="softmax", name="digit_pred")(m)
out_rot = Dense(4, activation="softmax", name="rot_pred")(m)

model = Model(inputs=inputs, outputs=[out_digit, out_rot])
model.compile(optimizer="adam", loss={"digit_pred": "sparse_categorical_crossentropy", "rot_pred": "sparse_categorical_crossentropy"}, metrics={"digit_pred": "accuracy", "rot_pred": "accuracy"})

# train model
history = model.fit(X_train, {"digit_pred": Y_train, "rot_pred": Y_rot_cls}, epochs=5, batch_size=128, validation_data=(X_test, {"digit_pred": Y_test, "rot_pred": Y_rot_cls_test}))

# visualize results
plt.figure(figsize=(10, 5))
plt.plot(history.history["digit_pred_accuracy"], label="Digit accuracy")
plt.plot(history.history["val_digit_pred_accuracy"], label="Validation digit accuracy")
plt.plot(history.history["rot_pred_accuracy"], label="Rotation accuracy")
plt.plot(history.history["val_rot_pred_accuracy"], label="Validation rotation accuracy")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# evaluate on test set
test_values = model.evaluate(X_test, {"digit_pred": Y_test, "rot_pred": Y_rot_cls_test}, batch_size=128)
print("Test:", test_values)

# visualize predictions
predictions = model.predict(X_test[:8])
digits = predictions[0]
rotations = predictions[1]

plt.figure(figsize=(10, 2))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"{Y_test[i]}/{Y_rotation_test[i]}")
    plt.xlabel(f"T:{np.argmax(digits[i])}\nD:{np.argmax(rotations[i])*90}")
    plt.axis("off")
plt.show()
