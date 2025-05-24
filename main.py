"""
MNIST Digit Classification and Rotation Detection - Simplified Version
======================================================================

This implementation replicates main.py functionality without TensorFlow and
without visualization to avoid matplotlib compatibility issues.

Features:
1. Real MNIST dataset loading
2. Data augmentation with rotations
3. Neural network layers from scratch
4. Training with mini-batch gradient descent
5. Prediction and evaluation
6. Multi-task learning

Author: Student Implementation (No TensorFlow, No Matplotlib)
"""

import numpy as np
from scipy.ndimage import rotate
import gzip
import struct
import urllib.request
import os

# =============================================================================
# STEP 1: MNIST Dataset Loading
# =============================================================================


def download_mnist_files():
    """Download MNIST dataset files manually"""
    print("Downloading MNIST dataset...")

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    if not os.path.exists("data"):
        os.makedirs("data")

    for file in files:
        filepath = f"data/{file}"
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, filepath)
            except Exception as e:
                print(f"Failed to download {file}: {e}")
                return False
    return True


def load_mnist_images(filename):
    """Load MNIST images from idx3-ubyte format"""
    with gzip.open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}")

        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    return images


def load_mnist_labels(filename):
    """Load MNIST labels from idx1-ubyte format"""
    with gzip.open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}")

        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)

    return labels


def load_mnist_data():
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")

    files_needed = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    if not all(os.path.exists(f"data/{f}") for f in files_needed):
        if not download_mnist_files():
            print("Creating synthetic dataset...")
            return create_synthetic_mnist()

    try:
        x_train = load_mnist_images("data/train-images-idx3-ubyte.gz")
        y_train = load_mnist_labels("data/train-labels-idx1-ubyte.gz")
        x_test = load_mnist_images("data/t10k-images-idx3-ubyte.gz")
        y_test = load_mnist_labels("data/t10k-labels-idx1-ubyte.gz")

        print(f"Loaded MNIST: Train {x_train.shape}, Test {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return create_synthetic_mnist()


def create_synthetic_mnist():
    """Create synthetic MNIST-like dataset"""
    print("Creating synthetic MNIST dataset...")
    np.random.seed(42)

    def create_digit_pattern(digit, size=28):
        pattern = np.zeros((size, size))
        center = size // 2

        if digit == 0:  # Circle
            y, x = np.ogrid[:size, :size]
            mask = (x - center) ** 2 + (y - center) ** 2 < (size // 3) ** 2
            pattern[mask] = 1
            inner_mask = (x - center) ** 2 + (y - center) ** 2 < (size // 4) ** 2
            pattern[inner_mask] = 0
        elif digit == 1:  # Vertical line
            pattern[:, center - 1 : center + 2] = 1
        elif digit == 2:  # Horizontal segments
            pattern[size // 4 : size // 4 + 3, :] = 1
            pattern[center - 1 : center + 2, : size // 2] = 1
            pattern[3 * size // 4 : 3 * size // 4 + 3, :] = 1
        else:
            # Random patterns for other digits
            pattern[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = np.random.random((size // 2, size // 2)) > 0.5

        return pattern

    # Generate data (smaller for demo)
    x_train, y_train = [], []
    for digit in range(10):
        for _ in range(200):  # 200 samples per digit
            pattern = create_digit_pattern(digit)
            noise = np.random.normal(0, 0.1, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)
            pattern = (pattern * 255).astype(np.uint8)
            x_train.append(pattern)
            y_train.append(digit)

    x_test, y_test = [], []
    for digit in range(10):
        for _ in range(40):  # 40 samples per digit
            pattern = create_digit_pattern(digit)
            noise = np.random.normal(0, 0.1, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)
            pattern = (pattern * 255).astype(np.uint8)
            x_test.append(pattern)
            y_test.append(digit)

    # Shuffle
    train_indices = np.random.permutation(len(x_train))
    test_indices = np.random.permutation(len(x_test))

    x_train = np.array(x_train)[train_indices]
    y_train = np.array(y_train)[train_indices]
    x_test = np.array(x_test)[test_indices]
    y_test = np.array(y_test)[test_indices]

    print("Note: Using synthetic MNIST data")
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# STEP 2: Data Augmentation
# =============================================================================


def augment_dataset(images, labels):
    """Augment dataset with rotations (same logic as main.py)"""
    print("Augmenting dataset with rotations...")

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
                rotated = rotate(image, degree, reshape=False)
                new_images.append(rotated)
                new_labels.append(digit)
                rotation_labels.append(degree)

    print(f"Dataset: {len(images)} -> {len(new_images)} (with rotations)")
    return np.array(new_images), np.array(new_labels), np.array(rotation_labels)


def rot_to_cls(deg):
    """Convert rotation degrees to class labels"""
    return {0: 0, 90: 1, 180: 2, 270: 3}[deg]


# =============================================================================
# STEP 3: Neural Network Implementation (FIXED VERSION)
# =============================================================================


class Layer:
    """Base layer class"""

    def forward(self, X):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    """Dense layer implementation - FIXED VERSION"""

    def __init__(self, units, activation=None, name=None):
        self.units = units
        self.activation = activation
        self.name = name
        self.weights = None
        self.bias = None

    def build(self, input_shape):
        # Xavier/Glorot initialization - FIXED
        self.weights = np.random.randn(input_shape, self.units) * np.sqrt(2.0 / input_shape)
        self.bias = np.zeros((1, self.units))

    def forward(self, X):
        if self.weights is None:
            self.build(X.shape[1])

        self.last_input = X
        self.z = np.dot(X, self.weights) + self.bias

        if self.activation == "relu":
            self.output = np.maximum(0, self.z)
        elif self.activation == "softmax":
            # Improved numerical stability
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            self.output = self.z

        return self.output

    def backward(self, grad_output, learning_rate=0.001):
        # FIXED: Removed double batch normalization
        if self.activation == "relu":
            # Correct ReLU gradient
            grad_output = grad_output * (self.z > 0)

        # No division by batch_size here since it's done in _compute_gradient
        grad_weights = np.dot(self.last_input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        # Update weights
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class MultiTaskNN:
    """Multi-task neural network for digit classification and rotation detection - FIXED VERSION"""

    def __init__(self):
        print("Building neural network...")

        # Improved architecture - FIXED
        self.dense1 = Dense(256, activation="relu")  # Increased capacity
        self.dense2 = Dense(128, activation="relu")
        self.dense3 = Dense(64, activation="relu")

        # Output heads
        self.digit_output = Dense(10, activation="softmax", name="digit_pred")
        self.rotation_output = Dense(4, activation="softmax", name="rot_pred")

        self.layers = [self.dense1, self.dense2, self.dense3]

        print("Network: 784 -> 256 -> 128 -> 64 -> [10 digits + 4 rotations]")

    def forward(self, X):
        """Forward pass"""
        # Flatten and normalize
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        X = X.astype("float32") / 255.0

        # Forward through shared layers
        current = X
        for layer in self.layers:
            current = layer.forward(current)

        # Output heads
        digit_pred = self.digit_output.forward(current)
        rotation_pred = self.rotation_output.forward(current)

        return digit_pred, rotation_pred

    def backward(self, X, y_digit, y_rotation, learning_rate=0.01):  # FIXED: Increased learning rate
        """Backward pass - FIXED VERSION"""
        # Forward pass
        digit_pred, rotation_pred = self.forward(X)

        # Compute gradients - FIXED
        digit_grad = self._compute_gradient(digit_pred, y_digit, 10)
        rotation_grad = self._compute_gradient(rotation_pred, y_rotation, 4)

        # Backward through output layers
        grad_digit = self.digit_output.backward(digit_grad, learning_rate)
        grad_rotation = self.rotation_output.backward(rotation_grad, learning_rate)

        # Combined gradient for shared layers
        combined_grad = grad_digit + grad_rotation

        # Backward through shared layers
        current_grad = combined_grad
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad, learning_rate)

        return digit_pred, rotation_pred

    def _compute_gradient(self, pred, true, num_classes):
        """Compute cross-entropy gradient - FIXED VERSION"""
        batch_size = len(pred)

        # One-hot encode true labels
        true_onehot = np.zeros((batch_size, num_classes))
        true_onehot[np.arange(batch_size), true] = 1

        # Gradient for softmax + cross-entropy (normalized by batch size)
        return (pred - true_onehot) / batch_size

    def predict(self, X):
        """Make predictions"""
        digit_pred, rotation_pred = self.forward(X)
        return digit_pred, rotation_pred

    def evaluate(self, X, y_digit, y_rotation):
        """Evaluate accuracy"""
        digit_pred, rotation_pred = self.predict(X)

        digit_acc = np.mean(np.argmax(digit_pred, axis=1) == y_digit)
        rotation_acc = np.mean(np.argmax(rotation_pred, axis=1) == y_rotation)

        return digit_acc, rotation_acc


# =============================================================================
# STEP 4: Training Function (FIXED VERSION)
# =============================================================================


def train_model(model, X_train, y_digit_train, y_rotation_train, X_test, y_digit_test, y_rotation_test, epochs=10, batch_size=64, learning_rate=0.01):  # FIXED: Better hyperparameters
    """Train the model - FIXED VERSION"""
    print(f"Training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("-" * 60)

    history = {"train_digit_acc": [], "train_rotation_acc": [], "test_digit_acc": [], "test_rotation_acc": []}

    num_samples = len(X_train)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Shuffle data
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_digit_shuffled = y_digit_train[indices]
        y_rotation_shuffled = y_rotation_train[indices]

        # Mini-batch training
        num_batches = num_samples // batch_size

        # Training loop with progress
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_shuffled[start_idx:end_idx]
            y_digit_batch = y_digit_shuffled[start_idx:end_idx]
            y_rotation_batch = y_rotation_shuffled[start_idx:end_idx]

            # Forward and backward pass
            model.backward(X_batch, y_digit_batch, y_rotation_batch, learning_rate)

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches}")

        # Evaluate on training set (subset for speed)
        train_subset = min(2000, len(X_train))
        train_indices = np.random.choice(len(X_train), train_subset, replace=False)
        train_digit_acc, train_rotation_acc = model.evaluate(X_train[train_indices], y_digit_train[train_indices], y_rotation_train[train_indices])

        # Evaluate on test set
        test_digit_acc, test_rotation_acc = model.evaluate(X_test, y_digit_test, y_rotation_test)

        # Store history
        history["train_digit_acc"].append(train_digit_acc)
        history["train_rotation_acc"].append(train_rotation_acc)
        history["test_digit_acc"].append(test_digit_acc)
        history["test_rotation_acc"].append(test_rotation_acc)

        print(f"Train - Digit: {train_digit_acc:.4f}, Rotation: {train_rotation_acc:.4f}")
        print(f"Test  - Digit: {test_digit_acc:.4f}, Rotation: {test_rotation_acc:.4f}")
        print()

    return history


# =============================================================================
# STEP 5: Main Function
# =============================================================================


def main():
    """Main function (replicates main.py structure)"""
    print("=" * 70)
    print("MNIST Digit Classification + Rotation Detection")
    print("(No TensorFlow Implementation)")
    print("=" * 70)

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Show sample info (without visualization)
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train[0].shape}")
    print(f"Unique digits: {sorted(set(y_train))}")

    # Show first 10 digit labels
    print(f"First 10 training labels: {y_train[:10]}")

    # Augment dataset
    X_train, Y_train, Y_rotation = augment_dataset(x_train, y_train)
    X_test, Y_test, Y_rotation_test = augment_dataset(x_test, y_test)

    # Convert rotation labels
    Y_rot_cls = np.array([rot_to_cls(r) for r in Y_rotation])
    Y_rot_cls_test = np.array([rot_to_cls(r) for r in Y_rotation_test])

    print(f"\nAugmented shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    # Build model
    model = MultiTaskNN()

    # Train model with fixed hyperparameters
    history = train_model(model, X_train, Y_train, Y_rot_cls, X_test, Y_test, Y_rot_cls_test, epochs=10, batch_size=64, learning_rate=0.01)

    # Final evaluation
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_digit_acc, final_rotation_acc = model.evaluate(X_test, Y_test, Y_rot_cls_test)
    print(f"Final Test Accuracies:")
    print(f"  Digit Classification: {final_digit_acc:.4f}")
    print(f"  Rotation Detection:   {final_rotation_acc:.4f}")

    # Show some predictions
    print(f"\nSample Predictions (first 10 test samples):")
    digit_pred, rotation_pred = model.predict(X_test[:10])

    print("True -> Predicted")
    print("Digit/Rotation -> Digit/Rotation")
    for i in range(10):
        true_digit = Y_test[i]
        true_rotation = Y_rotation_test[i]
        pred_digit = np.argmax(digit_pred[i])
        pred_rotation = np.argmax(rotation_pred[i]) * 90

        correct = "✓" if (true_digit == pred_digit and true_rotation == pred_rotation) else "✗"
        print(f"{true_digit}/{true_rotation:3d}° -> {pred_digit}/{pred_rotation:3d}° {correct}")

    print("\nTraining completed successfully!")
    print("This replicates main.py functionality without TensorFlow!")


if __name__ == "__main__":
    main()
