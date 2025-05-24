# Neural Network from Scratch - Student Project

This repository contains educational implementations of neural networks for digit classification and rotation detection, created as a learning exercise to understand deep learning fundamentals without relying on high-level frameworks.

## 🎯 Project Overview

**Objective**: Build a multi-task neural network that can:
1. **Classify digits** (0, 1, 2, 3, etc.)
2. **Detect rotation angles** (0°, 90°, 180°, 270°)

**Key Learning Goals**:
- Understand neural network architecture from first principles
- Implement forward propagation manually
- Implement backpropagation algorithm from scratch
- Learn multi-task learning concepts
- Practice with gradient descent optimization

## 📁 Files Description

### 1. `main.py` (Original TensorFlow Implementation)
- **Framework**: TensorFlow/Keras
- **Dataset**: Real MNIST digits
- **Features**: 
  - Complete CNN architecture with pooling layers
  - Data augmentation with rotations
  - Professional-grade training loop
  - Visualization of results

### 2. `mnist_nn_simple.py` (No TensorFlow - MNIST Version) ⭐ **RECOMMENDED**
- **Framework**: Only Numpy + SciPy
- **Dataset**: Real MNIST dataset (60k training, 10k test)
- **Features**:
  - Replicates main.py functionality without TensorFlow
  - Real MNIST data loading from scratch
  - Multi-task neural network implementation
  - Training, prediction, and evaluation phases
  - No visualization dependencies

### 3. `basic_digit_rotation_classifier.py` (Educational Version)
- **Framework**: Numpy + basic libraries
- **Dataset**: Real MNIST (with fallback to synthetic data)
- **Features**:
  - Manual neural network implementation
  - Step-by-step educational approach
  - Comprehensive documentation
  - Visualization and analysis tools

### 4. `basic_nn_from_scratch.py` (Pure Implementation)
- **Framework**: Only Numpy + SciPy
- **Dataset**: Custom 8x8 digit patterns
- **Features**:
  - Complete implementation from scratch
  - No external dependencies issues
  - Perfect for learning core concepts
  - Detailed explanations of each step

## 🧠 Neural Network Architecture

```
Input Layer (64 neurons)
         ↓
Hidden Layer (32 neurons, ReLU activation)
         ↓
    ┌────────────┬────────────┐
    ↓            ↓            
Digit Output   Rotation Output
(4 neurons,    (4 neurons,
 Softmax)       Softmax)
```

### Key Components Implemented:

1. **Activation Functions**:
   - ReLU: `f(x) = max(0, x)`
   - Softmax: `f(x) = exp(x) / sum(exp(x))`

2. **Loss Function**:
   - Cross-entropy loss for classification

3. **Optimization**:
   - Gradient descent with backpropagation
   - Mini-batch training

4. **Multi-task Learning**:
   - Shared hidden representation
   - Separate output heads for different tasks

## 🚀 How to Run

### Quick Start - MNIST without TensorFlow (Recommended)
```bash
python3 mnist_nn_simple.py
```

### Educational Version - Simple Patterns
```bash
python3 basic_nn_from_scratch.py
```

### Full MNIST Version (with visualization, if dependencies work)
```bash
python3 basic_digit_rotation_classifier.py
```

### Original TensorFlow Version
```bash
python3 main.py
```

## 📊 Expected Results

The `basic_nn_from_scratch.py` should achieve:
- **Digit Classification**: ~100% accuracy
- **Rotation Detection**: ~100% accuracy

Example output:
```
Final Test Accuracies:
  Digit Classification: 100.0%
  Rotation Detection:   100.0%

Example Predictions (first 10 samples):
True -> Predicted (Digit/Rotation)
  3/180° -> 3/180° ✓
  3/270° -> 3/270° ✓
  3/  0° -> 3/  0° ✓
```

## 🎓 Educational Value

### What You'll Learn:

1. **Forward Propagation**:
   - Matrix multiplications
   - Activation function applications
   - Multi-output handling

2. **Backpropagation**:
   - Gradient computation
   - Chain rule application
   - Weight updates

3. **Training Process**:
   - Mini-batch processing
   - Loss calculation
   - Optimization loops

4. **Multi-task Learning**:
   - Shared representations
   - Multiple loss functions
   - Joint optimization

### Code Highlights:

```python
# Forward propagation
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = ActivationFunctions.relu(self.z1)
    
    self.z2_digit = np.dot(self.a1, self.W2_digit) + self.b2_digit
    self.z2_rotation = np.dot(self.a1, self.W2_rotation) + self.b2_rotation
    
    self.digit_probs = ActivationFunctions.softmax(self.z2_digit)
    self.rotation_probs = ActivationFunctions.softmax(self.z2_rotation)
    
    return self.digit_probs, self.rotation_probs

# Backpropagation
def backward(self, X, y_digit, y_rotation, learning_rate=0.01):
    # ... gradient calculations ...
    # Update weights using computed gradients
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
    # ... etc ...
```

## 🔧 Implementation Details

### Data Augmentation Strategy:
- **Digit 0**: Only original orientation (looks same when rotated)
- **Other digits**: All 4 rotations (0°, 90°, 180°, 270°)

### Network Training:
- **Optimizer**: Gradient Descent
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Epochs**: 50

### Multi-task Loss:
```
Total Loss = Digit Classification Loss + Rotation Detection Loss
```

## 🎯 Key Differences from Original

| Aspect | Original (`main.py`) | Educational (`basic_nn_from_scratch.py`) |
|--------|---------------------|----------------------------------------|
| Framework | PyTorch | Pure Numpy |
| Dataset | Full MNIST | Simple 8x8 patterns |
| Architecture | CNN with pooling | Simple feedforward |
| Dependencies | torch, torchvision | numpy, scipy |
| Focus | Performance | Learning |

## 🏆 Achievements

✅ **Complete neural network from scratch**  
✅ **Multi-task learning implementation**  
✅ **100% accuracy on test data**  
✅ **Educational documentation**  
✅ **No external framework dependencies**  
✅ **Step-by-step learning approach**  

## 🚀 Next Steps

To extend this project, consider:

1. **Add more digits** (expand to full 0-9 classification)
2. **Implement different architectures** (add more hidden layers)
3. **Try different optimizers** (momentum, Adam)
4. **Add regularization** (dropout, L2 regularization)
5. **Experiment with learning rates** (learning rate scheduling)
6. **Implement convolutional layers** from scratch

## 📝 Learning Outcomes

By completing this project, you will have:

- ✅ Built a neural network completely from scratch
- ✅ Understood forward and backward propagation
- ✅ Implemented multi-task learning
- ✅ Learned gradient descent optimization
- ✅ Practiced with real machine learning problems
- ✅ Gained deep understanding of neural network fundamentals

---

**Created as an educational project to demonstrate neural network implementation from first principles.** 