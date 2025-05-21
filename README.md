# MNIST Digit Recognition with Rotation Detection

This project implements a deep learning model that can simultaneously recognize handwritten digits and detect their rotation angles from the MNIST dataset.

## Features

- Multi-task learning: Digit recognition and rotation angle detection
- Data augmentation with rotation transformations
- CNN-based architecture with two output heads
- Visualization of training results and predictions

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Perform data augmentation with rotations
3. Train the model
4. Display training progress and results
5. Show example predictions

## Model Architecture

The model uses a CNN architecture with:
- Two convolutional layers with max pooling
- Dense layers for feature extraction
- Two output heads:
  - Digit classification (10 classes)
  - Rotation angle classification (4 classes: 0°, 90°, 180°, 270°)

## Results

The model provides:
- Digit recognition accuracy
- Rotation angle detection accuracy
- Visual examples of predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details. 