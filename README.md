# Handwritten Digit Classifier (MNIST)

A feedforward neural network trained on the MNIST dataset to classify handwritten digits (0–9), built in a Kaggle notebook with TensorFlow/Keras. Built as a learning project — first pass at recognizing and addressing overfitting in a neural net.

## Overview

- **Dataset:** MNIST (60,000 training images, 10,000 test images, 28×28 grayscale)
- **Model:** Fully connected (dense) network — no convolutions
- **Framework:** TensorFlow / Keras
- **Test accuracy:** 97.79%

## Architecture

```
Flatten(input_shape=(28, 28))
Dense(128, activation='relu')
Dropout(0.3)
Dense(10, activation='softmax')
```

Total params: 101,770 (all trainable)

## Training

- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Epochs: up to 25, with early stopping (`monitor='val_loss'`, `patience=3`, `restore_best_weights=True`) — training stopped around epoch 13
- Validation split: 0.2
- Pixel values normalized to [0, 1] before training

## Results

Test accuracy: **97.79%**

## Overfitting: reduced, not eliminated

An earlier version of this model (no dropout, no early stopping, trained the full 25 epochs) showed clear overfitting — training loss kept falling to ~0.005 while validation loss climbed back up to ~0.12 after epoch 6. Adding dropout (0.3) and early stopping brought the final train/val loss gap down substantially (train ~0.05, val ~0.08–0.09) and improved test accuracy slightly (97.68% → 97.79%, though a gain this small is within normal run-to-run noise).

The underlying pattern is still visible, just smaller: validation loss bottoms out early and flattens while training loss keeps decreasing. This is a structural limitation of a dense (non-convolutional) architecture on image data, not something dropout/early stopping alone fully resolve — a CNN would likely reduce it further and push accuracy beyond dense-net's typical ~98% ceiling on MNIST.

## Usage

```python
import numpy as np
from tensorflow import keras

with np.load("mnist.npz") as data:
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]

X_train, X_test = X_train / 255.0, X_test / 255.0

model = keras.models.load_model("model.h5")  # if saved
prediction = model.predict(X_test[0].reshape(1, 28, 28))
```
