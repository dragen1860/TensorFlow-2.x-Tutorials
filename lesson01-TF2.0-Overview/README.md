# TensorFlow Overview

A quick look at some MNIST examples to get familiar with the core features of TensorFlow 2.0:

- **tf.keras:** A high-level, object-oriented API for fast prototyping of deep learning models
- **tf.GradientTape:** Records gradients on-the-fly for automatic differentiation and backprop
- **tf.train:** Optimizers for training and checkpoints for exporting models

# HowTO
Train MNIST with a **fully connected network**:
```
python fc_train.py
```

Train MNIST with a **convolutional network**:
```
python conv_train.py
```

![](features.png)