# TensorFlow 2.0 Tutorials

![2.0](res/tensorflow-2.0.jpg)

Timeline:
- Jan. 11, 2019: [TensorFlow r2.0 preview](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
- Aug. 14, 2018: [TensorFlow 2.0 is coming](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/bgug1G6a89A)

# TensorFlow 1.0 is DEAD!

[TensorFlow Sucks](http://nicodjimenez.github.io/2017/10/08/tensorflow.html).

In the past period, we have no choice but to stand the extremely bad TensorFlow 1.\*, and  a lot of people turn their heads to PyTorch(Yes!) to save time and money.

But now, we (will) have TensorFlow 2.0 finally.

Let's get started!

# Installation

1. CPU install
```python
pip install tf-nightly-2.0-preview
```

2. GPU install
```python
pip install tf-nightly-gpu-2.0-preview
```

Test installation:
```python
In [2]: import tensorflow  as tf

In [3]: tf.__version__
Out[3]: '2.0.0-dev20190129'
In [4]: tf.test.is_gpu_available()
...
totalMemory: 3.95GiB freeMemory: 3.00GiB
...
Out[4]: True

```


# Lesson1
Simple MNIST example to have a while glimpse of TF2.0
```
python main.py
```
```
epoch 0 : loss 2.379162 ; accuracy 0.09
epoch 1 : loss 0.23471013 ; accuracy 0.94
epoch 2 : loss 0.16313684 ; accuracy 0.94
epoch 3 : loss 0.12990189 ; accuracy 0.95
epoch 4 : loss 0.056507893 ; accuracy 0.99
epoch 5 : loss 0.05290828 ; accuracy 0.98
epoch 6 : loss 0.05637209 ; accuracy 0.98
epoch 7 : loss 0.09175427 ; accuracy 0.96
epoch 8 : loss 0.04277327 ; accuracy 1.0
epoch 9 : loss 0.010658806 ; accuracy 1.0
epoch 10 : loss 0.020584123 ; accuracy 0.99
epoch 11 : loss 0.008158814 ; accuracy 1.0
epoch 12 : loss 0.005305307 ; accuracy 1.0
epoch 13 : loss 0.0051660277 ; accuracy 1.0
epoch 14 : loss 0.0089087775 ; accuracy 1.0
epoch 15 : loss 0.0014608474 ; accuracy 1.0
epoch 16 : loss 0.01155874 ; accuracy 1.0
epoch 17 : loss 0.0078067114 ; accuracy 1.0
epoch 18 : loss 0.013161226 ; accuracy 0.99
epoch 19 : loss 0.0019802528 ; accuracy 1.0
Final epoch 19 : loss 0.0023814724 ; accuracy 1.0

```