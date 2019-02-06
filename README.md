# TensorFlow 2.0 Tutorials

![2.0](res/tensorflow-2.0.jpg)

Timeline:
- Jan. 11, 2019: [TensorFlow r2.0 preview](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
- Aug. 14, 2018: [TensorFlow 2.0 is coming](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/bgug1G6a89A)

# TensorFlow 1.0 is out-of-date!

[TensorFlow Sucks](http://nicodjimenez.github.io/2017/10/08/tensorflow.html).

We have TensorFlow 2.0 finally.

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
Simple classification example to show you how to train, load/save checkpoint, evaluate with new TF 2.0.

# Lesson 14
4 ways to achieve simple MNIST classifcation with TF 2.0.
1. keras compile & fit
2. forward and backward with keras standard layers.Dense
3. forward and backward with keras subclassing custom layer
4. forward and backward with tf.matmul

