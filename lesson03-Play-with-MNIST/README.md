# Play with MNIST

A detailed MNIST walk-through!

Let's start by loading MNIST from **keras.datasets** and preprocessing to get rows of normalized 784-dimensional vectors.

```python
import  tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(xs, ys),_ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)
```

<img src="mnist.gif" align="right" width="270" height="270">

Now let's build our network as a **keras.Sequential** model and instantiate a stochastic gradient descent optimizer from **keras.optimizers**.

```python
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()
```



Finally, we can iterate through our dataset and train our model.
In this example, we use **tf.GradientTape** to manually compute the gradients of the loss with respect to our network's trainable variables. GradientTape is just one of many ways to perform gradient steps in TensorFlow 2.0:

- **Tf.GradientTape:** Manually computes loss gradients with respect to given variables by recording operations within its context manager. This is the most flexible way to perform optimizer steps, as we can work directly with gradients and don't need a pre-defined Keras model or loss function.
- **Model.train():** Keras's built-in function for iterating through a dataset and fitting a Keras.Model on it. This is often the best choice for training a Keras model and comes with options for progress bar displays, validation splits, multiprocessing, and generator support.
- **Optimizer.minimize():** Computes and differentiates through a given loss function and performs a step to minimize it with gradient descent. This method is easy to implement, and can be conveniently slapped onto any existing computational graph to make a working optimization step.

```python
for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28*28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b, 10]
        loss = tf.square(out-y_onehot)
        # [b]
        loss = tf.reduce_sum(loss) / 32


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
```

# HowTO

Try it for yourself!

```
python main.py
``` 