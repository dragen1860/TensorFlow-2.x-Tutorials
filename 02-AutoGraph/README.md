# AutoGraph


Compare static graph using @tf.function VS dynamic graph.

AutoGraph helps you write complicated graph code using normal Python. Behind the scenes, AutoGraph automatically transforms your code into the equivalent TensorFlow graph code.

Let's take a look at TensorFlow graphs and how they work.

```python
ReLU_Layer = tf.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer = tf.keras.layers.Dense(10, input_shape=(100,))

# X and y are labels and inputs
```

<img src="graph.gif" align="left" width="302" height="538">

**TensorFlow 1.0:** Operations are added as nodes to the computational graph and are not actually executed until we call session.run(), much like defining a function that doesn't run until it is called.

```python
SGD_Trainer = tf.train.GradientDescentOptimizer(1e-2)

inputs = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.int16, shape=[None, 10])
hidden = ReLU_Layer(inputs)
logits = Logit_Layer(hidden)
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
loss = tf.reduce_mean(entropy)
train_step = SGD_Trainer.minimize(loss, 
    var_list=ReLU_Layer.weights+Logit_Layer.weights)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for step in range(1000):
    sess.run(train_step, feed_dict={inputs:X, labels:y})
```

**TensorFlow 2.0:** Operations are executed directly and the computational graph is built on-the-fly. However, we can still write functions and pre-compile computational graphs from them like in TF 1.0 using the *@tf.function* decorator, allowing for faster execution.

```python
SGD_Trainer = tf.optimizers.SGD(1e-2)

@tf.function
def loss_fn(inputs=X, labels=y):
    hidden = ReLU_Layer(inputs)
    logits = Logit_Layer(hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(entropy)

for step in range(1000):
    SGD_Trainer.minimize(loss_fn, 
        var_list=ReLU_Layer.weights+Logit_Layer.weights)
```

# HowTO

```
python main.py
```

and you will see some computation cost between static graph and dynamic graph.