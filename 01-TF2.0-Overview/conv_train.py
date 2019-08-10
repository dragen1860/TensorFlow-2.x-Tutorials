import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


model = tf.keras.Sequential([
    layers.Reshape(
        target_shape=[28, 28, 1],
        input_shape=(28, 28,)),
    layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(rate=0.4),
    layers.Dense(10)])

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.5)


def mnist_datasets():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, test_dataset


train_ds, test_ds = mnist_datasets()
train_ds = train_ds.shuffle(60000).batch(100)
test_ds = test_ds.batch(100)


def train_step(model, optimizer, images, labels):

    # Record the operations used to compute the loss, so that the gradient
    # of the loss with respect to the variables can be computed.
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = compute_loss(labels, logits)
        compute_accuracy(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(model, optimizer, dataset, log_freq=50):
    """
    Trains model on `dataset` using `optimizer`.
    """
    # Metrics are stateful. They accumulate values and return a cumulative
    # result when you call .result(). Clear accumulated values with .reset_states()
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    # Datasets can be iterated over like any other Python iterable.
    for images, labels in dataset:
        loss = train_step(model, optimizer, images, labels)
        avg_loss(loss)

        if tf.equal(optimizer.iterations % log_freq, 0):
            # summary_ops_v2.scalar('loss', avg_loss.result(), step=optimizer.iterations)
            # summary_ops_v2.scalar('accuracy', compute_accuracy.result(), step=optimizer.iterations)
            print('step:', int(optimizer.iterations),
                  'loss:', avg_loss.result().numpy(),
                  'acc:', compute_accuracy.result().numpy())
            avg_loss.reset_states()
            compute_accuracy.reset_states()


def test(model, dataset, step_num):
    """
    Perform an evaluation of `model` on the examples from `dataset`.
    """
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    for (images, labels) in dataset:
        logits = model(images, training=False)
        avg_loss(compute_loss(labels, logits))
        compute_accuracy(labels, logits)

    print('Model test set loss: {:0.4f} accuracy: {:0.2f}%'.format(
        avg_loss.result(), compute_accuracy.result() * 100))

    print('loss:', avg_loss.result(), 'acc:', compute_accuracy.result())
    # summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
    # summary_ops_v2.scalar('accuracy', compute_accuracy.result(), step=step_num)


# Where to save checkpoints, tensorboard summaries, etc.
MODEL_DIR = '/tmp/tensorflow/mnist'


def apply_clean():
    if tf.io.gfile.exists(MODEL_DIR):
        print('Removing existing model dir: {}'.format(MODEL_DIR))
        tf.io.gfile.rmtree(MODEL_DIR)


apply_clean()

checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Restore variables on creation if a checkpoint exists.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

NUM_TRAIN_EPOCHS = 5

for i in range(NUM_TRAIN_EPOCHS):
    start = time.time()
    #   with train_summary_writer.as_default():
    train(model, optimizer, train_ds, log_freq=500)
    end = time.time()
    print('Train time for epoch #{} ({} total steps): {}'.format(
        i + 1, int(optimizer.iterations), end - start))
    #   with test_summary_writer.as_default():
    #     test(model, test_ds, optimizer.iterations)
    checkpoint.save(checkpoint_prefix)
    print('saved checkpoint.')

export_path = os.path.join(MODEL_DIR, 'export')
tf.saved_model.save(model, export_path)
print('saved SavedModel for exporting.')