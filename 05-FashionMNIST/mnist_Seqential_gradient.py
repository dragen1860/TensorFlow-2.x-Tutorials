import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def mnist_dataset():
  (x, y), _ = datasets.fashion_mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds






def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train_one_step(model, optimizer, x, y):

  with tf.GradientTape() as tape:

    logits = model(x)
    loss = compute_loss(logits, y)

  # compute gradient
  grads = tape.gradient(loss, model.trainable_variables)
  # update to weights
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  accuracy = compute_accuracy(logits, y)

  # loss and accuracy is scalar tensor
  return loss, accuracy


def train(epoch, model, optimizer):

  train_ds = mnist_dataset()
  loss = 0.0
  accuracy = 0.0
  for step, (x, y) in enumerate(train_ds):
    loss, accuracy = train_one_step(model, optimizer, x, y)
    if step%500==0:
      print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
  return loss, accuracy



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

    train_dataset = mnist_dataset()

    model = keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(10)])


    optimizer = optimizers.Adam()

    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())


if __name__ == '__main__':
    main()