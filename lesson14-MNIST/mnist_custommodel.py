import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def mnist_dataset():
  (x, y), (x_val, y_val) = datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.shuffle(60000).batch(100)


  ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  ds_val = ds_val.map(prepare_mnist_features_and_labels)
  ds_val = ds_val.shuffle(10000).batch(100)

  return ds,ds_val






def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))





class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        # self.model = keras.Sequential([
        #     layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        #     layers.Dense(100, activation='relu'),
        #     layers.Dense(100, activation='relu'),
        #     layers.Dense(10)])

        self.layer1 = layers.Dense(100, activation=tf.nn.relu)
        self.layer2 = layers.Dense(100, activation=tf.nn.relu)
        self.layer3 = layers.Dense(10)

    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 28*28])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

    train_dataset, val_dataset = mnist_dataset()

    model = MyModel()
    model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy',
                  metrics=['accuracy'])



    model.fit(train_dataset.repeat(), epochs=30, steps_per_epoch=500, verbose=1,
              validation_data=val_dataset.repeat(),
              validation_steps=1
              )


if __name__ == '__main__':
    main()