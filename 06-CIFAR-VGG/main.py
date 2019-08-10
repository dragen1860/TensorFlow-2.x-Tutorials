import  os
import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers
import  argparse
import  numpy as np



from    network import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
argparser = argparse.ArgumentParser()



argparser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                           help="Directory where to write event logs and checkpoint.")
argparser.add_argument('--max_steps', type=int, default=1000000,
                            help="""Number of batches to run.""")
argparser.add_argument('--log_device_placement', action='store_true',
                            help="Whether to log device placement.")
argparser.add_argument('--log_frequency', type=int, default=10,
                            help="How often to log results to the console.")


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def prepare_cifar(x, y):

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y



def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def main():

    tf.random.set_seed(22)

    print('loading data...')
    (x,y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    # x = tf.convert_to_tensor(x)
    # y = tf.convert_to_tensor(y)
    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')




    model = VGG16([32, 32, 3])


    # must specify from_logits=True!
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)


    for epoch in range(250):

        for step, (x, y) in enumerate(train_loader):
            # [b, 1] => [b]
            y = tf.squeeze(y, axis=1)
            # [b, 10]
            y = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
                # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
                # mse_loss = tf.reduce_sum(tf.square(y-logits))
                # print(y.shape, logits.shape)
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            # MUST clip gradient here or it will disconverge!
            grads = [ tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                # for g in grads:
                #     print(tf.norm(g).numpy())
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()


        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                # [b, 1] => [b]
                y = tf.squeeze(y, axis=1)
                # [b, 10]
                y = tf.one_hot(y, depth=10)

                logits = model.predict(x)
                # be careful, these functions can accept y as [b] without warnning.
                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()







if __name__ == '__main__':
    main()