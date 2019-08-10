import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import  os


class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()

        # here must specify shape instead of tensor !
        # name here is meanless !
        # [dim_in, dim_out]
        self.w = self.add_variable('meanless-name', [13, 1])
        # [dim_out]
        self.b = self.add_variable('meanless-name', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)


    def call(self, x):

        x = tf.matmul(x, self.w) + self.b

        return x

def main():

    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')


    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()
    #
    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)
    # (404, 13) (404,) (102, 13) (102,)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    # Here has two mis-leading issues:
    # 1. (x_train, y_train) cant be written as [x_train, y_train]
    # 2.
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)


    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    for epoch in range(200):

        for step, (x, y) in enumerate(db_train):

            with tf.GradientTape() as tape:
                # [b, 1]
                logits = model(x)
                # [b]
                logits = tf.squeeze(logits, axis=1)
                # [b] vs [b]
                loss = criteon(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(epoch, 'loss:', loss.numpy())


        if epoch % 10 == 0:

            for x, y in db_val:
                # [b, 1]
                logits = model(x)
                # [b]
                logits = tf.squeeze(logits, axis=1)
                # [b] vs [b]
                loss = criteon(y, logits)

                print(epoch, 'val loss:', loss.numpy())





if __name__ == '__main__':
    main()