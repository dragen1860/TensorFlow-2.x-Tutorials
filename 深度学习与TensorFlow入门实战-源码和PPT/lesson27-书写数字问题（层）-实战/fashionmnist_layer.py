import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)

batchsz = 128

db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]
    layers.Dense(10) # [b, 32] => [b, 10], 330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28])
model.summary()
# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

def main():


    for epoch in range(30):


        for step, (x,y) in enumerate(db):

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))


        # test
        total_correct = 0
        total_num = 0
        for x,y in db_test:

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)







if __name__ == '__main__':
    main()