import  os
import  time
import  numpy as np
import  matplotlib.pyplot as plt
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras


from    model import Generator, Discriminator, cycle_consistency_loss, generator_loss, discriminator_loss




tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


learning_rate = 0.0002
batch_size = 1  # Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
epochs = 1000

"""### Load Datasets"""

path_to_zip = keras.utils.get_file('horse2zebra.zip',
                              cache_subdir=os.path.abspath('.'),
                              origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                              extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'horse2zebra/')

trainA_path = os.path.join(PATH, "trainA")
trainB_path = os.path.join(PATH, "trainB")

trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
print('train A:', trainA_size)
print('train B:', trainB_size)


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    # scale alread!
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    # image = (image / 127.5) - 1
    image = image * 2 - 1

    return image


train_datasetA = tf.data.Dataset.list_files(PATH + 'trainA/*.jpg', shuffle=False)
train_datasetA = train_datasetA.shuffle(trainA_size).repeat(epochs)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(batch_size)
train_datasetA = train_datasetA.prefetch(batch_size)
train_datasetA = iter(train_datasetA)

train_datasetB = tf.data.Dataset.list_files(PATH + 'trainB/*.jpg', shuffle=False)
train_datasetB = train_datasetB.shuffle(trainB_size).repeat(epochs)
train_datasetB = train_datasetB.map(lambda x: load_image(x))
train_datasetB = train_datasetB.batch(batch_size)
train_datasetB = train_datasetB.prefetch(batch_size)
train_datasetB = iter(train_datasetB)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discA = Discriminator()
discB = Discriminator()
genA2B = Generator()
genB2A = Generator()

discA_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
discB_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genB2A_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)



def generate_images(A, B, B2A, A2B, epoch):
    """

    :param A:
    :param B:
    :param B2A:
    :param A2B:
    :param epoch:
    :return:
    """
    plt.figure(figsize=(15, 15))
    A = tf.reshape(A, [256, 256, 3]).numpy()
    B = tf.reshape(B, [256, 256, 3]).numpy()
    B2A = tf.reshape(B2A, [256, 256, 3]).numpy()
    A2B = tf.reshape(A2B, [256, 256, 3]).numpy()
    display_list = [A, B, A2B, B2A]

    title = ['A', 'B', 'A2B', 'B2A']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('images/generated_%d.png'%epoch)
    plt.close()


def train(train_datasetA, train_datasetB, epochs, lsgan=True, cyc_lambda=10):

    for epoch in range(epochs):
        start = time.time()


        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
            try:
                # Next training minibatches, default size 1
                trainA = next(train_datasetA)
                trainB = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            genA2B_output = genA2B(trainA, training=True)
            genB2A_output = genB2A(trainB, training=True)


            discA_real_output = discA(trainA, training=True)
            discB_real_output = discB(trainB, training=True)

            discA_fake_output = discA(genB2A_output, training=True)
            discB_fake_output = discB(genA2B_output, training=True)

            reconstructedA = genB2A(genA2B_output, training=True)
            reconstructedB = genA2B(genB2A_output, training=True)

            # generate_images(reconstructedA, reconstructedB)

            # Use history buffer of 50 for disc loss
            discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=lsgan)
            discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)

            genA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan) + \
                          cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB,
                                                 cyc_lambda=cyc_lambda)
            genB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan) + \
                          cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB,
                                                 cyc_lambda=cyc_lambda)

        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
        genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.trainable_variables)

        discA_gradients = discA_tape.gradient(discA_loss, discA.trainable_variables)
        discB_gradients = discB_tape.gradient(discB_loss, discB.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        genB2A_optimizer.apply_gradients(zip(genB2A_gradients, genB2A.trainable_variables))

        discA_optimizer.apply_gradients(zip(discA_gradients, discA.trainable_variables))
        discB_optimizer.apply_gradients(zip(discB_gradients, discB.trainable_variables))


        if epoch % 40 == 0:
            generate_images(trainA, trainB, genB2A_output, genA2B_output, epoch)

            print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))




if __name__ == '__main__':
    train(train_datasetA, train_datasetB, epochs=epochs, lsgan=True, cyc_lambda=cyc_lambda)
