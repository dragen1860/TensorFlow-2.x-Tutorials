import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import  time
from    matplotlib import pyplot as plt

from    gd import Discriminator, Generator

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


batch_size = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


path_to_zip = keras.utils.get_file('facades.tar.gz',
                                  cache_subdir=os.path.abspath('.'),
                                  origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
                                  extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
print('dataset path:', PATH)

def load_image(image_file, is_train):
    """
    load and preprocess images
    :param image_file:
    :param is_train:
    :return:
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = image.shape[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random jittering

        # resizing to 286 x 286 x 3
        input_image = tf.image.resize(input_image, [286, 286])
        real_image = tf.image.resize(real_image, [286, 286])

        # randomly cropping to 256 x 256 x 3
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        if np.random.random() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image = tf.image.resize(input_image, size=[IMG_HEIGHT, IMG_WIDTH])
        real_image = tf.image.resize(real_image, size=[IMG_HEIGHT, IMG_WIDTH])

    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    # [256, 256, 3], [256, 256, 3]
    # print(input_image.shape, real_image.shape)

    # => [256, 256, 6]
    out = tf.concat([input_image, real_image], axis=2)

    return out





train_dataset = tf.data.Dataset.list_files(PATH+'/train/*.jpg')
# The following snippet can not work, so load it hand by hand.
# train_dataset = train_dataset.map(lambda x: load_image(x, True)).batch(1)
train_iter = iter(train_dataset)
train_data = []
for x in train_iter:
    train_data.append(load_image(x, True))
train_data = tf.stack(train_data, axis=0)
# [800, 256, 256, 3]
print('train:', train_data.shape)
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.shuffle(400).batch(1)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
# test_dataset = test_dataset.map(lambda x: load_image(x, False)).batch(1)
test_iter = iter(test_dataset)
test_data = []
for x in test_iter:
    test_data.append(load_image(x, False))
test_data = tf.stack(test_data, axis=0)
# [800, 256, 256, 3]
print('test:', test_data.shape)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_dataset = test_dataset.shuffle(400).batch(1)

generator = Generator()
generator.build(input_shape=(batch_size, 256, 256, 3))
generator.summary()
discriminator = Discriminator()
discriminator.build(input_shape=[(batch_size, 256, 256, 3), (batch_size, 256, 256, 3)])
discriminator.summary()

g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)




def discriminator_loss(disc_real_output, disc_generated_output):
    # [1, 30, 30, 1] with [1, 30, 30, 1]
    # print(disc_real_output.shape, disc_generated_output.shape)
    real_loss = keras.losses.binary_crossentropy(
                    tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

    generated_loss = keras.losses.binary_crossentropy(
                    tf.zeros_like(disc_generated_output), disc_generated_output, from_logits=True)

    real_loss = tf.reduce_mean(real_loss)
    generated_loss = tf.reduce_mean(generated_loss)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss



def generator_loss(disc_generated_output, gen_output, target):

    LAMBDA = 100

    gan_loss = keras.losses.binary_crossentropy(
                tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    gan_loss = tf.reduce_mean(gan_loss)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss



def generate_images(model, test_input, tar, epoch):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('images/epoch%d.png'%epoch)
    print('saved images.')
    # plt.show()








def main():

    epochs = 1000

    for epoch in range(epochs):

        start = time.time()

        for step, inputs in enumerate(train_dataset):

            input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
            # print(input_image.shape, target.shape)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # get generated pixel2pixel image
                gen_output = generator(input_image, training=True)
                # fed real pixel2pixel image together with original image
                disc_real_output = discriminator([input_image, target], training=True)
                # fed generated/fake pixel2pixel image together with original image
                disc_generated_output = discriminator([input_image, gen_output], training=True)

                gen_loss = generator_loss(disc_generated_output, gen_output, target)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            if step% 100 == 0:
                # print(disc_loss.shape, gen_loss.shape)
                print(epoch, step, float(disc_loss), float(gen_loss))

        if epoch % 1 == 0:

            for inputs in test_dataset:
                input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
                generate_images(generator, input_image, target, epoch)
                break

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))



    for inputs in test_dataset:
        input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
        generate_images(generator, input_image, target, 99999)
        break





if __name__ == '__main__':
    main()