import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras



class Encoder(keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        # Small variance in initialization helps with preventing colour inversion.
        self.conv1 = keras.layers.Conv2D(32, kernel_size=7, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        
        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        # Implement instance norm to more closely match orig. paper (momentum=0.1)?
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        return x


class Residual(keras.Model):

    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = keras.layers.Conv2D(128, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        

        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization() 

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = tf.add(x, inputs)

        return x


class Decoder(keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(3, kernel_size=7, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))


        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.tanh(x)

        return x


class Generator(keras.Model):

    def __init__(self, img_size=256, skip=False):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.skip = skip  # TODO: Add skip

        self.encoder = Encoder()
        if (img_size == 128):
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
        else:
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
            self.res7 = Residual()
            self.res8 = Residual()
            self.res9 = Residual()
        self.decoder = Decoder()


    def call(self, inputs, training=True):

        x = self.encoder(inputs, training)
        if (self.img_size == 128):
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
        else:
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
            x = self.res7(x, training)
            x = self.res8(x, training)
            x = self.res9(x, training)
        x = self.decoder(x, training)

        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.leaky = keras.layers.LeakyReLU(0.2)


        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()


    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)

        x = self.conv5(x)
        # x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan
        return x






def discriminator_loss(disc_of_real_output, disc_of_gen_output, lsgan=True):
    if lsgan:  # Use least squares loss
        # real_loss = tf.reduce_mean(tf.squared_difference(disc_of_real_output, 1))
        real_loss = keras.losses.mean_squared_error(disc_of_real_output, tf.ones_like(disc_of_real_output))
        generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))

        total_disc_loss = (real_loss + generated_loss) * 0.5  # 0.5 slows down rate that D learns compared to G
    else:  # Use vanilla GAN loss
        raise NotImplementedError
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_real_output),
                                                    logits=disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_of_gen_output),
                                                         logits=disc_of_gen_output)

        total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_of_gen_output, lsgan=True):
    if lsgan:  # Use least squares loss
        # gen_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, 1))
        gen_loss = keras.losses.mean_squared_error(disc_of_gen_output, tf.ones_like(disc_of_gen_output))
    else:  # Use vanilla GAN loss
        raise NotImplementedError
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output),
                                                   logits=disc_generated_output)
        # l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # Look up pix2pix loss
    return gen_loss


def cycle_consistency_loss(data_A, data_B, reconstructed_data_A, reconstructed_data_B, cyc_lambda=10):
    loss = tf.reduce_mean(tf.abs(data_A - reconstructed_data_A) + tf.abs(data_B - reconstructed_data_B))
    return cyc_lambda * loss
