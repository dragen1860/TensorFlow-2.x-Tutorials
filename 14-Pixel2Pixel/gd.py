import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras




class Downsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(keras.Model):

    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, x2, training=None):

        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = keras.layers.Conv2DTranspose(3, (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)


    def call(self, x, training=None):

        # x shape == (bs, 256, 256, 3)    
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class DiscDownsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = keras.layers.Conv2D(filters, (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training=None):

        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        # we are zero padding here with 1 because we need our shape to 
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = keras.layers.ZeroPadding2D()
        self.conv = keras.layers.Conv2D(512, (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = keras.layers.ZeroPadding2D()
        self.last = keras.layers.Conv2D(1, (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)


    def call(self, inputs, training=None):
        inp, target = inputs

        # concatenating the input and the target
        x = tf.concat([inp, target], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 30, 30, 1)

        return x