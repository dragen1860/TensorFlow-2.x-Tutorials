'''ResNet model for Keras.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

'''
import  tensorflow as tf
from    tensorflow.keras import layers

class _Bottleneck(tf.keras.Model):

    def __init__(self, filters, block, 
                 downsampling=False, stride=1, **kwargs):
        super(_Bottleneck, self).__init__(**kwargs)

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + block + '_branch'
        bn_name_base   = 'bn'  + block + '_branch'

        self.downsampling = downsampling
        self.stride = stride
        self.out_channel = filters3
        
        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=(stride, stride),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(filters2, (3, 3), padding='same',
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(filters3, (1, 1),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(name=bn_name_base + '2c')
         
        if self.downsampling:
            self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=(stride, stride),
                                               kernel_initializer='he_normal',
                                               name=conv_name_base + '1')
            self.bn_shortcut = layers.BatchNormalization(name=bn_name_base + '1')     
    
    def call(self, inputs, training=False):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        
        if self.downsampling:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut, training=training)
        else:
            shortcut = inputs
            
        x += shortcut
        x = tf.nn.relu(x)
        
        return x
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()

        shape[1] = shape[1] // self.stride
        shape[2] = shape[2] // self.stride
        shape[-1] = self.out_channel
        return tf.TensorShape(shape)        
        

class ResNet(tf.keras.Model):

    def __init__(self, depth, **kwargs):
        super(ResNet, self).__init__(**kwargs)
              
        if depth not in [50, 101]:
            raise AssertionError('depth must be 50 or 101.')
        self.depth = depth
    
        self.padding = layers.ZeroPadding2D((3, 3))
        self.conv1 = layers.Conv2D(64, (7, 7),
                                   strides=(2, 2),
                                   kernel_initializer='he_normal',
                                   name='conv1')
        self.bn_conv1 = layers.BatchNormalization(name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        
        self.res2a = _Bottleneck([64, 64, 256], block='2a',
                                 downsampling=True, stride=1)
        self.res2b = _Bottleneck([64, 64, 256], block='2b')
        self.res2c = _Bottleneck([64, 64, 256], block='2c')
        
        self.res3a = _Bottleneck([128, 128, 512], block='3a', 
                                 downsampling=True, stride=2)
        self.res3b = _Bottleneck([128, 128, 512], block='3b')
        self.res3c = _Bottleneck([128, 128, 512], block='3c')
        self.res3d = _Bottleneck([128, 128, 512], block='3d')
        
        self.res4a = _Bottleneck([256, 256, 1024], block='4a', 
                                 downsampling=True, stride=2)
        self.res4b = _Bottleneck([256, 256, 1024], block='4b')
        self.res4c = _Bottleneck([256, 256, 1024], block='4c')
        self.res4d = _Bottleneck([256, 256, 1024], block='4d')
        self.res4e = _Bottleneck([256, 256, 1024], block='4e')
        self.res4f = _Bottleneck([256, 256, 1024], block='4f')
        if self.depth == 101:
            self.res4g = _Bottleneck([256, 256, 1024], block='4g')
            self.res4h = _Bottleneck([256, 256, 1024], block='4h')
            self.res4i = _Bottleneck([256, 256, 1024], block='4i')
            self.res4j = _Bottleneck([256, 256, 1024], block='4j')
            self.res4k = _Bottleneck([256, 256, 1024], block='4k')
            self.res4l = _Bottleneck([256, 256, 1024], block='4l')
            self.res4m = _Bottleneck([256, 256, 1024], block='4m')
            self.res4n = _Bottleneck([256, 256, 1024], block='4n')
            self.res4o = _Bottleneck([256, 256, 1024], block='4o')
            self.res4p = _Bottleneck([256, 256, 1024], block='4p')
            self.res4q = _Bottleneck([256, 256, 1024], block='4q')
            self.res4r = _Bottleneck([256, 256, 1024], block='4r')
            self.res4s = _Bottleneck([256, 256, 1024], block='4s')
            self.res4t = _Bottleneck([256, 256, 1024], block='4t')
            self.res4u = _Bottleneck([256, 256, 1024], block='4u')
            self.res4v = _Bottleneck([256, 256, 1024], block='4v')
            self.res4w = _Bottleneck([256, 256, 1024], block='4w') 
        
        self.res5a = _Bottleneck([512, 512, 2048], block='5a', 
                                 downsampling=True, stride=2)
        self.res5b = _Bottleneck([512, 512, 2048], block='5b')
        self.res5c = _Bottleneck([512, 512, 2048], block='5c')
        
        
        self.out_channel = (256, 512, 1024, 2048)
    
    def call(self, inputs, training=True):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        
        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        C2 = x = self.res2c(x, training=training)
        
        x = self.res3a(x, training=training)
        x = self.res3b(x, training=training)
        x = self.res3c(x, training=training)
        C3 = x = self.res3d(x, training=training)
        
        x = self.res4a(x, training=training)
        x = self.res4b(x, training=training)
        x = self.res4c(x, training=training)
        x = self.res4d(x, training=training)
        x = self.res4e(x, training=training)
        x = self.res4f(x, training=training)
        if self.depth == 101:
            x = self.res4g(x, training=training)
            x = self.res4h(x, training=training)
            x = self.res4i(x, training=training)
            x = self.res4j(x, training=training)
            x = self.res4k(x, training=training)
            x = self.res4l(x, training=training)
            x = self.res4m(x, training=training)
            x = self.res4n(x, training=training)
            x = self.res4o(x, training=training)
            x = self.res4p(x, training=training)
            x = self.res4q(x, training=training)
            x = self.res4r(x, training=training)
            x = self.res4s(x, training=training)
            x = self.res4t(x, training=training)
            x = self.res4u(x, training=training)
            x = self.res4v(x, training=training)
            x = self.res4w(x, training=training) 
        C4 = x
        
        x = self.res5a(x, training=training)
        x = self.res5b(x, training=training)
        C5 = x = self.res5c(x, training=training)
        
        return (C2, C3, C4, C5)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch, H, W, C = shape
        
        C2_shape = tf.TensorShape([batch, H //  4, W //  4, self.out_channel[0]])
        C3_shape = tf.TensorShape([batch, H //  8, W //  8, self.out_channel[1]])
        C4_shape = tf.TensorShape([batch, H // 16, W // 16, self.out_channel[2]])
        C5_shape = tf.TensorShape([batch, H // 32, W // 32, self.out_channel[3]])
        
        return (C2_shape, C3_shape, C4_shape, C5_shape)