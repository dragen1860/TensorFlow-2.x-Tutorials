'''
FRN model for Keras.

# Reference:
- [Feature Pyramid Networks for Object Detection](
    https://arxiv.org/abs/1612.03144)

'''
import  tensorflow as tf
from    tensorflow.keras import layers

class FPN(tf.keras.Model):

    def __init__(self, out_channels=256, **kwargs):
        '''
        Feature Pyramid Networks
        
        Attributes
        ---
            out_channels: int. the channels of pyramid feature maps.
        '''
        super(FPN, self).__init__(**kwargs)
        
        self.out_channels = out_channels
        
        self.fpn_c2p2 = layers.Conv2D(out_channels, (1, 1), 
                                      kernel_initializer='he_normal', name='fpn_c2p2')
        self.fpn_c3p3 = layers.Conv2D(out_channels, (1, 1), 
                                      kernel_initializer='he_normal', name='fpn_c3p3')
        self.fpn_c4p4 = layers.Conv2D(out_channels, (1, 1), 
                                      kernel_initializer='he_normal', name='fpn_c4p4')
        self.fpn_c5p5 = layers.Conv2D(out_channels, (1, 1), 
                                      kernel_initializer='he_normal', name='fpn_c5p5')
        
        self.fpn_p3upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p3upsampled')
        self.fpn_p4upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self.fpn_p5upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        
        
        self.fpn_p2 = layers.Conv2D(out_channels, (3, 3), padding='SAME', 
                                    kernel_initializer='he_normal', name='fpn_p2')
        self.fpn_p3 = layers.Conv2D(out_channels, (3, 3), padding='SAME', 
                                    kernel_initializer='he_normal', name='fpn_p3')
        self.fpn_p4 = layers.Conv2D(out_channels, (3, 3), padding='SAME', 
                                    kernel_initializer='he_normal', name='fpn_p4')
        self.fpn_p5 = layers.Conv2D(out_channels, (3, 3), padding='SAME', 
                                    kernel_initializer='he_normal', name='fpn_p5')
        
        self.fpn_p6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')
        
            
    def call(self, inputs, training=True):
        C2, C3, C4, C5 = inputs
        
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + self.fpn_p5upsampled(P5)
        P3 = self.fpn_c3p3(C3) + self.fpn_p4upsampled(P4)
        P2 = self.fpn_c2p2(C2) + self.fpn_p3upsampled(P3)
        
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)
        
        return [P2, P3, P4, P5, P6]
        
    def compute_output_shape(self, input_shape):
        C2_shape, C3_shape, C4_shape, C5_shape = input_shape
        
        C2_shape, C3_shape, C4_shape, C5_shape = \
            C2_shape.as_list(), C3_shape.as_list(), C4_shape.as_list(), C5_shape.as_list()
        
        C6_shape = [C5_shape[0], (C5_shape[1] + 1) // 2, (C5_shape[2] + 1) // 2, self.out_channels]
        
        C2_shape[-1] = self.out_channels
        C3_shape[-1] = self.out_channels
        C4_shape[-1] = self.out_channels
        C5_shape[-1] = self.out_channels
        
        return [tf.TensorShape(C2_shape),
                tf.TensorShape(C3_shape),
                tf.TensorShape(C4_shape),
                tf.TensorShape(C5_shape),
                tf.TensorShape(C6_shape)]

if __name__ == '__main__':
    
    C2 = tf.random.normal((2, 256, 256,  256))
    C3 = tf.random.normal((2, 128, 128,  512))
    C4 = tf.random.normal((2,  64,  64, 1024))
    C5 = tf.random.normal((2,  32,  32, 2048))
    
    fpn = FPN()
    
    P2, P3, P4, P5, P6 = fpn([C2, C3, C4, C5])
    
    print('P2 shape:', P2.shape.as_list())
    print('P3 shape:', P3.shape.as_list())
    print('P4 shape:', P4.shape.as_list())
    print('P5 shape:', P5.shape.as_list())
    print('P6 shape:', P6.shape.as_list())