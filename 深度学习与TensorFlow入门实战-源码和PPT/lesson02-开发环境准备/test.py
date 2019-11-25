import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf



a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)

print('GPU:', tf.test.is_gpu_available())