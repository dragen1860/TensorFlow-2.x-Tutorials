import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)

print('GPU:', tf.test.is_gpu_available())