import tensorflow as tf
import matplotlib.pyplot as plt

img = plt.imread('/home/llong/Downloads/number.jpg') /255.
img2 = plt.imread('/home/llong/Downloads/number2.jpg') /255.
img = tf.convert_to_tensor(img, dtype=tf.float32)
img = tf.expand_dims(img, axis=0)
img = tf.image.resize(img, (1000,1000))
img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
img2 = tf.expand_dims(img2, axis=0)
img2 = tf.image.resize(img2, (1000,1000))

img = tf.concat([img, img2], axis=0)
print('img:', img.shape)

a = tf.image.crop_and_resize(img, [[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5]], [0, 1], crop_size=(500, 500))
print('a:', a.shape)

plt.subplot(2,2,1)
plt.imshow(img[0])
plt.subplot(2,2,2)
plt.imshow(img[1])
plt.subplot(2,2,3)
plt.imshow(a[0])
plt.subplot(2,2,4)
plt.imshow(a[1])
plt.show()
