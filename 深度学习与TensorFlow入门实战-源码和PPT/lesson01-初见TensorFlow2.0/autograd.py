import tensorflow as tf 


x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)


with tf.GradientTape() as tape:
	tape.watch([a, b, c])
	y = a**2 * x + b * x + c


[dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
print(dy_da, dy_db, dy_dc)

