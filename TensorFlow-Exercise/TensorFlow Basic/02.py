import tensorflow as tf

a = tf.constant(3)
b = tf.constant(4)

c = a + b
sess = tf.Session()
print(sess.run(c))