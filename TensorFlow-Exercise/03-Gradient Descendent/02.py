import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

W = tf.Variable(tf.random_uniform([1],-10.0,10.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X
cost = tf.reduce_mean(tf.pow(hypothesis-Y,2))

decent = W - tf.mul(0.1,tf.reduce_mean(tf.mul(hypothesis - Y,X)))
update = W.assign(decent)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sess.run(update,feed_dict={X:x_data,Y:y_data})
    print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W))


