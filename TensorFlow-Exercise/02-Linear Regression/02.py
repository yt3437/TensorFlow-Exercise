import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(1,100))
y_data = np.dot([0.1],x_data)+0.3

# 构造一个线性方程
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X= tf.placeholder(tf.float32)
Y= tf.placeholder(tf.float32)

hypothesis = W*x_data +b

# 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis-Y))
a=tf.Variable(0.5)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 初始化参数
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train,feed_dict={X:x_data,Y:y_data})
    if step % 20 == 0:
        print(sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

print(sess.run(hypothesis,feed_dict={X:5}))
print(sess.run(hypothesis,feed_dict={X:2.5}))
