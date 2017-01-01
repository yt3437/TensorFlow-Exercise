import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(1,100))
y_data = np.dot([0.1],x_data)+0.3

# 构造一个线性方程
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))
hypothesis = W*x_data +b

# 最小化方差
cost = tf.reduce_mean(tf.square(hypothesis-y_data))
a=tf.Variable(0.5)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 初始化参数
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(4001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))
