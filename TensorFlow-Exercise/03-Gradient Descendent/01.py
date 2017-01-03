import tensorflow as tf
import matplotlib.pyplot as plt

X = [1.,2.,3.]
Y = [1.,2.,3.]
m = len(X)
W = tf.placeholder(tf.float32)

W_val=[]
cost_val=[]

hypothesis = tf.mul(X,W)
cost = tf.reduce_mean(tf.pow(hypothesis-Y,2))

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(-30,50):
    print(i,sess.run(cost,feed_dict={W:i}))
    W_val.append(i)
    cost_val.append(sess.run(cost,feed_dict={W:i}))

plt.plot(W_val,cost_val,'ro')  #ro显示为点，无此参数显示为线
plt.xlabel('W_val')
plt.ylabel('cost_val')
plt.show()


