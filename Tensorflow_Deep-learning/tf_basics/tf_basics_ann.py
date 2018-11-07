import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x,W)#katsayı çarpması
z = tf.add(xW,b)#model

a = tf.sigmoid(z)#activation function

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)    
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})
    
    



    
#Simple Linear Regression Example
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

#plt.plot(x_data, y_label,"*")
#plt.show()


m = tf.Variable(.44)
b = tf.Variable(.87)
#y=mx+b

error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x+b
    error += (y-y_hat)**2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 1000
    for j in range(training_steps):
        sess.run(train)
    final_slope,final_intercept = sess.run([m,b])#intercept dediği bias
    
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test+final_intercept
plt.plot(x_test, y_pred_plot,"r")
plt.plot(x_data, y_label,"*")
plt.show()