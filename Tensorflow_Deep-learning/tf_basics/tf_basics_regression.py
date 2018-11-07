import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


x_data = np.linspace(.0,10.0,1000000)
noise = np.random.randn(len(x_data))

#y = mx+b;

y_true = (.5*x_data) + 5 + noise
x_df = pd.DataFrame(data=x_data, columns=["X Data"])
y_df = pd.DataFrame(data=y_true, columns=["y"])

my_data = pd.concat([x_df, y_df],axis=1)



batch_size = 8
m = tf.Variable(.81)#slope
b = tf.Variable(.17)#intercept


xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = m*xph+b

error = tf.reduce_sum(tf.square(yph-y_model))#lost function

optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for j in range(batches):
        rand_index = np.random.randint(len(x_data), size=batch_size)
        feed = {xph:x_data[rand_index], yph:y_true[rand_index]}
        sess.run(train, feed_dict=feed)
    slope,bias = sess.run([m,b])

y_hat=x_data*slope+bias
my_data.sample(250).plot(kind="scatter", x="X Data", y="y")
plt.plot(x_data,y_hat,"r")
plt.show()