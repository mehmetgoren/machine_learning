"""
    CNN
"""

"""
Variables

    For parameters to learn
    Values can be derived from training
    Initial values are required (often random)

Placeholders

    Allocated storage for data (such as for image pixel data during a feed)
    Initial values are not required (but can be set, see tf.placeholder_with_default)
    


    
    you use variables to hold and update parameters. Variables are in-memory buffers containing tensors.
    They must be explicitly initialized and can be saved to disk during and after training. 
    You can later restore saved values to exercise or analyze the model.
"""



import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#first_image = mnist.train.images[0].reshape(28,28)
#plt.imshow(first_image)


#placeholders
x = tf.placeholder(tf.float32, shape=[None,784])

#variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#create graph operations
y = tf.matmul(x,W)+b # it's a linear regression model formula.

#loss function
y_true = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.5)
train = optimizer.minimize(cross_entropy)

#create session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})#placeholder' lar burada
    
    #evaluate the model.
    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={x:mnist.test.images, y_true:mnist.test.labels})
    print(result)

