"""
    Deep Learning
"""

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("\n")


hi = tf.constant("Hi")


sess = tf.Session()
result = sess.run(hi)

print(result)

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print(sess.run(x+y))
    print(sess.run(x*y))
    
    
#başka bir kullanım
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x,y)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={x:12, y:42}))
    

import numpy as np
a = np.array([[5.0, 5.0]])
b = np.array([[2.0],[2.0]])


mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multiplication = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    print(sess.run(matrix_multiplication))