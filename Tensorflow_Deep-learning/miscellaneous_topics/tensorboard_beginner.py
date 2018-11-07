"""
    TEnsorboard
"""

import tensorflow as tf

a = tf.add(1,2)
b=tf.add(3,4)
c = tf.multiply(a,b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output",sess.graph)
    print(sess.run(c))
    writer.close()