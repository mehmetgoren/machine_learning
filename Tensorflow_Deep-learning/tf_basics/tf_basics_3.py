import numpy as np
import tensorflow as tf


np.random.seed(101)
tf.set_random_seed(101)


A = np.random.uniform(0,100,(5,5))

B = np.random.uniform(0,100,(5,1))#A nın kolon sayısı, b' ni,n row sayısına eşit olmalı

m = tf.matmul(A,B)#çıkan matri

with tf.Session() as sess:
     matrix = sess.run(m)
     sess.close()

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)

add_op = a+b
mul_op =a*b

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a: A, b:B})
    mul_result = sess.run(mul_op, feed_dict={a: A, b:B})#buradaki değerlerin çarpımı. gerçek matmul değil.
