import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.FATAL)


n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1+n2

sess = tf.InteractiveSession()

print(n3.eval())




g = tf.Graph()


#placeholder lar optimize edilecek parameytrelerdir ve içleri boştur.
tensor = tf.random_uniform((4,4),0,1)

var = tf.Variable(initial_value=tensor)

init = tf.global_variables_initializer()
sess.run(init)

print(var.eval())


ph = tf.placeholder(tf.float64, shape=(None,4))


sess.close()