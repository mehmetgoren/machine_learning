import tensorflow as tf

print(tf.__version__)


const = tf.constant(10)

matrix = tf.fill((4,4),10)

zeros = tf.zeros((4,4))

ones = tf.ones((4,4))

randoms = tf.random_normal((4,4), mean=0, stddev=1.0)

randu = tf.random_uniform((4,4))


ops = [const, matrix, zeros,ones,randoms,randu]

sess = tf.InteractiveSession()

for op in ops:
    print(op.eval())
    
    
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[10],[100]])
print(a.get_shape())


matmul = tf.matmul(a,b)
result = matmul.eval()

print(result)


sess.close()

