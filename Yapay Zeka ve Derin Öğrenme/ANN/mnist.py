import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#thr formula is y = x*w+b,w = coeffişcient, b is bias

mnist = input_data.read_data_sets("data/MNIST", one_hot=True)#one_hot encoding 2 = [0,0,1,0,0,0,0,0,0,0]

x = tf.placeholder(tf.float32, [None, 784])   #Node resim limiti, 784=28x28 matrix' in vector hali.
y_true = tf.placeholder(tf.float32, [None, 10])#None= gelecek resim sayısı, 10 = one_hot' dan gewlecek 10 haneli dizi.


#x = [500, 784]#¹500 adet image
#w = [784, 10]. matrriz çaprma olması içib row, column count' lar
#x*w = [500, 10]

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#x*w+b = [500,10] + [10] matrix toplama.

#x*w+b genelede logits deneir
logits = tf.matmul(x,w) + b # matmul means matrix multiplication

#train esnasında w ve b değerleri güncellenecek. Bu eğtimden gelen değerler softmax aktivisyan fonksiyonundan geçirilerek 0,1 arasına sıkışacak.
#her resim için en büyük değper hangi elemandaysa ona göre resim tahmin edilecek.
y = tf.nn.softmax(logits)#0-1 arası 0 kötü 1 e yaklaştıkça iyi . örneğin 0.9 %90 demek


#şimdi de lost/cost function ile supervides yanımımı gösterelim ve taqhminin ne kadar doğru / yanlış olduğunu bulalım.
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)#

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))#supoervised. arg_max ise en çok skoru alan tahmin
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


optimize = tf.train.GradientDescentOptimizer(.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 256

def training_step(iterations):
    for j in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch, y_true:y_batch}
        sess.run(optimize, feed_dict=feed_dict_train)
        
        
def test_accuracy():
    feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Testing Accuricy:", acc)
    

training_step(10000)
test_accuracy()