import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True, reshape=False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_true = tf.placeholder(tf.float32, [None, 10])

filter1 = 16
filter2 = 32

weight_1 = tf.Variable(tf.truncated_normal([5, 5, 1, filter1], stddev=0.1))#truncated_normal rastgele sayılar var başta. stddev' de uçtaki verileri budamak için.
bias_1 = tf.Variable(tf.constant(0.1, shape=[filter1]))
weight_2 = tf.Variable(tf.truncated_normal([5, 5, filter1, filter2], stddev=0.1))
bias_2 = tf.Variable(tf.constant(0.1, shape=[filter2]))

weight_3 = tf.Variable(tf.truncated_normal([7*7*filter2, 256], stddev=0.1))
bias_3 = tf.Variable(tf.constant(0.1, shape=[256]))
weight_4 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
bias_4 = tf.Variable(tf.constant(0.1, shape=[10]))

# input = [28, 28, 1]                     strides=[batch, x, y, depth]
y1 = tf.nn.relu(tf.nn.conv2d(x, weight_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1) # output = [28, 28, 16]
y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output = [14, 14, 16]
y2 = tf.nn.relu(tf.nn.conv2d(y1, weight_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2) # output = [14, 14, 32]
y2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output = [7, 7, 32]

flattened = tf.reshape(y2, shape=[-1, 7 * 7 * filter2])

y3 = tf.nn.relu(tf.matmul(flattened, weight_3) + bias_3)
logits = tf.matmul(y3, weight_4) + bias_4
y4 = tf.nn.softmax(logits)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
loss_graph = []
def training_step (iterations):
    for i in range (iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)

        loss_graph.append(train_loss)

        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iterations:', i, 'Training accuracy:', train_acc, 'Training loss:', train_loss)


feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
def test_accuracy ():
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])




training_step(10000)
test_accuracy()
plot_example_errors()

plt.plot(loss_graph, 'k-')
plt.title('Loss Grafiği')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()