# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 下载四个mnist
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])  # 784=28*28
w = tf.Variable(tf.zeros([784, 10]))  # 0-9共10个类
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)  # 激励函数softmax
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # cross_entropy作为loss function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 学习速率0.5优化目标cross_entropy
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次随机抽取100条样本构成一个mini_batch
    train_step.run({x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1),
                              tf.argmax(y_, 1))  # tf.argmax(y,1)求概率最大的那个类argmax(y_,1)找样本真实数字类别tf.equaal判断预测类别是否是正确的类别
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # boolֵ转换为float32
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# tensorboard --logdir=mnist_logs
