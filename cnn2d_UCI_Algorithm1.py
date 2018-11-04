import tensorflow as tf
import numpy as np
from sklearn import metrics

print("### Process1 --- data load ###")
train_x = np.load('UCI_data/np_2d/np_train_x_algorithm1.npy')
train_y = np.load('UCI_data/np_2d/np_train_y_2d.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_algorithm1.npy')
test_y = np.load('UCI_data/np_2d/np_test_y_2d.npy')
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### Process2 --- data spilt ###")

# define
seg_height = 36
seg_len = 128
num_channels = 1
num_labels = 6
batch_size = 100
learning_rate = 0.001
num_epoches = 10000
num_batches = train_x.shape[0] // batch_size
print("### num_batch: ", num_batches, " ###")

training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder(tf.float32, (None, seg_height, seg_len, num_channels))
Y = tf.placeholder(tf.float32, (None, num_labels))
print("### Process3 --- define ###")

# convolution layer 1
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=10,
    kernel_size=[5, 5],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[4, 4],
    strides=[4, 4],
    padding='same'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=100,
    kernel_size=[5, 5],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 4],
    strides=[2, 4],
    padding='valid'
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")

shape = pool2.get_shape().as_list()
flat = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=120,
    activation=tf.nn.relu
)
# fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")

# # fully connected layer 2
# fc2 = tf.layers.dense(
#     inputs=fc1,
#     units=100,
#     activation=tf.nn.tanh
# )
# fc2 = tf.nn.dropout(fc2, keep_prob=0.8)
# print("### fully connected layer 2 shape: ", fc2.shape, " ###")

# softmax layer 3
sof = tf.layers.dense(
    inputs=fc1,
    units=num_labels,
    activation=tf.nn.softmax
)
print("### fully connected layer 3 shape: ", sof.shape, " ###")

y_ = sof
print("### prediction shape: ", y_.get_shape(), " ###")

loss = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoches):
        # cost_history = np.empty(shape=[0], dtype=float)
        for b in range(num_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            _, c = session.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            # cost_history = np.append(cost_history, c)
        if (epoch + 1) % 5 == 0:
            print("Epoch: ", epoch+1, " Training Loss: ", c,
                  " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        if (epoch + 1) % 10 == 0:
            print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        if (epoch + 1) % 100 == 0:
            pred_y = session.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y,)
            print(cm, '\n')


# 2018/11/3  5c5*5-p4*4-10c5*5-p2*2-fc100-fc100-sof AdamOptimizer
# Epoch:  10  Training Loss:  7.9259453  Training Accuracy:  0.9257345
# Testing Accuracy: 0.846963
# [[415   8  73   0   0   0]
#  [  6 406  59   0   0   0]
#  [  3   9 408   0   0   0]
#  [  1  23   1 351 111   4]
#  [  1  54   0  65 412   0]
#  [  0  12  15   1   0 509]]
# Epoch:  80  Training Loss:  0.556166  Training Accuracy:  0.960691
# Testing Accuracy: 0.9012555
# [[475   2  19   0   0   0]
#  [  6 429  34   1   0   1]
#  [ 15   3 401   0   0   1]
#  [  0  20   0 373  95   3]
#  [  1  25   0  50 456   0]
#  [  0  18   0   1   0 518]]
# Epoch:  100  Training Loss:  0.14081058  Training Accuracy:  0.96517956
# Testing Accuracy: 0.90668476
# [[466   6  24   0   0   0]
#  [  5 433  30   2   0   1]
#  [  6  11 403   0   0   0]
#  [  0  20   1 405  60   5]
#  [  1  20   0  68 442   1]
#  [  0   7   9   1   0 520]]

# 2018/11/3 5c5*5-p4*4-10c5*5-p2*4-fc120-sof AdamOptimizer
# Epoch:  10  Training Loss:  5.966345  Training Accuracy:  0.92451036
# Testing Accuracy: 0.8506956
# [[455  36   5   0   0   0]
#  [ 19 448   3   0   1   0]
#  [ 35  93 290   0   1   1]
#  [  1  18   0 313 158   1]
#  [  0   7   0  48 477   0]
#  [  0  19   7   0   0 511]]
# Epoch:  80  Training Loss:  1.87744  Training Accuracy:  0.9510337
# Testing Accuracy: 0.88870037
# [[457  14  25   0   0   0]
#  [ 20 436  13   0   0   2]
#  [  6  15 399   0   0   0]
#  [  0  24   0 339 127   1]
#  [  0  16   0  45 471   0]
#  [  0  19   3   0   0 515]]
# Epoch:  100  Training Loss:  1.8083413  Training Accuracy:  0.95280194
# Testing Accuracy: 0.87750256
# [[440  34  21   0   0   1]
#  [  0 462   3   0   0   6]
#  [ 11  36 373   0   0   0]
#  [  0  23   0 340 125   3]
#  [  0  24   0  41 466   1]
#  [  1  23   3   0   0 510]]

# 2018/11/3 10c5*5-p4*4-100c5*5-p2*4-fc120-sof AdamOptimizer
# Epoch:  10  Training Loss:  3.9821105  Training Accuracy:  0.94096845
# Testing Accuracy: 0.8951476
# [[491   1   4   0   0   0]
#  [ 22 427  22   0   0   0]
#  [  3   4 413   0   0   0]
#  [  2  19   0 343 122   5]
#  [  2   9   0  58 463   0]
#  [  0  27   0   0   0 510]]
# Epoch:  20  Training Loss:  2.9272904  Training Accuracy:  0.9466812
# Testing Accuracy: 0.9083814
# Epoch:  30  Training Loss:  2.2685628  Training Accuracy:  0.9494015
# Testing Accuracy: 0.9165253
# [[483   2  11   0   0   0]
#  [  3 464   4   0   0   0]
#  [  1   2 417   0   0   0]
#  [  0  20   0 325 141   5]
#  [  2   4   0  29 497   0]
#  [  0  27   0   0   0 510]]
# Epoch:  40  Training Loss:  1.7567424  Training Accuracy:  0.9488574
# Testing Accuracy: 0.9175433
# Epoch:  60  Training Loss:  0.84365696  Training Accuracy:  0.9508977
# Testing Accuracy: 0.912114
# Epoch:  80  Training Loss:  0.72701436  Training Accuracy:  0.960963
# Testing Accuracy: 0.9070241
# Epoch:  100  Training Loss:  0.58589244  Training Accuracy:  0.9576986
# Testing Accuracy: 0.90872073


# 2018/11/3 10c5*5-p4*4-100c5*5-p2*4-fc120-sof AdamOptimizer  fc(tanh->relu)  Xdrop
# Epoch:  35  Training Loss:  2.2881691  Training Accuracy:  0.9357998
# Epoch:  40  Training Loss:  0.65968466  Training Accuracy:  0.94260067
# Testing Accuracy: 0.90770274
# Epoch:  45  Training Loss:  1.4417223  Training Accuracy:  0.94178456
# Epoch:  50  Training Loss:  1.0055159  Training Accuracy:  0.9383841
# Testing Accuracy: 0.8982016

# Epoch:  295  Training Loss:  0.060113505  Training Accuracy:  0.9908868
# Epoch:  300  Training Loss:  0.008091084  Training Accuracy:  0.9872144
# Testing Accuracy: 0.9175433
# [[479   2  15   0   0   0]
#  [ 12 455   4   0   0   0]
#  [ 21   2 397   0   0   0]
#  [  0  10   0 407  74   0]
#  [  0   5   0  97 430   0]
#  [  0   1   0   0   0 536]]
#
# Epoch:  305  Training Loss:  0.0593417  Training Accuracy:  0.9874864
# Epoch:  310  Training Loss:  0.014059154  Training Accuracy:  0.99115884
# Testing Accuracy: 0.91890055
# Epoch:  315  Training Loss:  0.0044265296  Training Accuracy:  0.99115884
# Epoch:  320  Training Loss:  0.011148976  Training Accuracy:  0.99265504
# Testing Accuracy: 0.9239905
# Epoch:  325  Training Loss:  0.026298754  Training Accuracy:  0.9866703
# Epoch:  330  Training Loss:  0.004654552  Training Accuracy:  0.9933351
# Testing Accuracy: 0.92840177