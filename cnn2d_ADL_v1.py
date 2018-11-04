import tensorflow as tf
import numpy as np
from sklearn import metrics

data = np.load('ADL_data/np_2d/np_data_2d_v1.npy')
labels = np.load('ADL_data/np_2d/np_labels_2d_v1.npy')
print("### Process1 --- data load ###")
train_test_split = np.random.rand(len(data)) < 0.70
train_x = data[train_test_split]
train_y = labels[train_test_split]
test_x = data[~train_test_split]
test_y = labels[~train_test_split]
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### Process2 --- data spilt ###")

# define
seg_height = 3
seg_len = 128
num_channels = 1
num_labels = 7
batch_size = 100
learning_rate = 0.001
num_epoches = 10000
num_batches = train_x.shape[0] // batch_size
print("### num_batch: ", num_batches, " ###")

training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder(tf.float32, (None, seg_height, seg_len, num_channels))
Y = tf.placeholder(tf.float32, (None, 7))
print("### Process3 --- define ###")

# convolution layer 1
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=64,
    kernel_size=[2, 2],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=128,
    kernel_size=[2, 2],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[1, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")

# convolution layer 3
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=[1, 2],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 3 shape: ", conv3.shape, " ###")

# pooling layer 3
pool3 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[3, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 3 shape: ", pool3.shape, " ###")

shape = pool3.get_shape().as_list()
flat = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=100,
    activation=tf.nn.tanh
)
fc1 = tf.nn.dropout(fc1, keep_prob=0.8)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")

# fully connected layer 1
fc2 = tf.layers.dense(
    inputs=fc1,
    units=100,
    activation=tf.nn.tanh
)
fc2 = tf.nn.dropout(fc2, keep_prob=0.8)
print("### fully connected layer 2 shape: ", fc2.shape, " ###")

# fully connected layer 3
fc3 = tf.layers.dense(
    inputs=fc2,
    units=num_labels,
    activation=tf.nn.softmax
)
print("### fully connected layer 3 shape: ", fc3.shape, " ###")

y_ = fc3
print("### prediction shape: ", y_.get_shape(), " ###")

loss = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoches):
        # cost_history = np.empty(shape=[0], dtype=float)
        for b in range(num_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            # cost_history = np.append(cost_history, c)
        print("Epoch: ", epoch+1, " Training Loss: ", c,
              " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        if (epoch+1) % 10 == 0:
            print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
            pred_y = session.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y,)
            print(cm, '\n')


# Epoch:  141  Training Loss:  1.9767199  Training Accuracy:  0.9901216
# Epoch:  142  Training Loss:  1.4983237  Training Accuracy:  0.9912614
# Epoch:  143  Training Loss:  1.2675704  Training Accuracy:  0.99354106
# Epoch:  144  Training Loss:  1.1754577  Training Accuracy:  0.9950608
# Epoch:  145  Training Loss:  1.1678376  Training Accuracy:  0.99354106
# Epoch:  146  Training Loss:  0.6718466  Training Accuracy:  0.99696046
# Epoch:  147  Training Loss:  0.88937  Training Accuracy:  0.993921
# Epoch:  148  Training Loss:  0.48354268  Training Accuracy:  0.99658054
# Epoch:  149  Training Loss:  0.73318017  Training Accuracy:  0.9946808
# Epoch:  150  Training Loss:  0.4538696  Training Accuracy:  0.99658054
# Testing Accuracy: 0.7822581
# [[ 92   0   0   0   1   2   1]
#  [  0  85   2   0   5   1  54]
#  [  1   2  58   1   5   4   3]
#  [  3   1   1 126  15  16   3]
#  [  6   5   3   7 103  13  14]
#  [  4   2   1  12  12 113   0]
#  [  2  30   1   4  14   0 288]]
#
# Epoch:  151  Training Loss:  0.4679497  Training Accuracy:  0.9946808
# Epoch:  152  Training Loss:  0.4485076  Training Accuracy:  0.99696046
# Epoch:  153  Training Loss:  0.3183949  Training Accuracy:  0.9943009
# Epoch:  154  Training Loss:  0.22222877  Training Accuracy:  0.9931611
# Epoch:  155  Training Loss:  0.39284918  Training Accuracy:  0.99354106
# Epoch:  156  Training Loss:  0.3349923  Training Accuracy:  0.99696046
# Epoch:  157  Training Loss:  0.3481661  Training Accuracy:  0.99696046
# Epoch:  158  Training Loss:  0.22500786  Training Accuracy:  0.99772036
# Epoch:  159  Training Loss:  0.2830991  Training Accuracy:  0.99734044
# Epoch:  160  Training Loss:  0.19624823  Training Accuracy:  0.9962006
# Testing Accuracy: 0.7822581
# [[ 91   1   0   0   1   2   1]
#  [  0  83   3   1   4   1  55]
#  [  0   2  58   1   7   3   3]
#  [  1   1   1 136   8  15   3]
#  [  5   5   2   9 102  16  12]
#  [  3   1   2  13   9 114   2]
#  [  3  27   2   3  15   1 288]]