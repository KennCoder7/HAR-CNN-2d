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
Y = tf.placeholder(tf.float32, (None, num_labels))
print("### Process3 --- define ###")

# convolution layer 1
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=32,
    kernel_size=[3, 2],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[3, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3, 2],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[3, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")

# convolution layer 3
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 2],
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

# 2018/11/2
# Epoch:  506  Training Loss:  0.6057888  Training Accuracy:  0.9947644
# Epoch:  507  Training Loss:  0.6810966  Training Accuracy:  0.99513835
# Epoch:  508  Training Loss:  0.6395975  Training Accuracy:  0.99663424
# Epoch:  509  Training Loss:  0.6074306  Training Accuracy:  0.99663424
# Epoch:  510  Training Loss:  0.48039648  Training Accuracy:  0.9970082
# Testing Accuracy: 0.8119181
# [[104   1   0   1   3   2   3]
#  [  0  85   5   1   3   0  47]
#  [  0   1  74   1   6   2   5]
#  [  0   0   1 115   3   9   1]
#  [  4   4   7   2 116  10  11]
#  [  5   1   3  11   6 106   1]
#  [  1  26   0   0   9   2 276]]
#
# Epoch:  511  Training Loss:  0.63626075  Training Accuracy:  0.9962603
# Epoch:  512  Training Loss:  0.42530075  Training Accuracy:  0.99663424
# Epoch:  513  Training Loss:  0.5696478  Training Accuracy:  0.9943904
# Epoch:  514  Training Loss:  0.53919524  Training Accuracy:  0.9947644
# Epoch:  515  Training Loss:  0.4412272  Training Accuracy:  0.99513835
# Testing Accuracy: 0.80912477
# [[103   1   0   3   3   2   2]
#  [  0  89   7   1   1   0  43]
#  [  0   1  75   3   3   2   5]
#  [  0   0   1 114   2  11   1]
#  [  4   5   5   6 115   9  10]
#  [  5   0   3  13   7 104   1]
#  [  0  32   1   0   9   2 270]]
