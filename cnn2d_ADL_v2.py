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
    kernel_size=[3, 1],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# # pooling layer 1
# pool1 = tf.layers.max_pooling2d(
#     inputs=conv1,
#     pool_size=[1, 2],
#     strides=[1, 2],
#     padding='same'
# )
# print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
# conv2 = tf.layers.conv2d(
#     inputs=conv1,
#     filters=64,
#     kernel_size=[2, 16],
#     strides=[1, 1],
#     padding='valid',
#     activation=tf.nn.relu
# )
# print("### convolution layer 2 shape: ", conv2.shape, " ###")

# # pooling layer 2
# pool2 = tf.layers.max_pooling2d(
#     inputs=conv2,
#     pool_size=[1, 2],
#     strides=[1, 2],
#     padding='same'
# )
# print("### pooling layer 2 shape: ", pool2.shape, " ###")

# convolution layer 3
conv3 = tf.layers.conv2d(
    inputs=conv1,
    filters=128,
    kernel_size=[1, 2],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 3 shape: ", conv3.shape, " ###")

# pooling layer 3
pool3 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[1, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 3 shape: ", pool3.shape, " ###")

# convolution layer 4
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=256,
    kernel_size=[1, 2],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 4 shape: ", conv4.shape, " ###")

# pooling layer 4
pool4 = tf.layers.max_pooling2d(
    inputs=conv4,
    pool_size=[1, 2],
    strides=[1, 2],
    padding='same'
)
print("### pooling layer 4 shape: ", pool4.shape, " ###")


shape = pool4.get_shape().as_list()
flat = tf.reshape(pool4, [-1, shape[1] * shape[2] * shape[3]])
print("### flat shape: ", flat.shape, " ###")

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=1000,
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


# Epoch:  101  Training Loss:  7.075082  Training Accuracy:  0.96483773
# Epoch:  102  Training Loss:  6.155076  Training Accuracy:  0.9714065
# Epoch:  103  Training Loss:  6.0230765  Training Accuracy:  0.97024727
# Epoch:  104  Training Loss:  4.7655134  Training Accuracy:  0.9764297
# Epoch:  105  Training Loss:  4.885109  Training Accuracy:  0.97797525
# Epoch:  106  Training Loss:  3.569831  Training Accuracy:  0.98106647
# Epoch:  107  Training Loss:  3.876173  Training Accuracy:  0.9841576
# Epoch:  108  Training Loss:  3.0217977  Training Accuracy:  0.9868624
# Epoch:  109  Training Loss:  3.1276848  Training Accuracy:  0.98608965
# Epoch:  110  Training Loss:  2.004346  Training Accuracy:  0.98879445
# Testing Accuracy: 0.74568963
# [[107   1   1   0   4   0   2]
#  [  0  65   3   1   6   1  67]
#  [  2   2  71   2   2   2   4]
#  [  0   3   3 124   6  12   2]
#  [  3   5   5   6 110  18  14]
#  [  0   5   1  24  10 120   2]
#  [  5  45   5   4  20   1 264]]
#
# Epoch:  111  Training Loss:  2.3203845  Training Accuracy:  0.9911128
# Epoch:  112  Training Loss:  1.873558  Training Accuracy:  0.9911128
# Epoch:  113  Training Loss:  1.5076559  Training Accuracy:  0.99149925
# Epoch:  114  Training Loss:  1.1773285  Training Accuracy:  0.9934312
# Epoch:  115  Training Loss:  1.0075154  Training Accuracy:  0.9938176
# Epoch:  116  Training Loss:  1.1456351  Training Accuracy:  0.9934312
# Epoch:  117  Training Loss:  0.8205775  Training Accuracy:  0.9930448
# Epoch:  118  Training Loss:  1.3033819  Training Accuracy:  0.9938176
# Epoch:  119  Training Loss:  0.5528083  Training Accuracy:  0.9911128
# Epoch:  120  Training Loss:  0.6316587  Training Accuracy:  0.9930448
# Testing Accuracy: 0.75
# [[105   2   0   0   5   0   3]
#  [  0  68   1   1   4   1  68]
#  [  2   2  69   3   4   1   4]
#  [  0   2   2 128   4  12   2]
#  [  2   5   2   9 116  14  13]
#  [  1   2   2  23  11 119   4]
#  [  2  51   4   4  16   0 267]]

