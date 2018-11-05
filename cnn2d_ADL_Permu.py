import tensorflow as tf
import numpy as np
from sklearn import metrics

data = np.load('ADL_data/np_2d_new/np_data_2d_permu.npy')
labels = np.load('ADL_data/np_2d_new/np_labels_2d_v1.npy')
print("### Process1 --- data load ###")
train_test_split = np.random.rand(len(data)) < 0.70
train_x = data[train_test_split]
train_y = labels[train_test_split]
test_x = data[~train_test_split]
test_y = labels[~train_test_split]
print("### train_x (data) shape: ", train_x.shape, " ###")
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### test_x (data) shape: ", test_x.shape, " ###")
print("### test_y (labels) shape: ", test_y.shape, " ###")
print("### Process2 --- data spilt ###")

# define
seg_height = 12
seg_len = 68
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
    filters=50,
    kernel_size=[3, 5],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 4],
    strides=[2, 4],
    padding='same'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=100,
    kernel_size=[2, 5],
    strides=[1, 1],
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 2],
    strides=[2, 2],
    padding='same'
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")


shape = pool2.get_shape().as_list()
flat = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=1000,
    activation=tf.nn.relu
)
# fc1 = tf.nn.dropout(fc1, keep_prob=0.8)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")


# softmax layer
sof = tf.layers.dense(
    inputs=fc1,
    units=num_labels,
    activation=tf.nn.softmax
)
print("### softmax layer shape: ", sof.shape, " ###")

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
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
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

# 2018/11/3 5c3*5-p2*4-10c2*5-p2*2-fc240
# Epoch:  805  Training Loss:  0.015575705  Training Accuracy:  0.9810976
# Epoch:  810  Training Loss:  0.023369756  Training Accuracy:  0.99004066
# Testing Accuracy: 0.7480094
# Epoch:  815  Training Loss:  0.01135402  Training Accuracy:  0.9971545
# Epoch:  820  Training Loss:  0.0110703865  Training Accuracy:  0.9971545
# Testing Accuracy: 0.74473065
# Epoch:  825  Training Loss:  0.010770791  Training Accuracy:  0.99552846
# Epoch:  830  Training Loss:  0.010189488  Training Accuracy:  0.9973577
# Testing Accuracy: 0.7480094

# 2018/11/3 5c3*5-p2*4-10c2*5-p2*2-fc240  fc(act tanh->relu)
# Epoch:  240  Training Loss:  0.041788254  Training Accuracy:  0.98662615
# Testing Accuracy: 0.78584903
# Epoch:  270  Training Loss:  0.5308273  Training Accuracy:  0.97892606
# Testing Accuracy: 0.78396225

# 2018/11/3 5c3*5-p2*4-10c2*5-p2*2-fc240  fc(act relu->None)
# Epoch:  235  Training Loss:  5.9071913  Training Accuracy:  0.83319855
# Epoch:  240  Training Loss:  5.674221  Training Accuracy:  0.8360291
# Testing Accuracy: 0.71266
# Epoch:  265  Training Loss:  5.831894  Training Accuracy:  0.7472705
# Epoch:  270  Training Loss:  4.770355  Training Accuracy:  0.83926404
# Testing Accuracy: 0.7178758
# Epoch:  485  Training Loss:  1.3425074  Training Accuracy:  0.88980997
# Epoch:  490  Training Loss:  1.7497208  Training Accuracy:  0.88839465
# Testing Accuracy: 0.7164533
# Testing Accuracy: 0.7855398
# Epoch:  325  Training Loss:  0.13839129  Training Accuracy:  0.9972045
# Epoch:  330  Training Loss:  0.10333597  Training Accuracy:  0.9972045
# Testing Accuracy: 0.78163165
# Epoch:  335  Training Loss:  0.09034504  Training Accuracy:  0.99680513
# Epoch:  340  Training Loss:  0.06625876  Training Accuracy:  0.995008
# Testing Accuracy: 0.7860283

# 2018/11/3 50c3*5-p2*4-100c2*5-p2*2-fc1000
# Epoch:  195  Training Loss:  0.16976179  Training Accuracy:  0.9877501
# Epoch:  200  Training Loss:  0.06683847  Training Accuracy:  0.9908126
# Testing Accuracy: 0.8168753
# [[202   0   1   2   4   3   5]
#  [  3 185   6   1  13   2  65]
#  [  4   3 124   3  16   8   4]
#  [  2   0   0 221   9  42   0]
#  [  4   6   9  19 228  15  15]
#  [  2   0   1  29  18 231   3]
#  [  9  21   6   1  36   5 571]]