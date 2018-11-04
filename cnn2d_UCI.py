import tensorflow as tf
import numpy as np
from sklearn import metrics

print("### Process1 --- data load ###")
train_x = np.load('UCI_data/np_2d/np_train_x_2d.npy')
train_y = np.load('UCI_data/np_2d/np_train_y_2d.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_2d.npy')
test_y = np.load('UCI_data/np_2d/np_test_y_2d.npy')
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### Process2 --- data spilt ###")

# define
seg_height = 9
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
    strides=[3, 2],
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
    strides=[3, 2],
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
    strides=[3, 2],
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
            batch_x = train_x[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            _, c = session.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            # cost_history = np.append(cost_history, c)
        print("Epoch: ", epoch+1, " Training Loss: ", c,
              " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        if (epoch+1) % 10 == 0:
            print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
            pred_y = session.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y,)
            print(cm, '\n')


# Epoch:  71  Training Loss:  0.6382424  Training Accuracy:  0.9541621
# Epoch:  72  Training Loss:  2.3883731  Training Accuracy:  0.9457291
# Epoch:  73  Training Loss:  0.4843847  Training Accuracy:  0.9570185
# Epoch:  74  Training Loss:  0.9577097  Training Accuracy:  0.9562024
# Epoch:  75  Training Loss:  0.93285394  Training Accuracy:  0.95130575
# Epoch:  76  Training Loss:  0.40557405  Training Accuracy:  0.94708925
# Epoch:  77  Training Loss:  10.495555  Training Accuracy:  0.92138195
# Epoch:  78  Training Loss:  0.67067504  Training Accuracy:  0.9400163
# Epoch:  79  Training Loss:  4.0036383  Training Accuracy:  0.9526659
# Epoch:  80  Training Loss:  1.1851147  Training Accuracy:  0.9548422
# Testing Accuracy: 0.9144893
# [[461  13  22   0   0   0]
#  [  4 429  38   0   0   0]
#  [ 10   2 408   0   0   0]
#  [  0   7   0 388  96   0]
#  [  0   1   0  61 470   0]
#  [  0   0   0   0   0 537]]
#
# Epoch:  81  Training Loss:  1.3421048  Training Accuracy:  0.9523939
# Epoch:  82  Training Loss:  0.45150667  Training Accuracy:  0.9568825
# Epoch:  83  Training Loss:  5.5236573  Training Accuracy:  0.9389282
# Epoch:  84  Training Loss:  0.3815871  Training Accuracy:  0.95973885
# Epoch:  85  Training Loss:  0.3512088  Training Accuracy:  0.9434168
# Epoch:  86  Training Loss:  5.634105  Training Accuracy:  0.9349837
# Epoch:  87  Training Loss:  0.4791569  Training Accuracy:  0.94178456
# Epoch:  88  Training Loss:  0.58426017  Training Accuracy:  0.9460011
# Epoch:  89  Training Loss:  0.7302053  Training Accuracy:  0.94640917
# Epoch:  90  Training Loss:  0.34810036  Training Accuracy:  0.9556583
# Testing Accuracy: 0.89888024
# [[460  13  23   0   0   0]
#  [ 28 409  34   0   0   0]
#  [ 15   3 402   0   0   0]
#  [  0  11   1 366 113   0]
#  [  0   2   0  64 466   0]
#  [  0   0   0   0   0 537]]
#
# Epoch:  91  Training Loss:  0.3713098  Training Accuracy:  0.960827
# Epoch:  92  Training Loss:  0.73379207  Training Accuracy:  0.94817734
# Epoch:  93  Training Loss:  0.30592585  Training Accuracy:  0.95742655
# Epoch:  94  Training Loss:  0.33291447  Training Accuracy:  0.95593035
# Epoch:  95  Training Loss:  0.34828228  Training Accuracy:  0.96055496
# Epoch:  96  Training Loss:  0.26480007  Training Accuracy:  0.95905876
# Epoch:  97  Training Loss:  0.3033659  Training Accuracy:  0.9510337
# Epoch:  98  Training Loss:  0.35549858  Training Accuracy:  0.960963
# Epoch:  99  Training Loss:  0.26124662  Training Accuracy:  0.9502176
# Epoch:  100  Training Loss:  0.28566658  Training Accuracy:  0.96055496
# Testing Accuracy: 0.8985409
# [[461  10  23   0   2   0]
#  [ 10 419  42   0   0   0]
#  [ 13   3 404   0   0   0]
#  [  0  11   0 364 116   0]
#  [  0   2   0  69 461   0]
#  [  0   0   0   0   0 537]]

