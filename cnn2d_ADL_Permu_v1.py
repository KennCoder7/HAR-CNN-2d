import tensorflow as tf
import numpy as np
from sklearn import metrics

data = np.load('ADL_data/np_2d_new/np_data_2d_permu_v1.npy')
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
seg_height = 18
seg_len = 68
num_channels = 1
num_labels = 7
batch_size = 100
learning_rate = 0.0001
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
    kernel_size=[5, 5],
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
    kernel_size=[4, 5],
    strides=[1, 1],
    padding='valid',
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
fc1 = tf.nn.dropout(fc1, keep_prob=0.8)
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


# 2018/11/5 50c3*5-p2*4-100c2*5-p2*2-fc1000
# Epoch:  385  Training Loss:  0.05417257  Training Accuracy:  0.9891063
# Epoch:  390  Training Loss:  0.004651963  Training Accuracy:  0.99172884
# Testing Accuracy: 0.7812202
# Epoch:  395  Training Loss:  0.31923997  Training Accuracy:  0.99556184
# Epoch:  400  Training Loss:  0.008859173  Training Accuracy:  0.99172884
# Testing Accuracy: 0.7831268
# [[179   2   1   2   5   3   2]
#  [  1 197   8   0  13   3  46]
#  [  1   8 131   7  11   4   8]
#  [  1   1   3 205   7  56   0]
#  [  3  14  14  21 204  33  17]
#  [  0   2   1  47   7 214   2]
#  [  3  47   2   6  30   1 525]]

# 2018/11/5 50c5*5v-p2*4s-100c4*5v-p2*2s-fc1000
# Epoch:  635  Training Loss:  0.014884807  Training Accuracy:  0.9997953
# Epoch:  640  Training Loss:  0.10860743  Training Accuracy:  0.9997953
# Testing Accuracy: 0.82895344
# Epoch:  645  Training Loss:  0.014204593  Training Accuracy:  0.9997953
# Epoch:  650  Training Loss:  0.019544793  Training Accuracy:  0.9997953
# Testing Accuracy: 0.8197326