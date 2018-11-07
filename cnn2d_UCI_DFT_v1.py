import tensorflow as tf
import numpy as np
from sklearn import metrics
import time

initial_time = time.time()
print("### Process1 --- initial ###")
train_x = np.load('UCI_data/np_2d/np_train_x_dft_v2.npy')
train_y = np.load('UCI_data/np_2d/np_train_y_2d.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_dft_v2.npy')
test_y = np.load('UCI_data/np_2d/np_test_y_2d.npy')
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### Process2 --- data load ###")

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
    strides=[1, 2],
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.average_pooling2d(
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
pool2 = tf.layers.average_pooling2d(
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
    units=120,
    activation=tf.nn.relu
)
fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")

# # fully connected layer 2
# fc2 = tf.layers.dense(
#     inputs=fc1,
#     units=100,
#     activation=tf.nn.tanh
# )
# fc2 = tf.nn.dropout(fc2, keep_prob=0.8)
# print("### fully connected layer 2 shape: ", fc2.shape, " ###")

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
        if (epoch + 1) % 50 == 0:
            print("# Epoch: ", epoch+1, " Training Loss: ", c,
                  " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        if (epoch + 1) % 100 == 0:
            print("# Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        # if (epoch + 1) % 100 == 0:
        #     pred_y = session.run(tf.argmax(y_, 1), feed_dict={X: test_x})
        #     cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y,)
        #     print(cm, '\n')


# 2018/11/5 10c5*5-p4*4-100c5*5-p2*4-fc120-sof AdamOptimizer
# Epoch:  455  Training Loss:  0.0153962895  Training Accuracy:  0.9855822
# Epoch:  460  Training Loss:  0.07526402  Training Accuracy:  0.9893907
# Testing Accuracy: 0.9093994
# Epoch:  465  Training Loss:  0.017043516  Training Accuracy:  0.9878945
# Epoch:  470  Training Loss:  0.48552203  Training Accuracy:  0.9851741
# Testing Accuracy: 0.90091616

# 2018/11/5 10c5*5-p4*4-100c5*5-p2*4-fc120-sof AdamOptimizer  max_pooling-->average_pooling
# Epoch:  645  Training Loss:  0.16690484  Training Accuracy:  0.98843855
# Epoch:  650  Training Loss:  0.43511367  Training Accuracy:  0.98626226
# Testing Accuracy: 0.90872073
# Epoch:  655  Training Loss:  0.2106536  Training Accuracy:  0.98762244
# Epoch:  660  Training Loss:  0.07344559  Training Accuracy:  0.98775846
# Testing Accuracy: 0.9192399

# 2018/11/5 data v1-->v2 lr/10
# Epoch:  100  Training Loss:  0.5538861  Training Accuracy:  0.9661317
# Testing Accuracy: 0.892433
# Epoch:  200  Training Loss:  0.40065053  Training Accuracy:  0.9863983
# Testing Accuracy: 0.9233118
# Epoch:  300  Training Loss:  0.6553461  Training Accuracy:  0.9960555
# Testing Accuracy: 0.9192399
# Epoch:  400  Training Loss:  0.00082163897  Training Accuracy:  0.9978237
# Testing Accuracy: 0.9243298
# Epoch:  500  Training Loss:  2.2411463e-05  Training Accuracy:  0.9990479
# Testing Accuracy: 0.92602646