import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy
import datetime

import tensorflow as tf
sess = tf.InteractiveSession()

# image number to output
BATCH_SIZE = 50
IMAGE_TO_DISPLAY = 10

# Show the time
pre=datetime.datetime.today()
print pre

labeled_images = pd.read_csv('/Users/mac/Desktop/programme/Python/dlworking/train.csv')

images = labeled_images.iloc[:,1:]
images = images.astype(np.float)
labels = labeled_images.iloc[:,:1]
labels = labels.astype(np.float)

# change '0' into '[1,0,0,0,0,0,0,0,0,0]'
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels_flat = labeled_images.iloc[:,0].values
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

# shuffle the data
nums=labels_flat.shape[0]
perm = np.arange(nums)
np.random.shuffle(perm)
for i in range(0,42000):
    images.values[i] = images.values[perm[i]]
    labels[i] = labels[perm[i]]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# serve data by batches
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
print num_examples
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
       
    if index_in_epoch > num_examples:
        # shuffle the data
        nums=labels_flat.shape[0]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        for i in range(0,num_examples):
            train_images.values[i] = train_images.values[perm[i]]
            train_labels[i] = train_labels[perm[i]]
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

W_conv1 = weight_variable([7, 7, 1, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([7, 7, 64, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 256, 2048])
b_fc1 = bias_variable([2048])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(y,1)
sess.run(tf.global_variables_initializer())

for i in range(0,30000):
    batch_x, batch_y = next_batch(BATCH_SIZE)
    if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: test_images.values, y_: test_labels,
                                                  keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_images.values, y_: test_labels, keep_prob: 1.0}))

now=datetime.datetime.today()
print now
print now-pre

# read test data from CSV file 
predict_images = pd.read_csv('/Users/mac/Desktop/programme/Python/dlworking/test.csv').values
predict_images = predict_images.astype(np.float)

print('predict_images({0[0]},{0[1]})'.format(predict_images.shape))

# using batches is more resource efficient
predicted_lables = np.zeros(predict_images.shape[0])
for i in range(0,predict_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: predict_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})
np.savetxt('/Users/mac/Desktop/programme/Python/dlworking/testing_answer1.csv', 
           np.c_[range(1,len(predict_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

now=datetime.datetime.today()
print now
print now-pre

