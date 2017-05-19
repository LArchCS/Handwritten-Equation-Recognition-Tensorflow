import tensorflow as tf
import pickle
import random
import numpy as np

# variables
steps = 5000
batchSize = 100
convolution = (1, 1)
kennelSize = (2, 2)
maxPoll = (2, 2)
#Optimizer = AdamOptimizer
#https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
learningRate = 0.001
layer1Feature = 16
layer1Patch = 5, 5
layer2Feature = 32
layer2Patch = 5, 5
hiddenLayer = 100  # the more hiddenLayer number, the less general the model will perform
dropoffRate = 0.5  # reduce overfitting
layer3Feature = 64
layer3Patch = 5, 5

#read data
pfile = open("train1.pkl","rb")
data = pickle.load(pfile)

sess = tf.InteractiveSession()

# create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 40])
W = tf.Variable(tf.zeros([784, 40]))
b = tf.Variable(tf.zeros([40]))
y = tf.matmul(x, W) + b

#methods
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
def max_pool_1x1(x):
    return tf.nn.max_pool(x, ksize=[1, kennelSize[0], kennelSize[1], 1], strides=[1, 1, 1, 1], padding='SAME')

#First convolutional layer
W_conv1 = weight_variable([layer1Patch[0], layer1Patch[1], 1, layer1Feature])
b_conv1 = bias_variable([layer1Feature])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#Second convolutional layer
W_conv2 = weight_variable([layer2Patch[0], layer2Patch[1], layer1Feature, layer2Feature])
b_conv2 = bias_variable([layer2Feature])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD LAYER
W_conv3 = weight_variable([layer2Patch[0], layer3Patch[1], layer2Feature, layer3Feature])
b_conv3 = bias_variable([layer3Feature])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_1x1(h_conv3)

#densely connected layer
W_fc1 = weight_variable([7 * 7 * layer3Feature, hiddenLayer])  # hidden layer
b_fc1 = bias_variable([hiddenLayer])  # hidden layer

h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * layer3Feature])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
W_fc2 = weight_variable([hiddenLayer, 40])
b_fc2 = bias_variable([40])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# train and evaluate
for i in range(steps):
  batch = random.sample(data["images"], 50)
  batch_x = []
  batch_y = []
  for batchd in batch:
      batch_x.append(data["images"][batchd])
      batch_y.append(data["labels"][batchd])
  if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoffRate})

save_path = saver.save(sess, "model.ckpt")
print ("Model saved in file: ", save_path)

batch_x = []
batch_y = []
for batchd in data["images"]:
    batch_x.append(data["images"][batchd])
    batch_y.append(data["labels"][batchd])

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: batch_x , y_: batch_y, keep_prob: 1.0}))
