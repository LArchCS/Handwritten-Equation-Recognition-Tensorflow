import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import pickle

# variables
steps = 5000
batchSize = 124
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

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']
brules = {}
for i in range(0,len(sy)):
    brules[i] = sy[i]

def predictint():
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 40]))
    b = tf.Variable(tf.zeros([40]))

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

    # First Convolutional Layer
    W_conv1 = weight_variable([layer1Patch[0], layer1Patch[1], 1, layer1Feature])
    b_conv1 = bias_variable([layer1Feature])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([layer2Patch[0], layer2Patch[1], layer1Feature, layer2Feature])
    b_conv2 = bias_variable([layer2Feature])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # THIRD LAYER
    W_conv3 = weight_variable([layer2Patch[0], layer3Patch[1], layer2Feature, layer3Feature])
    b_conv3 = bias_variable([layer3Feature])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_1x1(h_conv3)

    # Densely Connected Layer
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

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.getcwd()+"/model.ckpt")
        #print ("Model restored.")
        nf = open("result.txt", 'w')
        tfile = open("test.pkl","rb")
        nnfile = open("undesired.txt",'w')
        data = pickle.load(tfile)

        number = 0
        hit = 0
        for f in data["images"]:
            # print (fn)
            prediction=tf.argmax(y_conv,1)
            predint = prediction.eval(feed_dict={x: [data["images"][f]],keep_prob: 1.0}, session=sess)
            # print f
            # print brules[predint[0]]
            nf.write("%s\t%s\n" %(f,brules[predint[0]]))
            ins = f.split('.')[0].split('_')
            label = ins[3]
            if ins[3] == "o":
                label = "0"
            if ins[3] == "frac" or ins[3] == "bar":
                label = "-"
            if ins[3] == "mul":
                label = "x"
            if brules[predint[0]] == label:
                hit = hit +1
            else:
                nnfile.write("%s\t%s\n" %(f,brules[predint[0]]))
            number = number + 1
        nf.close()

        print "see result is in result.txt"
        print "Accuracy is ", (hit/float(number))

def main():
    predint = predictint()

if __name__ == "__main__":
    main()
