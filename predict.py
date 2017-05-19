#add your imports here
import boundingBox
import predict_function as pf
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
from sys import argv
from glob import glob
import numpy as np
import os
import pickle
import pprint
import operator
"""
add whatever you think it's essential here
"""

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


# definition of classification
sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin', 'pm', 'frac', 'cos', 'delta', 'bar', 'div','^','_']

variable = ['1', '0', '3', '2', '4', '6', 'pi', 'A', 'a', 'c', 'b', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', '(', ')']
brules = {}
for i in range(0,len(sy)):
    brules[i] = sy[i]

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

class SymPred():
    def __init__(self,prediction, x1, y1, x2, y2):
        """
        <x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
        (x1,y1)
               .--------
               |           |
               |           |
                --------.
                         (x2,y2)
        """
        self.prediction = prediction
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def __str__(self):
        return self.prediction + '\t' + '\t'.join([
                                                str(self.x1),
                                                str(self.y1),
                                                str(self.x2),
                                                str(self.y2)])

class ImgPred():
    def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
        """
        sym_pred_list is list of SymPred
        latex is the latex representation of the equation
        """
        self.image_name = image_name
        self.latex = latex
        self.sym_pred_list = sym_pred_list
    def __str__(self):
        res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
        for sym_pred in self.sym_pred_list:
            res += str(sym_pred) + '\n'
        return res

def predict(image_path):
    # bounding box on given image and sort the symbols by x and y
    test_symbol_list = boundingBox.createSymbol(image_path)
    test_symbol_list = sorted(test_symbol_list, key=operator.itemgetter(2, 3))
    pre_symbol_list = []

    # for each symbol image in image list
    for i in range(len(test_symbol_list)):
        test_symbol = test_symbol_list[i]
        # prepare the each symbol image into standard size
        imvalue, image = pf.imageprepare(test_symbol[0])
        # predict
        prediction = tf.argmax(y_conv, 1)
        predint = prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)
        # analysis for dot pattern
        if test_symbol[1] != "dot":
            predict_result = brules[predint[0]]
        else:
            predict_result = "dot"
        test_symbol = (test_symbol[0], predict_result, test_symbol[2], test_symbol[3], test_symbol[4], test_symbol[5])
        test_symbol_list[i] = test_symbol

    # combine potential part in equation
    updated_symbol_list = pf.update(image_path, test_symbol_list)
    
    # for each result in result list add it into return list
    for s in updated_symbol_list:
        pre_symbol = SymPred(s[1], s[2], s[3], s[4], s[5])
        pre_symbol_list.append(pre_symbol)

    # predict the latex expression of equation
    equation = pf.toLatex(updated_symbol_list)

    # out put the result
    head, tail = os.path.split(image_path)
    img_prediction = ImgPred(tail, pre_symbol_list, equation)

    return img_prediction

if __name__ == '__main__':
    image_folder_path = argv[1]
#     image_folder_path = "./data/testEqual"
#     image_folder_path = "./data/annotated_test_Equal"
    isWindows_flag = False 
    if len(argv) == 3:
        isWindows_flag = True
    if isWindows_flag:
        image_paths = glob(image_folder_path + '\\*png')
    else:
        image_paths = glob(image_folder_path + '/*png')
    results = []

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.getcwd()+"/model/model.ckpt")
        print ("Model restored.")

        for image_path in image_paths:
            print (image_path)
            impred = predict(image_path)
            results.append(impred)

    with open('predictions.txt','w') as fout:
        for res in results:
            fout.write(str(res))
