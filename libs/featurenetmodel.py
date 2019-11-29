#import torch
#import torchvision
#from torch.utils.data.dataset import Dataset
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
#from pathlib import Path
#import numpy as np
#from libs.utils import progress_bar
#import random
#
#from torchvision import transforms, utils
#from torch.autograd import Variable
#import torchvision.models as models
import tensorflow as tf




def inference2(models):
 
    
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',shape=[7, 7, 7, 1, 32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(models, weights, strides=[1, 2, 2, 2, 1], padding='SAME')
        batch_norm = tf.contrib.layers.batch_norm(conv,data_format='NHWC', center=True,scale=True,scope='batch_norm')
        pre_activation = tf.nn.bias_add(batch_norm, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv1)

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',shape=[5, 5, 5, 32, 32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv1, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv2)

    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',shape=[4, 4, 4, 32, 64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv2, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv3)

    # with tf.variable_scope('pooling1') as scope:
    #     pool1 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', name='pooling2')

    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',shape=[3, 3, 3, 64, 64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv3, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv4)

    # pool2
    with tf.variable_scope('pooling2') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', name='pooling2')

    #local
    with tf.variable_scope('local') as scope:
        #reshape = tf.reshape(pool2, shape=[batch_size, -1])
        reshape = tf.reshape(pool2, shape=[-1, 262144])
        dim = reshape.get_shape()[1].value
        #print('..................................')
        #print(dim)
        #print('..................................')
        weights = tf.get_variable('weights',shape=[dim, 128],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[128],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',shape=[128, 24],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[24],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local, weights), biases, name='softmax_linear')

    return softmax_linear
