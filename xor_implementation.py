# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 16:46:36 2018

@author: Filip Ceglik
"""

import tensorflow as tf

##########


#creating placeholders for data
#the placeholders are floats to avoid 

x = tf.placeholder(tf.float32, shape=[4,2], name="x_input")  #a collection representing possible inputs for XOR
y = tf.placeholder(tf.float32, shape=[4,1], name="y_input")  #a collection for output results from XOR


xV_1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="theta_1")
yV_1 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="theta_2")


bias_1 = tf.Variable(tf.zeros([2]), name="bias_1")
bias_2 = tf.Variable(tf.zeros([1]), name="bias_2")



model = tf.sigmoid(tf.matmul(x,xV_1) + bias_1)
hypothesis = tf.sigmoid(tf.matmul(model,yV_1) + bias_2)

cost_function = tf.reduce_mean(( (y * tf.log(hypothesis)) + ((1-y) * tf.log(1.0 - hypothesis)) ) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost_function)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()

session = tf.Session()
writer = tf.summary.FileWriter("./logs/xor_logs", session.graph_def)
session.run(init)

for i in range(100000):
    session.run(train_step, feed_dict={x: XOR_X, y: XOR_Y})
    if i % 1000 == 0:
            print('Epoch ', i)
            print('Hypothesis ', session.run(hypothesis, feed_dict={x: XOR_X, y: XOR_Y}))
            print('Theta1 ', session.run(xV_1))
            print('Bias1 ', session.run(bias_1))
            print('Theta2 ', session.run(yV_1))
            print('Bias2 ', session.run(bias_2))
            print('cost ', session.run(cost_function, feed_dict={x: XOR_X, y: XOR_Y}))
           