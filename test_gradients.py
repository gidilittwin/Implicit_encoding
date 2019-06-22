#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:22:34 2019

@author: gidi
"""

import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian




X = tf.random_normal([3, 3])

a = tf.constant([3.])
b = tf.constant([4.])
c = tf.constant([1.])


y = tf.reduce_sum(tf.nn.relu(c*tf.nn.relu(b*tf.nn.relu(a*X))) - 1,axis=-1,keep_dims=True)
#y = tf.nn.relu(c*tf.nn.relu(b*tf.nn.relu(a*X))) - 1


grad = tf.gradients(y, X)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    grad_value = sess.run(grad)
    print(grad_value)
