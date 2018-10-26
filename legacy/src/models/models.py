import os
import sys
import time
import tensorflow as tf
import tensorflow.contrib.data
import numpy as np
from scipy import misc
from src.models.common_ops import *




def mydropout(mode_node, x, prob):
  # TODO: test if this is a tensor scalar of value 1.0

    return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)

def squeezenet_features(images,mode_node):

    num_modules = 8
    use_residual = 1
    num_downsample = 5
    squeeze_factor = 4
    base_dim = 64
    kernel0_size = 5
    dim_factor = 2
    avg_pooling = True
    fire_activation = tf.nn.relu
    activation = tf.nn.relu
    num_downsample = min(num_downsample, num_modules+1)
    downsample_schedule = [0] # always downsample after conv0
    step = max(float(num_modules)/(num_downsample-1), 1.0) # rest of downsampling should cover the fire modules
    downsample_schedule += np.unique(np.round(np.arange(step,num_modules+1,step)).astype(int)).tolist()
    assert(len(downsample_schedule) == num_downsample)

    images = tf.slice(images,[0,0,0,0,2],[-1,-1,-1,-1,1])
    images = tf.squeeze(images,axis=3)

    
    if not squeeze_factor == 1.0:
        res_schedule = range(1,num_modules+1,2)
    else:
        res_schedule = range(1,num_modules+1)
    # num_dim_expansions = max(1,num_modules-len(res_schedule)) # residual blocks cannot expand the dimension
    # expansion_factor = np.power(float(out_dim)/base_dim, 1.0/num_dim_expansions)
    if not(use_residual):
        dim_factor = np.sqrt(dim_factor) # with residual, factor applied every other module, so make the factor equivalent to the residual case
    # last downsampling operation replaced by average pooling
    if avg_pooling and num_modules in downsample_schedule:
        downsample_schedule.pop()

    # convolutional features starting with a simple conv, followed by fire modules
    conv0 = conv2ds(images, mode_node, base_dim, kernel0_size, name='conv0', activation=activation)
    features = max_pool(conv0, kernel_size=3, stride=2, padding='SAME')
#    if visocc_maps is not None:
#        features = tf.concat((features, visocc_maps), axis=3, name='ConcatVisOcc')
#        base_dim += self.visocclusion_conf['visocc_num_maps']
#        for conv_i in xrange(self.visocclusion_conf['vis_num_comb_conv']):
#            features = conv2bn(features, base_dim, 3, 1, tf.nn.relu, name='comb_conv{}'.format(conv_i+1))
    curr_dim = base_dim
    for ind in xrange(1, num_modules+1):
        if ind == num_modules:
            layer_name = 'features_out'
        else:
            layer_name = 'fire%d' % ind
        residual = use_residual and ind in res_schedule
        if not(residual):
            curr_dim = round((curr_dim * dim_factor) / squeeze_factor) * squeeze_factor
        squeeze_dim = curr_dim / squeeze_factor
        features = fire_module(features, mode_node, squeeze_dim, curr_dim, residual, activation=fire_activation, name=layer_name)
        if ind in downsample_schedule:
            features = max_pool(features, kernel_size=3, stride=2, padding='SAME')
        print(features.shape.as_list())
    if avg_pooling:
        init_scale = np.ones((1,6,6,1024),dtype='float32')
        init_bias = np.zeros((1,6,6,1024),dtype='float32')   
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        features = features*scale + bias
        features = avg_pool(features, kernel_size=(features.shape[1], features.shape[2]), padding='VALID')

    return features





def bones_ph(y,skeleton):
    bones = []
    for i in range(5):
        b = []
        for j in range(5):
            if j==0:
                b.append(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(y[:,skeleton[i,j],:]),1)),1))
            else:
                b.append(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(y[:,skeleton[i,j],:] - y[:,skeleton[i,j-1],:]),1)),1))
        bones.append(b)
    return bones


def regression_fully_connected(features, mode_node, batch_size, grid_size_gt, scope, reuse = False, regression_name='reg_out'):
    if reuse:
        scope.reuse_variables()
    fc1_dim = 768
    fc2_dim = 768
    fc3_dim = 768
    features = tf.contrib.layers.flatten(features)
    """this is the original regression layer, kept here for comparison"""
    fc1 = mydropout(mode_node, lrelu(linear(features, fc1_dim, scope = 'fc1', bn = False)), 0.7)
    fc2 = mydropout(mode_node, lrelu(linear(fc1,      fc2_dim, scope = 'fc2', bn = False)), 0.7)
    fc3 = mydropout(mode_node, lrelu(linear(fc2,      fc3_dim, scope = 'fc3', bn = False)), 0.7)
#    fc3 = linear(fc2, fc3_dim, scope = 'fc3', bn = True)
#    fc4 = mydropout(self, linear(fc3, emb_dim, scope = 'fc4', bn = False), 1.0)
    fc5 = linear(fc3, 63, scope = 'final', bn = False)
    marks = tf.reshape(fc5,(batch_size,3,21))*(grid_size_gt-1)
#    marks = tf.sigmoid(tf.reshape(fc5,(batch_size,3,21)))*(grid_size_gt-1)
    
    
    with tf.variable_scope("xentropy") as SCOPE:
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
        pred_cloud = tf.transpose(marks,(0,2,1))    

    return logits, pred_cloud

def adv_classifier(self, features, regression_name='reg_out'):
    fc1_dim = 512
    fc2_dim = 512
    fc3_dim = 512
    emb_dim = 64
    features = tf.contrib.layers.flatten(features)
    """this is the original regression layer, kept here for comparison"""
    fc1 = mydropout(self, lrelu(linear(features, fc1_dim, scope = 'fc1', bn = True)), 0.5)
    fc2 = mydropout(self, lrelu(linear(fc1, fc2_dim, scope = 'fc2', bn = True)), 0.5)
    fc3 = mydropout(self, lrelu(linear(fc2, fc3_dim, scope = 'fc3', bn = True)), 0.5)
#    fc3 = linear(fc2, fc3_dim, scope = 'fc3', bn = True)
#    fc4 = mydropout(self, linear(fc3, emb_dim, scope = 'fc4', bn = False), 1.0)
    fc5 = linear(fc3, 8, scope = 'final', bn = False)
 
    return fc5

def classifier_top(self, features, regression_name='reg_out'):
    features = tf.contrib.layers.flatten(features)
    """this is the original regression layer, kept here for comparison"""
    out = linear(features, 8, scope = 'final', bn = False)
 
    return out

def regression_fully_connected_cov(self, features, regression_name='reg_out'):
    fc1_dim = 768
    fc2_dim = 768
    fc3_dim = 768
    emb_dim = 50
    features = tf.contrib.layers.flatten(features)
    """this is the original regression layer, kept here for comparison"""
    fc1 = mydropout(self, lrelu(linear(features, fc1_dim, scope = 'fc1', bn = False)), 1.0)
    fc2 = mydropout(self, lrelu(linear(fc1, fc2_dim, scope = 'fc2', bn = False)), 1.0)
    fc3 = mydropout(self, lrelu(linear(fc2, fc3_dim, scope = 'fc3', bn = False)), 1.0)
#    fc3 = linear(fc2, fc3_dim, scope = 'fc3', bn = True)
    emb = linear(fc3, emb_dim, scope = 'fc4', bn = False)
    ph = tf.placeholder(tf.float32, shape=(emb_dim, emb_dim), name='cov')
    k = tf.matmul(emb,ph)
    fc5 = linear(k, self.y_dim, scope = 'final', bn = False)
 
    return fc5, emb, k, ph

def regression_fully_connected_per_seq(self, features, num_of_seq, regression_name='reg_out'):
    feature_dim = 32
    fc1_dim = 768
    fc2_dim = 768
    fc3_dim = 768
#    emb_dim = 64
    features = tf.contrib.layers.flatten(features)
    features = linear(features, feature_dim, 'dim_reduction', bn = False)
    fc = []
    for seq in range(num_of_seq + 1):
        
        """this is the original regression layer, kept here for comparison"""
        fc.append(mydropout(self, lrelu(linear(features, fc1_dim, 'fc1'+str(seq), bn = False)), 1.0))
        fc[seq] = mydropout(self, lrelu(linear(fc[seq], fc2_dim, 'fc2'+str(seq), bn = False)), 1.0)
        fc[seq] = mydropout(self, lrelu(linear(fc[seq], fc3_dim, 'fc3'+str(seq), bn = False)), 1.0)
    #    fc4 = mydropout(self, linear(fc3, emb_dim, 'fc4', bn = False), 1.0)
        fc[seq] = linear(fc[seq], self.y_dim, regression_name + str(seq), bn = False)
    return fc[0:num_of_seq], fc[num_of_seq]
#    return fc[0:num_of_seq], tf.add_n(fc[0:num_of_seq])/num_of_seq                