import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
import scopes
from utils import *
from tensorflow.contrib.layers.python.layers import batch_norm
LOSSES_COLLECTION = '_losses'

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter


#class batch_norm(object):
#    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#        with tf.variable_scope(name):
#            self.epsilon  = epsilon
#            self.momentum = momentum
#            self.name = name

#    def __call__(self, x, train=True):
#        return tf.contrib.layers.batch_norm(x,
#                                            decay=self.momentum, 
#                                            updates_collections=None,
#                                            epsilon=self.epsilon,
#                                            scale=True,
#                                            is_training=train,
#                                            scope=self.name)

# standalone function instead of the above batch_norm class
# NOTE: the value of is_training should be set according to whether this is a training run or a test run, in order to update the statistics
# NOTE 2: the update_collections is by default None, which means the averages get updated locally; for more efficient updates:
#           set the value to tf.GraphKeys.UPDATE_OPS, and make sure to follow the guidelines from the docs:
#   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#   with tf.control_dependencies(update_ops):
#       train_op = optimizer.minimize(loss)
#@scopes.add_arg_scope
#def batchnorm(x, epsilon=1e-5, decay = 0.9, is_training=True, updates_collections=None, scale=True, name="batch_norm"):
#    is_training = tf.get_collection('istrainvar')
#    return tf.contrib.layers.batch_norm(x,
#                                        decay=decay,
#                                        updates_collections=updates_collections,
#                                        epsilon=epsilon,
#                                        scale=scale,
#                                        is_training=is_training,
#                                        scope=name)
    
def batchnorm(inputT, is_training=False, scope=None):
#    is_training = tf.get_collection('istrainvar')[0]
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope,
                                      reuse=True))
                   
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
            x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="conv2d", activation=None):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME') + b
        tf.add_to_collection('l2_loss',(tf.nn.l2_loss(w)))                   
        return conv

def conv2ds(inputs, mode_node, num_filters, kernel_size, strides=(1,1), activation=None, bn = True, name=None):
    """wrapper for tensorflow conv2d layer, with fewer inputs for initialization (the 's' stands for 'slim')"""
    k_h, k_w = _two_element_tuple(kernel_size)
    d_h, d_w = _two_element_tuple(strides)
    outputs = conv2d(inputs, num_filters, k_h, k_w, d_h, d_w, name=name, activation=activation)
    if bn:
        outputs = batchnorm(outputs, mode_node, scope = name)
    if activation is not None:
        outputs = activation(outputs)
    return outputs

#def conv2ds(inputs, num_filters, kernel_size, strides=(1,1), activation=None, bn = True, name=None):
#    """wrapper for tensorflow conv2d layer, with fewer inputs for initialization (the 's' stands for 'slim')"""
#    if bn:
#        outputs = batchnorm(inputs, name = name)
#    outputs = tf.nn.relu(outputs)
#    k_h, k_w = _two_element_tuple(kernel_size)
#    d_h, d_w = _two_element_tuple(strides)
#    outputs = conv2d(inputs, num_filters, k_h, k_w, d_h, d_w, name=name, activation=activation)
#    return outputs

def squeeze(inputs, mode_node, num_outputs):
    return conv2ds(inputs, mode_node, num_outputs, (1, 1), name='squeeze')


def expand(inputs,mode_node, num_outputs):
    with tf.variable_scope('expand'):
        if np.mod(num_outputs,2) == 1:
            num_out_1x1 = num_outputs/2+1
        else:
            num_out_1x1 = num_outputs / 2
        e1x1 = conv2ds(inputs, mode_node, num_out_1x1, [1, 1], name='1x1')
        e3x3 = conv2ds(inputs, mode_node, num_outputs/2, [3, 3], name='3x3')
    return tf.concat(values=[e1x1, e3x3], axis=3)


def fire_module(inputs, mode_node, squeeze_depth, expand_depth, residual=False, bn=False, activation=None, name='fire'):
    with tf.variable_scope(name):
        net = squeeze(inputs, mode_node, squeeze_depth)
        if activation is not None:
            net = activation(net)
        outputs = expand(net,mode_node, expand_depth)
        if residual:
            outputs = tf.add(inputs, outputs)
#        if bn:
#            outputs = batchnorm(outputs, name = 'fire_batch')
        if activation is not None:
            outputs =  activation(outputs)
        return outputs
    
    
def reduction_module(features, name, activation = None):
#    f1 = conv2ds(features, features.get_shape()[-1]/2, kernel_size = [5,5], strides=(2,2), activation=activation, bn = True, name=name + 'conv1')
#    with tf.variable_scope(name):
#        net = squeeze(features, features.get_shape()[-1]/2)
#        if activation is not None:
#            net = activation(net)
#        net = conv2ds(net, features.get_shape()[-1]/2, kernel_size = [3,3], strides=(2,2), activation=activation, bn = True, name=name + 'conv2')
#        if activation is not None:
#            net = activation(net)
#        outputs = tf.concat((f1,net),3)

        outputs = conv2ds(features, features.get_shape()[-1], kernel_size = [3,3], strides=(2,2), activation=activation, bn = True, name=name + 'conv1')
        return outputs
    
    

#@scopes.add_arg_scope
def residual_block(inputs, name='res_block', num_out=None, bn_params=None):
    '''residual convolution block, implementing the following connection:
    in--1-k-1--out
       |__1__| 
    where the numbers denote the kernel size, and k is the conv_kernel
    NOTE: the skip connection is convolved only if num_out is different than num_in
    '''
    conv_kernel = 3
    num_in = inputs.shape[-1].value
    if num_out is None:
        num_out = num_in

    with tf.variable_scope(name):
        with scopes.arg_scope([conv2d], bn_params=bn_params):
            half_num_in = int(num_in//2)
            out_1 = conv2d(inputs, half_num_in, kernel_size=1, activation=tf.nn.relu, name='conv1_1')
            out_1 = conv2d(out_1, half_num_in, kernel_size=conv_kernel, activation=tf.nn.relu, name='conv1_2')
            out_1 = conv2d(out_1, num_out, kernel_size=1, activation=tf.nn.relu, name='conv1_3')

            if num_out == num_in:
                out_2 = inputs
            else:
                out_2 = conv2d(inputs, num_out, kernel_size=1, activation=tf.nn.relu, name='conv2')

            # add the skip connection
            return out_1+out_2

#@scopes.add_arg_scope
#def hourglass(inputs, num_downsample, name='HG', bn_params=None):
#    '''Hourglass block is recursive, stopping when num_downsample is 1
#    each iteration creates the lower and upper features and connects them
#    '''
#    pooling_kernel = 2
#    with tf.variable_scope(name):
#        with scopes.arg_scope([residual_block, conv2bn], bn_params=bn_params):
#            # pre-sampling block
#            upper1 = residual_block(inputs, 'up{}_1'.format(num_downsample))
#            # downsampling
#            lower1 = max_pool(upper1, [pooling_kernel,pooling_kernel], stride=2)
#            # post-sampling block
#            lower1 = residual_block(lower1, 'low{}_1'.format(num_downsample))
#            upsample_shape = tf.shape(upper1)[1:3]
#
#            if num_downsample > 1:
#                lower2 = hourglass(lower1, num_downsample-1, name='Level_{}'.format(num_downsample-1), bn_params=bn_params)
#            else:
#                lower2 = lower1
#
#            # symmetric pre-upsampling block
#            lower3 = residual_block(lower2, 'low{}_3'.format(num_downsample))
#            upper2 = tf.image.resize_nearest_neighbor(lower3, upsample_shape, name='up{}_2'.format(num_downsample))
#
#            return upper1 + upper2


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        deconv_shape = deconv.get_shape().as_list()

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1]+deconv_shape[1:])

        if with_w:
            return deconv, w, biases
        else:
            return deconv
         
def lrelu(x, leak=0.2, name="LRelU"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x)



def _two_element_tuple(int_or_tuple):
    """Converts `int_or_tuple` to height, width.

    Several of the functions that follow accept arguments as either
    a tuple of 2 integers or a single integer.  A single integer
    indicates that the 2 values of the tuple are the same.

    This functions normalizes the input value by always returning a tuple.

    Args:
        int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

    Returns:
        A tuple with 2 values.

    Raises:
        ValueError: If `int_or_tuple` it not well formed.
    """
    if isinstance(int_or_tuple, (list, tuple)):
        if len(int_or_tuple) != 2:
            raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
        return int(int_or_tuple[0]), int(int_or_tuple[1])
    if isinstance(int_or_tuple, int):
        return int(int_or_tuple), int(int_or_tuple)
    if isinstance(int_or_tuple, tf.TensorShape):
        if len(int_or_tuple) == 2:
            return int_or_tuple[0], int_or_tuple[1]
    raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                                     'length 2')

def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
    """Adds a Max Pooling layer.

    It is assumed by the wrapper that the pooling is only done per image and not
    in depth or batch.

    Args:
        inputs: a tensor of size [batch_size, height, width, depth].
        kernel_size: a list of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: a list of length 2: [stride_height, stride_width].
            Can be an int if both strides are the same.  Note that presently
            both strides must have the same value.
        padding: the padding method, either 'VALID' or 'SAME'.
        scope: Optional scope for op_scope.

    Returns:
        a tensor representing the results of the pooling operation.
    Raises:
        ValueError: if 'kernel_size' is not a 2-D list
    """
    with tf.op_scope([inputs], scope, 'MaxPool'):
        kernel_h, kernel_w = _two_element_tuple(kernel_size)
        stride_h, stride_w = _two_element_tuple(stride)
        return tf.nn.max_pool(inputs,
                                                    ksize=[1, kernel_h, kernel_w, 1],
                                                    strides=[1, stride_h, stride_w, 1],
                                                    padding=padding)

def avg_pool(inputs, kernel_size, stride=1, padding='VALID', scope=None):
    """Adds an Average Pooling layer.

    It is assumed by the wrapper that the pooling is only done per image and not
    in depth or batch.

    Args:
        inputs: a tensor of size [batch_size, height, width, depth].
        kernel_size: a list of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: a list of length 2: [stride_height, stride_width].
            Can be an int if both strides are the same.  Note that presently
            both strides must have the same value.
        padding: the padding method, either 'VALID' or 'SAME'.
        scope: Optional scope for op_scope.

    Returns:
        a tensor representing the results of the pooling operation.
    Raises:
        ValueError: if 'kernel_size' is not a 2-D list
    """
    with tf.op_scope([inputs], scope, 'AvgPool'):
        kernel_h, kernel_w = _two_element_tuple(kernel_size)
        stride_h, stride_w = _two_element_tuple(stride)
        return tf.nn.avg_pool(inputs,
                                                    ksize=[1, kernel_h, kernel_w, 1],
                                                    strides=[1, stride_h, stride_w, 1],
                                                    padding=padding)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, bn = True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        tf.add_to_collection('l2_loss',(tf.nn.l2_loss(matrix)))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        output = tf.matmul(input_, matrix) + bias
#        if bn:
#            output = batchnorm(output, scope = scope)
        if with_w:
            return output, matrix, bias
        else:
            return output



def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0, weight=1.0, scope=None):
    """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

    It can scale the loss by weight factor, and smooth the labels.

    Args:
        logits: [batch_size, num_classes] logits outputs of the network .
        one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
        label_smoothing: if greater than 0 then smooth the labels.
        weight: scale the loss by this factor.
        scope: Optional scope for op_scope.

    Returns:
        A tensor with the softmax_cross_entropy loss.
    """
    logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
    with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
        num_classes = one_hot_labels.get_shape()[-1].value
        one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
        if label_smoothing > 0:
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                                                        labels=one_hot_labels,
                                                                                                                        name='xentropy')
        weight = tf.convert_to_tensor(weight,
                                                                    dtype=logits.dtype.base_dtype,
                                                                    name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
        tf.add_to_collection(LOSSES_COLLECTION, loss)
        return loss











