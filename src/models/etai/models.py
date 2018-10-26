import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
def lrelu(x):
    return tf.nn.relu(x) -0.2*tf.nn.relu(-x)

def BatchNorm(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope,
                                      reuse=True))

def get_weights(shape):
   weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=1/np.sqrt(shape[1])))
   biases = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_n',(tf.nn.l2_loss(weights)))
   tf.add_to_collection('weights',weights)
   tf.add_to_collection('weights',biases)
   return weights, biases


def get_conv_weights(shape):
   weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=1/np.sqrt(shape[0]*shape[1]*shape[2])))
   biases = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_n',(tf.nn.l2_loss(weights)))
   return weights, biases

def identity(inp):
    return inp

 








def conv_model1(inp, isTrainVar, batchsize, num_of_labels, activation, sample_size, eps, bn = False):

    with tf.variable_scope('net_vars') as scope:
        with tf.variable_scope('input_block1') as scope:
            tmp = conv_layer(inp, [5,5,3,192], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            
        with tf.variable_scope('input_block2') as scope:
            tmp = conv_layer(tmp, [1,1,192,192], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block2.1') as scope:
            tmp = conv_layer(tmp, [3,3,192,240], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.9), lambda: tmp)
            
        with tf.variable_scope('input_block3') as scope:
            tmp = conv_layer(tmp, [1,1,240,240], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block3.1') as scope:
            tmp = conv_layer(tmp, [2,2,240,260], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.8), lambda: tmp)
            
        with tf.variable_scope('input_block4') as scope:
            tmp = conv_layer(tmp, [1,1,260,260], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block4.1') as scope:
            tmp = conv_layer(tmp, [2,2,260,280], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.7), lambda: tmp)
            
        with tf.variable_scope('input_block5') as scope:
            tmp = conv_layer(tmp, [1,1,280,280], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block5.1') as scope:
            tmp = conv_layer(tmp, [2,2,280,300], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.6), lambda: tmp)
            
        with tf.variable_scope('input_block6') as scope:
            tmp = conv_layer(tmp, [1,1,300,300], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.5), lambda: tmp)
        with tf.variable_scope('input_block6.1') as scope:
            logits = conv_layer(tmp, [1,1,300,num_of_labels], isTrainVar, scope, identity, sample_size, eps, bn = 0, stab = 0, stride = 1)

            logits = tf.reshape(logits,(batchsize,num_of_labels))
    return logits


def conv_model2(inp, isTrainVar, batchsize, num_of_labels, activation, sample_size, eps, bn = False):

    with tf.variable_scope('net_vars') as scope:
        with tf.variable_scope('input_block1') as scope:
            tmp = conv_layer(inp, [3,3,3,384], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            
        with tf.variable_scope('input_block2') as scope:
            tmp = conv_layer(tmp, [1,1,384,384], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block2.1') as scope:
            tmp = conv_layer(tmp, [2,2,384,384], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block2.2') as scope:
            tmp = conv_layer(tmp, [2,2,384,640], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block2.3') as scope:
            tmp = conv_layer(tmp, [2,2,640,640], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.9), lambda: tmp)
            
        with tf.variable_scope('input_block3') as scope:
            tmp = conv_layer(tmp, [1,1,640,640], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block3.1') as scope:
            tmp = conv_layer(tmp, [2,2,640,768], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block3.2') as scope:
            tmp = conv_layer(tmp, [2,2,768,768], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block3.3') as scope:
            tmp = conv_layer(tmp, [2,2,768,768], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.8), lambda: tmp)
            
        with tf.variable_scope('input_block4') as scope:
            tmp = conv_layer(tmp, [1,1,768,768], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block4.1') as scope:
            tmp = conv_layer(tmp, [2,2,768,896], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block4.2') as scope:
            tmp = conv_layer(tmp, [2,2,896,896], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.7), lambda: tmp)
            
        with tf.variable_scope('input_block5') as scope:
            tmp = conv_layer(tmp, [3,3,896,896], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block5.1') as scope:
            tmp = conv_layer(tmp, [2,2,896,1024], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block5.2') as scope:
            tmp = conv_layer(tmp, [2,2,1024,1024], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.nn.max_pool(tmp,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.6), lambda: tmp)
            
        with tf.variable_scope('input_block6') as scope:
            tmp = conv_layer(tmp, [1,1,1024,1024], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block6.1') as scope:
            tmp = conv_layer(tmp, [2,2,1024,1152], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
            tmp = tf.cond(isTrainVar, lambda: tf.nn.dropout(tmp, 0.5), lambda: tmp)
        
        with tf.variable_scope('input_block7') as scope:
            tmp = conv_layer(tmp, [1,1,1152,1152], isTrainVar, scope, activation, sample_size, eps, bn = bn, stab = 1, stride = 1)
        with tf.variable_scope('input_block7.1') as scope:
            logits = conv_layer(tmp, [1,1,1152,num_of_labels], isTrainVar, scope, identity, sample_size, eps, bn = 0, stab = 0, stride = 1)

            logits = tf.reshape(logits,(batchsize,num_of_labels))
    return logits

 
def stability_loss(inp, sample_size, eps, randvec = None):
    if randvec is not None:
        o1 = tf.gather(inp,randvec[0:sample_size,0])
        me1, var1 = tf.nn.moments(o1,0)
        o2 = tf.gather(inp,randvec[sample_size:2*sample_size])
        me2, var2 = tf.nn.moments(o2,0)
    else:    
        me1, var1 = tf.nn.moments(inp[0:sample_size,:],0)
        me2, var2 = tf.nn.moments(inp[sample_size:2*sample_size,:],0)
    var1 = tf.abs(var1) 
    var2 = tf.abs(var2) 
    tf.add_to_collection('l2_norm',(tf.reduce_mean(tf.square(1 - var1/(var2+eps)))))

def conv_layer(inp, shape, isTrainVar, scope, activation, samples, eps, bn = 0, stab = 1, stride = 1):
    weights, bias = get_conv_weights(shape)
    out = tf.nn.conv2d(inp, weights, strides=[1, stride, stride, 1], padding='SAME') + bias
    if bn==1:
        out = BatchNorm(out, isTrainVar, scope)
    out = activation(out)
    if stab==1:
#        reshaped = tf.reshape(weights,(np.prod(shape[0:3]),shape[3]))
#        randvec = tf.cast(tf.random_uniform((2*samples,1),0,np.prod(shape[0:3])),tf.int32)
#        stability_loss(reshaped, samples, eps, randvec)
#        stability_loss(out, samples, eps)
        reshaped = tf.reshape(weights,(np.prod(shape[0:3]),shape[3]))
        idx = tf.cast(tf.random_uniform((2*samples,1),0,np.prod(shape[0:3])),tf.int32)
        out1 = tf.gather(reshaped,idx[0:samples,:])
        out2 = tf.gather(reshaped,idx[samples:2*samples,:])
        me1, var1 = tf.nn.moments(out1,axes = [0])
    #    std1 = tf.sqrt(tf.abs(var1))
        var1 = tf.abs(var1)
        me2, var2 = tf.nn.moments(out2,axes = [0])
    #    std2 = tf.sqrt(tf.abs(var2))
        var2 = tf.abs(var2)
        tf.add_to_collection('l2_norm',(tf.reduce_mean(tf.square(1 - var1/(var2+eps)))))
#        
        o1 = out[0:samples,:]
        o2 = out[samples:2*samples,:]
        m1, v1 = tf.nn.moments(o1,axes = [0])
        v1 = tf.abs(v1)
    #    std1 = tf.sqrt(tf.abs(var1))
        m2, v2 = tf.nn.moments(o2,axes = [0])
        v2 = tf.abs(v2)
    #    std2 = tf.sqrt(tf.abs(var2))
        tf.add_to_collection('l2_norm',(tf.reduce_mean(tf.square(1 - v1/(v2+eps)))))
    return out



