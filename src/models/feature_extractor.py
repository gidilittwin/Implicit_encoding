import numpy as np
import tensorflow as tf
skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])
from model_ops import cell1D, cell2D_res, CONV2D,BatchNorm


def mydropout(mode_node, x, prob):
  # TODO: test if this is a tensor scalar of value 1.0

    return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
    
#%% DETECTION
        

def volumetric_softmax(node,name):
#    T = tf.get_variable(name='Temperature', initializer = [1.],trainable = False, dtype='float32')
#    node = node/T
    sums = tf.reduce_sum(tf.exp(node), axis=[1,2,3], keepdims=True)
    softmax = tf.divide(tf.exp(node),sums,name=name)
    return softmax
    
     

def multiplexer(data_node, mode_node):
    with tf.variable_scope("Multiplexer"):
        current = cell1D(data_node,64, mode_node, SCOPE='l1', with_act=False, with_bn=False)        
#        current = cell1D(current,256, mode_node, SCOPE='l2', with_act=True, with_bn=False)
#        current = cell1D(current,256, mode_node, SCOPE='l3', with_act=False, with_bn=False)
        current = tf.tanh(current)
    return current




   

def resnet_34(example,args_):
    mode = args_[0]
    base_size = args_[1]
    in_node = example
    ch_in = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("Input"):
        conv0_w, conv0_b = CONV2D([5,5,ch_in,base_size])
        c0 = tf.nn.conv2d(in_node,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        c0 = tf.nn.bias_add(c0, conv0_b)
#        c0 = tf.nn.max_pool(c0, ksize=[1,3,3,1], strides=(1,2,2,1), padding='SAME')
#        c1 = BatchNorm(c1,mode,SCOPE)
#        c1 = tf.nn.relu(c1)        
    with tf.variable_scope("Residuals"):
        current = cell2D_res(c0,      3, base_size, base_size, mode, 2, 'r1')
        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r2')
        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r3')
        
        current = cell2D_res(current, 3, base_size,   base_size*2, mode, 2, 'r4')
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r5')
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r6')
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r7')

        current = cell2D_res(current, 3, base_size*2, base_size*4, mode, 2, 'r8')
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r9')
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r10')
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r11')
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r12')
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r13')

        current = cell2D_res(current, 3, base_size*4, base_size*8, mode, 2, 'r14')
        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r15')
        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r16')
        
    with tf.variable_scope('Pooling'):
#        current = BatchNorm(current,mode,SCOPE)
        featue_size = current.get_shape().as_list()
        current     = tf.nn.avg_pool(current,[1,featue_size[1],featue_size[2],1],[1,1,1,1],padding='VALID')
#        batch1           = BatchNorm(current,mode,scope)
#        relu1            = tf.nn.relu(batch1)
#        conv2_w, _       = CONV2D([featue_size[1],featue_size[2],featue_size[3],1],learn_bias=False)
#        features         = tf.nn.depthwise_conv2d_native(relu1,conv2_w,strides=[1, 1, 1, 1],padding='VALID')
        return tf.squeeze(current,axis=(1,2))


def resnet_50(example,params,mode):
#    in_node = example['orthographic_image']
    in_node = example
    ch_in = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("Input"):
        conv1_w, conv1_b = CONV2D([7,7,ch_in,64])
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
        c1 = max_pool(c1, kernel_size=3, stride=2, padding='SAME')
     
    with tf.variable_scope("Residuals") as SCOPE:
        current = cell2D_res_bottleneck(c1,      3, 64, 256, mode, 1, 'r1')
        current = cell2D_res_bottleneck(current, 3, 64, 256, mode, 1, 'r2')
        current = cell2D_res_bottleneck(current, 3, 64, 256, mode, 1, 'r3')
        
        current = cell2D_res_bottleneck(current, 3, 128, 512, mode, 2, 'r4')
        current = cell2D_res_bottleneck(current, 3, 128, 512, mode, 1, 'r5')
        current = cell2D_res_bottleneck(current, 3, 128, 512, mode, 1, 'r6')
        current = cell2D_res_bottleneck(current, 3, 128, 512, mode, 1, 'r7')

        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 2, 'r8')
        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 1, 'r9')
        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 1, 'r10')
        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 1, 'r11')
        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 1, 'r12')
        current = cell2D_res_bottleneck(current, 3, 256, 1024, mode, 1, 'r13')

        current = cell2D_res_bottleneck(current, 3, 512, 2048, mode, 2, 'r14')
        current = cell2D_res_bottleneck(current, 3, 512, 2048, mode, 1, 'r15')
        current = cell2D_res_bottleneck(current, 3, 512, 2048, mode, 1, 'r16')
        
#        current = BatchNorm(current,mode,SCOPE)
        featue_size = current.get_shape().as_list()[1]
        current     = tf.nn.avg_pool(current,[1,featue_size,featue_size,1],[1,1,1,1],padding='VALID')
        size_pool   = current.get_shape().as_list()








    