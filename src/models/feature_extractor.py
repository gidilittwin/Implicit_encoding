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




   

def resnet_44(example,args_):
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
        current = cell2D_res(c0,      3, base_size, base_size, mode, 2, 'r1',use_bn=True)#68
        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r2',use_bn=True)
        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r3',use_bn=True)
        
        current = cell2D_res(current, 3, base_size,   base_size*2, mode, 2, 'r4',use_bn=True)#34
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r5',use_bn=True)
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r6',use_bn=True)
        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r7',use_bn=True)

        current = cell2D_res(current, 3, base_size*2, base_size*4, mode, 2, 'r8',use_bn=True)#17
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r9',use_bn=True)
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r10',use_bn=True)
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r11',use_bn=True)
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r12',use_bn=True)
        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r13',use_bn=True)

        current = cell2D_res(current, 3, base_size*4, base_size*8, mode, 2, 'r14',use_bn=True)#8
        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r15',use_bn=True)
        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r16',use_bn=True)
        
        current = cell2D_res(current, 3, base_size*8, base_size*16, mode, 2, 'r17',use_bn=True)#4
        current = cell2D_res(current, 3, base_size*16, base_size*16, mode, 1, 'r18',use_bn=True)
        current = cell2D_res(current, 3, base_size*16, base_size*16, mode, 1, 'r19',use_bn=True)

#        current = cell2D_res(current, 3, base_size*16, base_size*32, mode, 2, 'r20',use_bn=True)#5
#        current = cell2D_res(current, 3, base_size*32, base_size*32, mode, 1, 'r21',use_bn=True)
#        current = cell2D_res(current, 3, base_size*32, base_size*32, mode, 1, 'r21',use_bn=True)
        
        return current











def regressor(current,args_):
    mode = args_[0]
    weights = []
    theta = args_[2]
    with tf.variable_scope("fully"):
        featue_size_ = current.get_shape().as_list()
        featue_size = tf.shape(current)
        current     = tf.nn.avg_pool(current,[1,featue_size_[1],featue_size_[2],1],[1,1,1,1],padding='VALID')
        features    = tf.squeeze(current,axis=(1,2))
        
        features = cell1D(features,128, mode, SCOPE='decode1', with_act=True, with_bn=False)
        features = cell1D(features,256, mode, SCOPE='decode2', with_act=True, with_bn=False)
#        features = cell1D(features,256, mode, SCOPE='decode3', stddev=stdev, bias_start=0.0, with_act=True, with_bn=False, act_type=tf.nn.selu)
#        features = cell1D(features,256, mode, SCOPE='decode4', stddev=stdev, bias_start=0.0, with_act=True, with_bn=False, act_type=tf.nn.selu)

        for ii in range(len(theta)):
            layer_out = theta[ii]['w']
            layer_in  = theta[ii]['in']
            stdev    = tf.sqrt(2./256)
            ww = tf.reshape(cell1D(features,layer_in*layer_out, mode, SCOPE='w'+str(ii), with_act=False, with_bn=False),(featue_size[0],layer_in,layer_out) )
            bb = tf.reshape(cell1D(features,layer_out,          mode, SCOPE='b'+str(ii), with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            gg = tf.reshape(cell1D(features,layer_out,          mode, SCOPE='g'+str(ii), with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            weights.append({'w':ww,'b':bb,'g':gg})
#            weights.append({'w':ww,'b':bb})
    return weights

















