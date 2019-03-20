import numpy as np
import tensorflow as tf
skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])
from model_ops import cell1D, cell2D_res, CONV2D,BatchNorm
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

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




   

#def resnet_44(example,args_):
#    mode = args_[0]
#    base_size = args_[1]
#    in_node = example
#    ch_in = in_node.get_shape().as_list()[-1]
#    with tf.variable_scope("Input"):
#        conv0_w, conv0_b = CONV2D([5,5,ch_in,base_size])
#        c0 = tf.nn.conv2d(in_node,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
#        c0 = tf.nn.bias_add(c0, conv0_b)
##        c0 = tf.nn.max_pool(c0, ksize=[1,3,3,1], strides=(1,2,2,1), padding='SAME')
##        c1 = BatchNorm(c1,mode,SCOPE)
##        c1 = tf.nn.relu(c1)        
#    with tf.variable_scope("Residuals"):
#        BN = True
#        current = cell2D_res(c0,      3, base_size, base_size, mode, 2, 'r1',use_bn=BN)#68
#        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r2',use_bn=BN)
#        current = cell2D_res(current, 3, base_size, base_size, mode, 1, 'r3',use_bn=BN)
#        
#        current = cell2D_res(current, 3, base_size,   base_size*2, mode, 2, 'r4',use_bn=BN)#34
#        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r5',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r6',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*2, base_size*2, mode, 1, 'r7',use_bn=BN)
#
#        current = cell2D_res(current, 3, base_size*2, base_size*4, mode, 2, 'r8',use_bn=BN)#17
#        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r9',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r10',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r11',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r12',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*4, base_size*4, mode, 1, 'r13',use_bn=BN)
#
#        current = cell2D_res(current, 3, base_size*4, base_size*8, mode, 2, 'r14',use_bn=BN)#8
#        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r15',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*8, base_size*8, mode, 1, 'r16',use_bn=BN)
#        
#        current = cell2D_res(current, 3, base_size*8, base_size*16, mode, 2, 'r17',use_bn=BN)#4
#        current = cell2D_res(current, 3, base_size*16, base_size*16, mode, 1, 'r18',use_bn=BN)
#        current = cell2D_res(current, 3, base_size*16, base_size*16, mode, 1, 'r19',use_bn=BN)
#
##        current = cell2D_res(current, 3, base_size*16, base_size*32, mode, 2, 'r20',use_bn=True)#5
##        current = cell2D_res(current, 3, base_size*32, base_size*32, mode, 1, 'r21',use_bn=True)
##        current = cell2D_res(current, 3, base_size*32, base_size*32, mode, 1, 'r21',use_bn=True)
#        
#        return current



   

def resnet_config(example,args_):
    mode      = args_[0]
    config    = args_[1]
    in_node   = example
    ch_in     = in_node.get_shape().as_list()[-1]
    base_size = config.model_params['encoder']['base_size']
    with tf.variable_scope("Input"):
        params  = config.model_params['encoder']['input']
        conv0_w, conv0_b = CONV2D([params['k'],params['k'],ch_in,base_size])
        c0      = tf.nn.conv2d(in_node,conv0_w,strides=[1, params['stride'], params['stride'], 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
    with tf.variable_scope("Residuals"):
        for ii, layer in enumerate(config.model_params['encoder']['residuals']):
            if ii<config.bn_l0:
                BN = 0
            else:
                BN = config.batch_norm                
            current = cell2D_res(current, layer['k'], base_size*layer['s_in'],  base_size*layer['s_out'], mode, layer['stride'], 'r'+str(ii+1),use_bn=BN)#68
    
    featue_size_ = current.get_shape().as_list()
    current      = tf.nn.avg_pool(current,[1,featue_size_[1],featue_size_[2],1],[1,1,1,1],padding=config.padding)
    featue_size_ = current.get_shape().as_list()
    features     = tf.reshape(current,(-1,featue_size_[1]*featue_size_[2]*featue_size_[3]))
    return features



def BatchNorm_hard(inputT, is_training=True, scope=None, deploy=False):
    # Note: is_training is tf.placeholder(tf.bool) type
    if deploy==False:
        return tf.cond(is_training,  
                    lambda: batch_norm(inputT, is_training=True,  
                                       center=False, scale=False, decay=0.9, updates_collections=None, scope=scope),  
                    lambda: batch_norm(inputT, is_training=False,  
                                       center=False, scale=False, decay=0.9, updates_collections=None, scope=scope, reuse = True))  
    else:
        return batch_norm(inputT, is_training=False,center=False, scale=False, decay=0.9, updates_collections=None, scope=scope, reuse = False)  




def sample_normal(shape, is_training=True, scope=None, deploy=False):
    # Note: is_training is tf.placeholder(tf.bool) type
    if deploy==False:
        return tf.cond(is_training,  
                    lambda: tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,name='std_norm'),
                    lambda: tf.zeros(shape,dtype=tf.float32,name='std_norm'))




def regressor(features,args_):
    mode    = args_[0]
    config  = args_[1]
    weights = []
    theta   = config.model_params['theta']
    featue_size  = tf.shape(features)
    with tf.variable_scope("fully",reuse=tf.AUTO_REUSE):
               
#        mean       = cell1D(features,featue_size_[-1], mode, SCOPE='mean', with_act=False, with_bn=False)
#        log_stddev = cell1D(features,featue_size_[-1], mode, SCOPE='log_stddev', with_act=False, with_bn=False)
#        std_norm   = sample_normal(tf.shape(mean),mode)
#        features   = mean + tf.exp(log_stddev) * std_norm
#        tf.add_to_collection('VAE_loss',[mean,log_stddev])
        
        for ii, layer in enumerate(config.model_params['decoder']):
            features = cell1D(features,layer['size'], mode, SCOPE='decode'+str(ii+1), with_act=layer['act'], with_bn=layer['batch_norm'])
        tf.add_to_collection('embeddings',features)
        
        if config.multi_image:
            features =    tf.tile(tf.reduce_mean(features,axis=0,keep_dims=True) ,(featue_size[0],1)) 
        
        for ii in range(len(theta)):
            layer_out = theta[ii]['w']
            layer_in  = theta[ii]['in']
#            stdev    = tf.sqrt(2./1024)
            stdev    = 0.02
            ww = tf.reshape(cell1D(features,layer_in*layer_out, mode, SCOPE='w'+str(ii),stddev=stdev, with_act=False, with_bn=False),(featue_size[0],layer_in,layer_out) )
            bb = tf.reshape(cell1D(features,layer_out,          mode, SCOPE='b'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            gg = 1.+ tf.reshape(cell1D(features,layer_out,          mode, SCOPE='g'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            weights.append({'w':ww,'b':bb,'g':gg})
        tf.add_to_collection('weights',weights)
    return weights

















