import numpy as np
import tensorflow as tf
from model_ops import cell1D, cell2D_res, CONV2D

def mydropout(mode_node, x, prob):
    if prob!=1.0:
        return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
    else:
        return x



#%% DETECTION
        

def volumetric_softmax(node,name):
    sums = tf.reduce_sum(tf.exp(node), axis=[1,2,3], keepdims=True)
    softmax = tf.divide(tf.exp(node),sums,name=name)
    return softmax
    
     

def multiplexer(example,args_):
    mode      = args_[0]
    config    = args_[1]
    in_node   = tf.one_hot(tf.squeeze(example,-1),43794)
    with tf.variable_scope("Multiplexer"):
        current = cell1D(in_node,128, mode, SCOPE='l1', with_act=False, with_bn=False)        
        current = cell1D(current,256, mode, SCOPE='l2', with_act=True, with_bn=False)
        current = cell1D(current,256, mode, SCOPE='l3', with_act=True, with_bn=False)
        current = cell1D(current,512, mode, SCOPE='l4', with_act=True, with_bn=False)
        current = cell1D(current,512, mode, SCOPE='l5', with_act=False, with_bn=False)
    return current



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
        for ii, layer in enumerate(config.model_params['decoder']):
            features = cell1D(features,layer['size'], mode, SCOPE='decode'+str(ii+1), with_act=layer['act'], with_bn=layer['batch_norm'])
        
        features = mydropout(mode, features, config.dropout)
        tf.add_to_collection('embeddings',features)
        features_size = features.get_shape().as_list()[-1]

        # In case of multi-view training, avaerage embeddings in groups of  config.multi_image_views
        if config.multi_image:
            features_avg = []
            step_size = config.multi_image_views
            for ll in range(0,24,step_size):
                features_avg.append(tf.tile(tf.reduce_mean(features[ll:ll+step_size,:],axis=0,keep_dims=True) ,(step_size,1)) )
            features = tf.concat(features_avg,axis=0)    

        # branch out
        for ii in range(len(theta)):
#            if 'expand' in config.model_params.keys():
#                num_layers_in_branch = len(config.model_params['expand'])
#                current = features    
#                if num_layers_in_branch==1:
#                    factor  = config.model_params['expand'][0]['factor']
#                    current = cell1D(current,features_size*factor, mode, SCOPE='expand'+str(ii), with_act=True, with_bn=False)
#                else:
#                    for ll in range(num_layers_in_branch):
#                        factor  = config.model_params['expand'][ll]['factor']
#                        current = cell1D(current,features_size*factor, mode, SCOPE='expand'+str(ii)+'_'+str(ll), with_act=True, with_bn=False)
#            else:
#                current = features            
            
            if 'expand' in config.model_params.keys():
                current = features        
                for ll in range(len(config.model_params['expand'])):
                    factor  = config.model_params['expand'][ll]['factor']
                    current = cell1D(current,features_size*factor, mode, SCOPE='expand'+str(ii)+'_'+str(ll), with_act=True, with_bn=False)
            else:
                current = features

            layer_out = theta[ii]['w']
            layer_in  = theta[ii]['in']
            stdev    = 0.02
            ww = tf.reshape(cell1D(current,layer_in*layer_out, mode, SCOPE='w'+str(ii),stddev=stdev, with_act=False, with_bn=False),(featue_size[0],layer_in,layer_out) )
            bb = tf.reshape(cell1D(current,layer_out,          mode, SCOPE='b'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            gg = 1.+ tf.reshape(cell1D(current,layer_out,          mode, SCOPE='g'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,layer_out) )
            weights.append({'w':ww,'b':bb,'g':gg})
        tf.add_to_collection('weights',weights)
    return weights

















