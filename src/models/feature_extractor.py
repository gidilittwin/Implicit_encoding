import numpy as np
import tensorflow as tf
from model_ops import cell1D, cell2D_res, CONV2D, cell1D_residual, cell2D_t
from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim


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
    config    = args_[-1]
    in_node   = tf.one_hot(tf.squeeze(example,-1),43794)
    width     = config.block_width
    embedding_size = config.embedding_size
    with tf.variable_scope("Multiplexer"):
        current = cell1D(in_node,embedding_size, mode, SCOPE='one_hot_emb', with_act=False, with_bn=False, stddev=0.1)    
        for block in range(config.num_blocks):
            is_0    = block!=0
            current = cell1D_residual(current,width, mode, relu0=is_0, SCOPE='block_'+str(block))
        if config.bottleneck!=width:
            current = cell1D(current,config.bottleneck, mode, SCOPE='bootleneck', with_act=False, with_bn=False, stddev=np.sqrt(2)/np.sqrt(width))    
    return current


def mnist_config(example,args_):
    mode      = args_[0]
    config    = args_[1]
    in_node   = tf.cast(example,tf.float32)/255.0
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
    
def mlp(xyz, mode_node, theta, config):
    features = cell1D(theta,64, mode_node, SCOPE='decode', with_act=True, with_bn=False)
    features  = tf.reshape(features,(128,1,1,64))
    features = tf.tile(features,(1,784,1,1))
    xyz  = tf.expand_dims(xyz,axis=2)
    inputs = tf.concat((xyz,features),axis=-1)
    with tf.variable_scope("Input"):
        conv0_w, conv0_b = CONV2D([1,1,66,256])
        c0      = tf.nn.conv2d(inputs,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
        current = tf.tanh(current)
    with tf.variable_scope("1"):
        conv0_w, conv0_b = CONV2D([1,1,256,256])
        c0      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
        current = tf.tanh(current)
    with tf.variable_scope("2"):
        conv0_w, conv0_b = CONV2D([1,1,256,256])
        c0      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)  
        current = tf.tanh(current)
    with tf.variable_scope("3"):
        conv0_w, conv0_b = CONV2D([1,1,256,256])
        c0      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
        current = tf.tanh(current)
    with tf.variable_scope("4"):
        conv0_w, conv0_b = CONV2D([1,1,256,256])
        c0      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(c0, conv0_b)
        current = tf.tanh(current)
    with tf.variable_scope("out"):
        conv0_w, conv0_b = CONV2D([1,1,256,1])
        c0      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        sdf = tf.nn.bias_add(c0, conv0_b)
    sdf = tf.squeeze(sdf,(2,3))
    return sdf


def render(example,args_):
    mode      = args_[0]
    config    = args_[1]
    current   = tf.expand_dims(example,axis=1)
    ch_in     = current.get_shape().as_list()[-1]
    ch_out    = 128
    for layer in range(8):
        current = cell2D_t(current, 3, 3, ch_in, ch_out, mode, stride=2, SCOPE="render_layer_"+str(layer), padding='SAME', bn=True, act=True)                                
        ch_in   = ch_out
    with tf.variable_scope("project_image"):
        conv0_w, conv0_b = CONV2D([1,1,ch_out,3])
        current      = tf.nn.conv2d(current,conv0_w,strides=[1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(current, conv0_b)  
    return current


def resnet_config(example,args_):
    mode      = args_[0]
    config    = args_[1]
    in_node   = tf.image.resize_images(example,[137,137])
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
    

def resnet_50(example,args_):   
    in_node   = tf.image.resize_images(example,[299,299])
    in_node   = tf.subtract(in_node, 0.5)
    in_node   = tf.multiply(in_node, 2.0)   
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
       net, end_points = resnet_v2.resnet_v2_50(in_node, is_training=False)
    features = tf.squeeze(net,(1,2))
    features = tf.stop_gradient(features)
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
        
        # In case of multi-view training, avaerage embeddings in groups of  config.multi_image_views
        if config.multi_image:
#            features_avg = []
#            num_obj =  config.batch_size/config.multi_image_views
#            for ll in range(num_obj):
#                features_avg.append(tf.tile(tf.reduce_mean(features[ll:config.batch_size:num_obj,:],axis=0,keep_dims=True) ,(config.multi_image_views,1)) )
#            centers = tf.stack(features_avg,axis=1)
#            centers = tf.reshape(centers,(config.batch_size,-1))
#            tf.add_to_collection('centers',centers)   
            if config.multi_image_pool=='mean':
                features = tf.tile(tf.reduce_mean(features,axis=0,keep_dims=True),(featue_size[0],1))
            elif config.multi_image_pool=='max':
                features = tf.tile(tf.reduce_max(features,axis=0,keep_dims=True),(featue_size[0],1))
                
        for ii, layer in enumerate(config.model_params['decoder']):
            features = cell1D(features,layer['size'], mode, SCOPE='decode'+str(ii+1), with_act=layer['act'], with_bn=layer['batch_norm'])
        features = mydropout(mode, features, config.dropout)
        tf.add_to_collection('embeddings',features)
        features_size = features.get_shape().as_list()[-1]

        # branch out
        for ii in range(len(theta)):        
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




def conv_regressor(features,args_):
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

        # branch out
        for ii in range(len(theta)):        
            if 'expand' in config.model_params.keys():
                current = features        
                for ll in range(len(config.model_params['expand'])):
                    factor  = config.model_params['expand'][ll]['factor']
                    current = cell1D(current,features_size*factor, mode, SCOPE='expand'+str(ii)+'_'+str(ll), with_act=True, with_bn=False)
            else:
                current = features
            layer_out = theta[ii]['w']
            layer_in  = theta[ii]['in']
            kernel_size  = theta[ii]['k']
            stdev    = 0.02
            ww = tf.reshape(cell1D(current,layer_in*layer_out*kernel_size*kernel_size, mode, SCOPE='w'+str(ii),stddev=stdev, with_act=False, with_bn=False),(featue_size[0],kernel_size,kernel_size,layer_out,layer_in) )
            bb = tf.reshape(cell1D(current,layer_out,          mode, SCOPE='b'+str(ii),stddev=stdev, with_act=False, with_bn=False) ,(featue_size[0],1,1,layer_out) )
            weights.append({'w':ww,'b':bb})
        tf.add_to_collection('weights',weights)
    return weights












