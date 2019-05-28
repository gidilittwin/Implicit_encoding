import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def mydropout(mode_node, x, prob):
  # TODO: test if this is a tensor scalar of value 1.0

    return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
 

def BatchNorm(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,  
                lambda: batch_norm(inputT, is_training=True,  
                                   center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),  
                lambda: batch_norm(inputT, is_training=False,  
                                   center=True, scale=True, decay=0.9, updates_collections=None, scope=scope, reuse = True))  

    
    

def CONV2D(shape,bias=True):
#   initializer = tf.random_normal_initializer( stddev=1./np.sqrt(shape[0]*shape[1]*shape[2]))
   initializer = tf.random_normal_initializer( stddev=np.sqrt(2./(shape[0]*shape[1]*shape[2])))
   conv_weights = tf.get_variable('weights',shape, initializer = initializer)
   if bias==True:
       conv_biases  = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   else:
       conv_biases = []
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases


def lrelu(x, leak=0.2, name="LRelU"):
   with tf.name_scope(name):
       return tf.maximum(x, leak*x)        


def cell_2d_cnn(in_node,scope,mode,weights,act=True,normalize=False,bn=False):
    with tf.variable_scope(scope):
        if normalize==True:
            weights['w'] = weights['w']/tf.norm(weights['w'],axis=2,keep_dims=True)
            c1 = tf.matmul(in_node,weights['w'])*weights['g'] + weights['b']            
        else:
            c1 = tf.matmul(in_node,weights['w'])*weights['g'] + weights['b']
        if bn==True:
            c1 = BatchNorm(c1,mode,scope)
        if act!=None:
            c1 = act(c1)
    return c1


def cell_conv_t(in_node,scope,mode,weights,act=True,normalize=False,bn=False):
    with tf.variable_scope(scope):

        ww          = weights['w']
        bb          = weights['b']
        ww_shape    = ww.get_shape().as_list()       
        input_shape = in_node.get_shape().as_list()
        batch_size  = in_node.get_shape().as_list()[0]
        if len(input_shape)==3:
            ww     = tf.transpose(tf.squeeze(ww,(1,2)),(0,2,1))
            bb     = tf.squeeze(bb,1)
            c1     = tf.matmul(in_node,ww) + bb 
            result = tf.reshape(c1,(batch_size,2,2,-1))        
        else:
            result = []
            for inst in range(batch_size):
                if ww_shape[1]==3:
                    c1 = tf.nn.conv2d_transpose(in_node[inst:inst+1,:,:,:],
                                            ww[inst,:,:,:,:],
                                            [1,input_shape[1]*2,input_shape[2]*2,ww_shape[-2]],
                                            strides = [1,2,2,1]) 
                elif ww_shape[1]==1:
                    c1 = tf.nn.conv2d_transpose(in_node[inst:inst+1,:,:,:],
                                            ww[inst,:,:,:,:],
                                            [1,input_shape[1],input_shape[2],ww_shape[-2]],
                                            strides = [1,1,1,1])                     
                result.append(c1)
            result = tf.concat(result,axis=0) + bb
        if bn==True:
            result = BatchNorm(result,mode,scope)
        if act!=None:
            result = act(result)
    return result






def deep_shape(xyz, mode_node, theta, config):
    if config.symetric:
        x,y,z = tf.split(xyz,[1,1,1],axis=2)
        x = tf.abs(x)
        image = tf.concat((x,y,z),axis=2)
    else:
        image        = xyz
    
    dnn_params = config.model_params['dnn_params']    
    if dnn_params['activations']=='relu':
        act_=tf.nn.relu
    elif dnn_params['activations']=='elu':
        act_=tf.nn.elu
    elif dnn_params['activations']=='tanh':
        act_=tf.tanh        
    elif dnn_params['activations']=='lrelu':
        act_=lrelu 
        
    image_shape = image.get_shape().as_list()
    if len(image_shape)==4:
        image = tf.reshape(image,(image_shape[0],-1,3))
    for ii in range(len(theta)):
        if ii<len(theta)-1:
            act=act_
            bn = False
        else:
            act=None
            bn = False
        in_size = image.get_shape().as_list()[-1]
        print('layer '+str(ii)+' size = ' + str(in_size) +' out size='+str(theta[ii]['w']))
        image = cell_2d_cnn(image,   'l'+str(ii),mode_node,theta[ii],act=act,normalize=dnn_params['normalize'],bn=bn) 
    sdf = image
    if len(image_shape)==4:
        sdf = tf.reshape(sdf,(1,image_shape[1],image_shape[2]))    

    return sdf


def deep_sensor(image, mode_node, theta, config):
    dnn_params = config.model_params['dnn_params']    
    if dnn_params['activations']=='relu':
        act_=tf.nn.relu
    elif dnn_params['activations']=='elu':
        act_=tf.nn.elu
    elif dnn_params['activations']=='tanh':
        act_=tf.tanh        
    elif dnn_params['activations']=='lrelu':
        act_=lrelu 
        
    for ii in range(len(theta)):
        if ii<len(theta)-1:
            act=act_
            bn = False
        else:
            act=None
            bn = False
        in_size = image.get_shape().as_list()[-1]
        print('layer '+str(ii)+' size = ' + str(in_size) +' out size='+str(theta[ii]['w']))
        image = cell_conv_t(image,   'l'+str(ii),mode_node,theta[ii],act=act,normalize=dnn_params['normalize'],bn=bn) 
    sdf = image
    return sdf





def sample_points(model_fn,args,shape = [1,100,100],samples=None,use_samps=False):
    if use_samps==False:
        grid_shape  = [shape[0],shape[1],shape[2],3]
        samples     = tf.random_uniform(grid_shape,
                                        minval=-1.0,
                                        maxval=1.0,
                                        dtype=tf.float32)
        samples = samples/tf.norm(samples,axis=-1,keep_dims=True)    
        grid_shape  = [shape[0],shape[1],shape[2],1]
        U           = tf.random_uniform(grid_shape,
                                        minval=0.0,
                                        maxval=1.0,
                                        dtype=tf.float32)   
        samples = samples*tf.pow(U,1/3.)
        
    response    = model_fn(samples,args)
    dy_dx   = []
    d2y_dx2 = []
    for ii in range(shape[0]):
        dydx   = tf.gradients(response[ii,:,:],samples)[0]
        d2ydx2 = tf.gradients(dydx,samples)[0]
        dy_dx.append(dydx)     
        d2y_dx2.append(d2ydx2)     
    dy_dx = tf.concat(dy_dx,axis=0) 
    d2y_dx2 = tf.concat(d2y_dx2,axis=0) 
    dy_dx_n = tf.norm(dy_dx,axis=-1,keep_dims=True)
    evals = {'x':samples,'y':response,'dydx':dy_dx,'d2ydx2':d2y_dx2,'dydx_norm':dy_dx_n}
    return evals
        



def sample_points_list(model_fn,args,shape = [1,1000],samples=None,use_samps=False):
    if use_samps==False:
        grid_shape  = [shape[0],shape[1],3]
        samples     = tf.random_uniform(grid_shape,
                                        minval=-1.0,
                                        maxval=1.0,
                                        dtype=tf.float32)
        samples = samples/tf.norm(samples,axis=-1,keep_dims=True)    
        grid_shape  = [shape[0],shape[1],1]
        U           = tf.random_uniform(grid_shape,
                                        minval=0.0,
                                        maxval=1.0,
                                        dtype=tf.float32)   
        samples = samples*tf.pow(U,1/3.)
        
    response    = model_fn(samples,args)
#    dy_dx   = []
#    for ii in range(shape[0]):
#        dydx   = tf.gradients(response[ii,:,:],samples)[0]
#        dy_dx.append(dydx)     
#    dy_dx = tf.concat(dy_dx,axis=0) 
#    dy_dx_n = tf.norm(dy_dx,axis=-1,keep_dims=True)
#    mask    = tf.cast(tf.greater(response,0.),tf.float32)
#    evals = {'x':samples,'y':response,'dydx':dy_dx,'dydx_norm':dy_dx_n,'mask':mask}
    evals = {'x':samples,'y':response}
    return evals
        





def sample_points_list_2D(model_fn,args,shape = [32,28],samples=None,use_samps=False):
    config = args[-1]
    if use_samps==False:
        x = np.linspace(0.,1., shape[1])
        u,v = np.meshgrid(x,x)
        samples = tf.concat((tf.reshape(u,(1,-1,1)),tf.reshape(v,(1,-1,1))),axis=-1)
        samples = tf.cast(tf.tile(samples,(shape[0],1,1)),tf.float32)
    response    = model_fn(samples,args)
    evals = {'x':samples,'y':response}
    return evals
        

def const_vector(model_fn,args,shape = [32,10],samples=None,use_samps=False):
    config = args[-1]
    if use_samps==False:
        samples = tf.ones((shape[0],1,shape[1]),tf.float32)
#        samples = tf.cast(tf.tile(samples,(shape[0],1,1)),tf.float32)
    response    = model_fn(samples,args)
    evals = {'x':samples,'y':response}
    return evals



def camera_vector(model_fn,args,shape = [32,10],samples=None,use_samps=False):
    config = args[-1]
    if use_samps==False:
        samples = tf.ones((shape[0],1,shape[1]),tf.float32)
#        samples = tf.cast(tf.tile(samples,(shape[0],1,1)),tf.float32)
    response    = model_fn(samples,args)
    evals = {'x':samples,'y':response}
    return evals









