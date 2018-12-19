import numpy as np
import tensorflow as tf
import signed_dist_functions as SDF
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



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

    
    

def CONV2D(shape):
#   initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN') 
   initializer = tf.random_normal_initializer( stddev=1./np.sqrt(shape[0]*shape[1]*shape[2]))
#   initializer = tf.random_normal_initializer( stddev=np.sqrt(2./(shape[0]*shape[1]*shape[2])))
   conv_weights = tf.get_variable('weights',shape, initializer = initializer)
   conv_biases  = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases







#%% DETECTION
        


#def cell_2d(image,output_dim,scope,mode,act=True):
#    with tf.variable_scope(scope):
#        input_dim = image.get_shape().as_list()[-1]
#        conv1_w, conv1_b = CONV2D([1,1,input_dim,output_dim])
#        c1 = tf.nn.conv2d(image,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
#        c1 = tf.nn.bias_add(c1, conv1_b)
#        if act==True:
##            c1 = BatchNorm(c1,mode,scope)
#            c1 = tf.nn.relu(c1)
#    return c1
    

def cell_2d(in_node,scope,mode,weights,bias,act=True,normalize=False,bn=False):
    with tf.variable_scope(scope):
        in_node = tf.expand_dims(in_node,2)
        input_dim = weights
        output_dim = bias
        conv1_w, conv1_b = CONV2D([1,1,input_dim,output_dim])
        if normalize==True:
#            conv1_w = conv1_w/tf.norm(conv1_w,axis=-1,keep_dims=True)
            #        matrix = tf.nn.l2_normalize(matrix,(0,1))
            matrix = tf.squeeze(conv1_w,axis=(0,1))
            e,v  = tf.linalg.eigh(tf.matmul(tf.transpose(matrix),matrix))
            large_e = e[-1]
            conv1_w = conv1_w/large_e*2.5
            
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
        if bn==True:
            c1 = BatchNorm(c1,mode,scope)
        if act==True:
#            c1 = tf.nn.relu(c1)
            c1 = tf.nn.selu(c1)
#            c1 = tf.tanh(c1)
        c1 = tf.squeeze(c1,2)
    return c1

#def cell_2d_cnn(in_node,scope,mode,weights,bias,act=True,normalize=False,bn=False):
#    with tf.variable_scope(scope):
#        if normalize==True:
#            weights = weights/tf.norm(weights,axis=1,keep_dims=True)
#        c1 = tf.matmul(in_node,weights) + bias
#        if bn==True:
#            c1 = BatchNorm(c1,mode,scope)
#        if act==True:
#            c1 = tf.nn.relu(c1)
##            c1 = tf.tanh(c1)
#    return c1


def softargmax_3d(pred, grid_size_gt, name=None):
    gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
    grid_x = (gt_meshgrid[0]).astype(np.float32)
    grid_y = (gt_meshgrid[1]).astype(np.float32)
    grid_z = (gt_meshgrid[2]).astype(np.float32)
    grid_x_tf = tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32')
    grid_y_tf = tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32')
    grid_z_tf = tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32')
    pred_x = tf.expand_dims(tf.reduce_sum(pred * tf.expand_dims(tf.expand_dims(grid_x_tf,axis=0),axis=4),axis=[1,2,3]),axis=2)
    pred_y = tf.expand_dims(tf.reduce_sum(pred * tf.expand_dims(tf.expand_dims(grid_y_tf,axis=0),axis=4),axis=[1,2,3]),axis=2)
    pred_z = tf.expand_dims(tf.reduce_sum(pred * tf.expand_dims(tf.expand_dims(grid_z_tf,axis=0),axis=4),axis=[1,2,3]),axis=2)
    coordinates = tf.concat((pred_y,pred_x,pred_z),axis=2)
    return coordinates
         
   
   
#def deep_sdf2(xyz, mode_node, theta):
##    xyz_sq = tf.sqrt(tf.reduce_sum(xyz**2,axis=-1,keep_dims=True))
##    image = tf.concat((xyz,xyz_sq),axis=-1)
#    image = xyz
#
#    for ii in range(len(theta)):
#        if ii<len(theta)-1:
#            act=True
#        else:
#            act=False
##        image = cell_2d_cnn (image,   'l'+str(ii),mode_node,theta[ii]['w'],theta[ii]['b'],act=act,normalize=False) 
#        image = cell_2d (image,   'l'+str(ii),mode_node,theta[ii]['w'],theta[ii]['b'],act=act,normalize=True) 
#
#    grads = tf.gradients(image,xyz)[0]
#    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=2,keep_dims=True))
#    sdf = image/grads_norm
##    image = tf.nn.relu(image)
#    
#    return sdf, image 



def deep_sdf2(xyz, mode_node, theta):
    image        = xyz
    image_shape = image.get_shape().as_list()
    if len(image_shape)==4:
        image = tf.reshape(image,(image_shape[0],-1,3))
    for ii in range(len(theta)):
        if ii<len(theta)-1:
            act=True
            bn = False
        else:
            act=False
            bn = False
        in_size = image.get_shape().as_list()[-1]
        print('layer '+str(ii)+' size = ' + str(in_size) +' out size='+str(theta[ii]['w']))
        image = cell_2d(image,   'l'+str(ii),mode_node,in_size,theta[ii]['w'],act=act,normalize=False,bn=bn) 
    sdf = image
    if len(image_shape)==4:
        sdf = tf.reshape(sdf,(1,image_shape[1],image_shape[2]))    
#    grads = tf.gradients(sdf,xyz)[0]
#    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=2,keep_dims=True))
#    sdf = sdf/grads_norm
    return sdf




def deep_sdf4(xyz,embeddings, mode_node, theta):
    image        = xyz
    image_shape = image.get_shape().as_list()
    if len(image_shape)==4:
        image = tf.reshape(image,(image_shape[0],-1,3))
    
    for ii in range(len(theta)):
        if ii<len(theta)-1:
            act=True
            bn = False
        else:
            act=False
            bn = False
        in_size = image.get_shape().as_list()[-1]
        print('layer '+str(ii)+' size = ' + str(in_size) +' out size='+str(theta[ii]['w']))
        image = cell_2d(image,   'l'+str(ii),mode_node,in_size,theta[ii]['w'],act=act,normalize=False,bn=bn) 
    sdf = image
    if len(image_shape)==4:
        sdf = tf.reshape(sdf,(1,image_shape[1],image_shape[2]))    

#    grads = tf.gradients(sdf,xyz)[0]
#    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=2,keep_dims=True))
#    sdf = sdf/grads_norm

    return sdf






    
def deep_sdf3(in_, mode_node, theta):

    image_shape = in_.get_shape().as_list()
    if len(image_shape)==4:
        in_ = tf.reshape(in_,(image_shape[0],-1,3))    
    embeddings = tf.reshape(in_[:,:,3:],(1,-1,64,4))
    xyz        = in_[:,:,0:3]
    spheres = []
    for shape in range(64):
        theta = []
        theta.append(embeddings[:,:,shape,0:3])
        theta.append(embeddings[:,:,shape,3:4])
        spheres.append(SDF.sphere_net(xyz,theta))
    merged = tf.concat(spheres,axis=-1)
    sdf = tf.reduce_min(merged,axis=-1,keepdims=True)
   
    if len(image_shape)==4:
        sdf = tf.reshape(sdf,(1,image_shape[1],image_shape[2]))    

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
    dy_dx   = []
    d2y_dx2 = []
    for ii in range(shape[0]):
        dydx   = tf.gradients(response[ii,:,:],samples)[0]
        d2ydx2 = tf.gradients(dydx,samples)[0]
        dy_dx.append(dydx)     
        d2y_dx2.append(d2ydx2)     
    dy_dx = tf.concat(dy_dx,axis=0) 
    d2y_dx2 = tf.concat(d2y_dx2,axis=0) 
    dy_dx_n = tf.norm(dy_dx,axis=-1,keepdims=True)
    mask    = tf.cast(tf.greater(response,0.),tf.float32)
    evals = {'x':samples,'y':response,'dydx':dy_dx,'d2ydx2':d2y_dx2,'dydx_norm':dy_dx_n,'mask':mask}
    return evals
        


