import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])
from model_ops import cell1D,CONV3D,cell3D_res,cell2D_res,cell_deconv_3D,cell3D_res_regular,cell3D,cell3D_res_deconv,CONV2D,BatchNorm,cell3D_regular,cell3D_res_deconv_regular,cell3D_res_gated,reg_etai



def mydropout(mode_node, x, prob):
  # TODO: test if this is a tensor scalar of value 1.0

    return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
    
#%% DETECTION
        

def volumetric_softmax(node,name):
#    T = tf.get_variable(name='Temperature', initializer = [1.],trainable = False, dtype='float32')
#    node = node/T
    sums = tf.reduce_sum(tf.exp(node), axis=[1,2,3], keep_dims=True)
    softmax = tf.divide(tf.exp(node),sums,name=name)
    return softmax
    


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
    
    
def pose_discriminator(point_cloud, point_cloud_gt, mode_node, batch_size, grid_size_gt):
    point_cloud = tf.reshape(point_cloud,(batch_size,-1))
    point_cloud_gt = tf.reshape(point_cloud_gt,(batch_size,-1))
    point_cloud_batch = tf.concat((point_cloud,point_cloud_gt),axis=0)
    labels = tf.concat((tf.zeros([batch_size,], tf.int32),tf.ones([batch_size,], tf.int32)),axis=0)
    current = tf.nn.dropout(cell1D(point_cloud_batch,16, mode_node, SCOPE='fc1', with_act=True) ,1.0)
#    current = tf.nn.dropout(cell1D(current          ,16, mode_node, SCOPE='fc2', with_act=True) ,1.0)
#    current = tf.nn.dropout(cell1D(current          ,16, mode_node, SCOPE='fc3', with_act=False),1.0)
    current = cell1D(current,2, mode_node, SCOPE='logits', with_act=False, with_bn=False)
    batch = {}
#    batch['logits_g'] = tf.slice(current,[0,0],[batch_size,-1])
#    batch['logits_d'] = tf.slice(current,[batch_size,0],[batch_size,-1])
    batch['logits'] = current
    batch['labels'] = labels
    return batch
    
    
         
    

def model_1(data_node, mode_node, batch_size, vox_params):
    """The Model definition."""
    ch_in = data_node.get_shape().as_list()[-1]
    etaiFlag = True
#    mean, var = tf.nn.moments(data_node, axes=[0],keep_dims=True)  
#    data_node = (data_node-mean)/tf.sqrt(var)
#    data_node = tf.stop_gradient(data_node)
    
    with tf.variable_scope("input"):
        print(data_node.get_shape())
        conv1_w, conv1_b = CONV3D([5,5,5,ch_in,128])
        conv1 = tf.nn.conv3d(data_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(conv1, conv1_b)
        if etaiFlag:
            with tf.variable_scope('reg'):
                reg_etai(current,10,eps_init=vox_params['eps'])              
        print(current.get_shape())
    with tf.variable_scope("residuals"):     
        num_layers = len(vox_params['filter'])
        for layer in range(num_layers):
            current = cell3D_res(current, 3, vox_params['filter'][layer], mode_node, vox_params['downsample'][layer], 'layer_'+str(layer) ,bn=False,etaiFlag=etaiFlag,eps_init=vox_params['eps'])
        size_out = current.get_shape().as_list()[1]
    with tf.variable_scope("logits"):        
        features = tf.nn.avg_pool3d(current,[1,size_out,size_out,size_out,1],[1,size_out,size_out,size_out,1],padding='SAME')
        features = tf.reshape(features,(batch_size,-1))
        logits = cell1D(features,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
    return logits

def model_2(data_node, mode_node, batch_size, vox_params, in_size=44):
    """The Model definition."""
    ch_in = data_node.get_shape().as_list()[-1]
    with tf.variable_scope("input"):
        print(data_node.get_shape())
        conv1_w, conv1_b = CONV3D([5,5,5,ch_in,128])
        conv1 = tf.nn.conv3d(data_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(conv1, conv1_b)
        print(current.get_shape())
    with tf.variable_scope("residuals"):     
        num_layers = len(vox_params['filter'])
        for layer in range(num_layers):
            current = cell3D_res_gated(current, 3, vox_params['filter'][layer], mode_node, vox_params['downsample'][layer], 'layer_'+str(layer))
        size_out = current.get_shape().as_list()[1]
    with tf.variable_scope("logits"):        
        features = tf.nn.avg_pool3d(current,[1,size_out,size_out,size_out,1],[1,size_out,size_out,size_out,1],padding='SAME')
        features = tf.reshape(features,(batch_size,-1))
        logits = cell1D(features,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
    return logits



def model_3(data_node, mode_node, batch_size, vox_params, in_size=44):
    """The Model definition."""
    ch_in = data_node.get_shape().as_list()[-1]
    with tf.variable_scope("input"):
        print(data_node.get_shape())
        conv1_w, conv1_b = CONV3D([2,2,2,ch_in,256])
        conv1 = tf.nn.conv3d(data_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        current = tf.nn.bias_add(conv1, conv1_b)
        print(current.get_shape())
    with tf.variable_scope("residuals"):     
#        for layer in range(4):
        current = cell3D_res(current, 2, 256, mode_node, 1, 'layer_'+str(1))
        current = cell3D_res(current, 2, 256, mode_node, 1, 'layer_'+str(2))
        current = cell3D_res(current, 2, 256, mode_node, 1, 'layer_'+str(3))
        current = cell3D_res(current, 2, 256, mode_node, 1, 'layer_'+str(4))
        current = cell3D_res(current, 2, 256, mode_node, 2, 'layer_'+str(5))
        
        
        size_out = current.get_shape().as_list()[1]
    with tf.variable_scope("logits"):        
        features = tf.nn.avg_pool3d(current,[1,2,2,2,1],[1,2,2,2,1],padding='SAME')
        features = tf.reshape(current,(batch_size,-1))
        logits = cell1D(features,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
    return logits






def resnet(data_node, mode_node,layers):
    shape = data_node.get_shape().as_list()
    weights = []

    ch_in = shape[-1]
    data_node_wrap = tf.concat((data_node,tf.slice(data_node,[0,0,0,0],[-1,-1,1,-1])),axis=2)
#    data_node_wrap = tf.concat((data_node_wrap,tf.slice(data_node_wrap,[0,0,0,0],[-1,1,-1,-1])),axis=1)
    with tf.variable_scope("input") as SCOPE:
        conv1_w, conv1_b = CONV2D([2,2,ch_in,32])
        c1 = tf.nn.conv2d(data_node_wrap,conv1_w,strides=[1, 1, 1, 1],padding='VALID')
        c1 = tf.nn.bias_add(c1, conv1_b)
    with tf.variable_scope("residuals") as SCOPE:
        current = cell2D_res(c1,      3, 32, 32, mode_node, 2, 'r1')
        current = cell2D_res(current, 3, 32, 64, mode_node, 2, 'r2')
        current = cell2D_res(current, 3, 64, 64, mode_node, 2, 'r3')
        current = cell2D_res(current, 3, 64, 128, mode_node, 2, 'r4')
        current = cell2D_res(current, 3, 128, 256, mode_node, 2, 'r5')
        current = BatchNorm(current,mode_node,SCOPE)
        current = tf.nn.avg_pool(current,[1,7,7,1],[1,1,1,1],padding='SAME')
    with tf.variable_scope("fully") as SCOPE:
        features = tf.reshape(current,(shape[0],-1))
        for ii in range(len(layers)-1):
            layer_in = layers[ii]
            layer_out = layers[ii+1]
            w = tf.reshape(cell1D(features,layer_in*layer_out, mode_node, SCOPE='w'+str(ii), with_act=False, with_bn=False),(shape[0],layer_in,layer_out) )
            b = tf.reshape(cell1D(features,layer_out, mode_node, SCOPE='b'+str(ii), with_act=False, with_bn=False) ,(shape[0],1,layer_out) )

            weights.append({'w':w,'b':b})
    return weights
#,
#            'w2':w2,'b2':b2,
#            'w3':w3,'b3':b3,
#            'w4':w4,'b4':b4}
