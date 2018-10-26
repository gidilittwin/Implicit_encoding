import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])
from pointnet.models import pointnet_cls as POINT
from model_ops import cell1D,CONV3D,cell3D_res,cell2D_res,cell_deconv_3D,cell3D_res_regular,cell3D,cell3D_res_deconv,CONV2D,BatchNorm,cell3D_regular,cell3D_res_deconv_regular,cell2D_gated
import kinetic as KINETIC



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
    
    
         
    

def model_3(data_node, mode_node, batch_size, in_size=44):
    """The Model definition."""
    ch_in = data_node.get_shape().as_list()[-1]
    with tf.variable_scope("input"):
        print(data_node.get_shape())
        conv1_w, conv1_b = CONV3D([5,5,5,ch_in,16])
        conv1 = tf.nn.conv3d(data_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())
    with tf.variable_scope("residuals"):        
        current1 = cell3D_res(conv1, 2, 16, 16, mode_node, 2, 'downsample1')
        current2 = cell3D_res(current1, 3, 16, 32, mode_node, 1, 'residual1')
        current3 = cell3D_res(current2, 3, 32, 32, mode_node, 1, 'residual2')
        current4 = cell3D_res(current3, 3, 32, 32, mode_node, 1, 'residual3')
    with tf.variable_scope("encoder"):
        current5 = cell3D_res(current4, 2, 32, 32, mode_node, 2, 'downsample2')
        current6 = cell3D_res(current5, 3, 32, 64, mode_node, 1, 'residual4')
        current7 = cell3D_res(current6, 2, 64, 64, mode_node, 2, 'downsample3')
        current8 = cell3D_res(current7, 3, 64, 128, mode_node, 1, 'residual5')
    with tf.variable_scope("decoder"):
        current9 = cell3D_res(current8, 3, 128, 128, mode_node, 1, 'residual6')
        current10 = cell3D_res_deconv(current9, 2, 128, 64, mode_node, 2,[batch_size,in_size/4,in_size/4,in_size/4,64], 'upsampling1')
        current10 = current10 + cell3D_res(current6, 3, 64, 64, mode_node, 1, 'skip1')
        current11 = cell3D_res(current10, 3, 64, 64, mode_node, 1, 'residual7')
        current12 = cell_deconv_3D(current11, 2, 64, 32,[batch_size,in_size/2,in_size/2,in_size/2,32], mode_node, 2, 'upsampling2')
        current12 = current12 + cell3D_res(current4, 3, 32, 32, mode_node, 1, 'skip2')
    with tf.variable_scope("output"):
        current = cell3D_res(current12, 3, 32, 21, mode_node, 1, 'residual8')
#        current = cell_deconv_3D(current, 2, 32, 21,[batch_size,in_size,in_size,in_size,21], mode_node, 2, 'upsampling3')
        current = cell3D(current, 1, 21, 21, mode_node, 1, 'basic2')
        logits = cell3D(current, 1, 21, 21, mode_node, 1, 'basic3')
    return logits



def model_3_3(data_node,in_node2, mode_node, batch_size, in_size=44,grid_size_gt=44):
    """The Model definition."""
    ch_in = data_node.get_shape().as_list()[-1]
    with tf.variable_scope("input"):
        print(data_node.get_shape())
        conv1 = cell3D_regular(data_node, 1, ch_in, 32, mode_node, 1, 'basic')
        conv1 = tf.nn.max_pool3d(conv1,[1,2,2,2,1],[1,2,2,2,1],padding='SAME')
    with tf.variable_scope("residuals"):        
        current2 = cell3D_res_regular(conv1,    1, 32, 32, mode_node, 1, 'residual1')
        current3 = cell3D_res_regular(current2, 1, 32, 32, mode_node, 1, 'residual2')
        current3 = tf.nn.max_pool3d(current3,[1,2,2,2,1],[1,2,2,2,1],padding='SAME')
        current4 = cell3D_res_regular(current3, 1, 32, 64, mode_node, 1, 'residual3')
    with tf.variable_scope("encoder"):
        current5 = cell3D_res_regular(current4, 1, 64, 64, mode_node, 1, 'downsample2')
        current5 = tf.nn.max_pool3d(current5,[1,2,2,2,1],[1,2,2,2,1],padding='SAME')
        current6 = cell3D_res_regular(current5, 1, 64, 64, mode_node, 1, 'residual4')
        current7 = cell3D_res_regular(current6, 1, 64, 64, mode_node, 1, 'downsample3')
        current7 = tf.nn.max_pool3d(current7,[1,2,2,2,1],[1,2,2,2,1],padding='SAME')
        current8 = cell3D_res_regular(current7, 1, 64, 128, mode_node, 1, 'residual5')
    with tf.variable_scope("decoder"):
        current9 = cell3D_res_regular(current8, 1, 128, 128, mode_node, 1, 'residual6')
        current10 = cell3D_res_deconv_regular(current9, 1, 128, 64, mode_node, 2,[batch_size,in_size/8,in_size/8,in_size/8,64], 'upsampling1')
        current10 = current10 + cell3D_res_regular(current6, 1, 64, 64, mode_node, 1, 'skip1')
        current11 = cell3D_res_regular(current10, 1, 64, 64, mode_node, 1, 'residual7')
        current12 = cell3D_res_deconv_regular(current11, 1, 64, 64, mode_node, 2,[batch_size,in_size/4,in_size/4,in_size/4,64], 'upsampling2')
        current12 = current12 + cell3D_res_regular(current4, 1, 64, 64, mode_node, 1, 'skip2')
    with tf.variable_scope("output"):
        current = cell3D_res_regular(current12, 1, 64, 32, mode_node, 1, 'residual8')
        current = cell_deconv_3D(current, 1, 32, 21,[batch_size,in_size/2,in_size/2,in_size/2,21], mode_node, 2, 'upsampling3')
        current = cell3D_regular(current, 1, 21, 21, mode_node, 1, 'basic2')
        logits = cell3D_regular(current, 1, 21, 21, mode_node, 1, 'basic3', act=False)
    return logits, logits




def bone_logits(current):
    with tf.variable_scope("bones"):
        bones_logits = []
        for i in np.arange(0,skeleton.shape[0]):
            bones = skeleton[i,:]
            for jj in np.arange(1,bones.shape[0]):
                bone_coordinate1 = tf.slice(current,[0,bones[jj-1],0],[-1,1,-1])
                bone_coordinate2 = tf.slice(current,[0,bones[jj],0],[-1,1,-1])
                bone_vector = bone_coordinate1-bone_coordinate2
                bone_length = tf.sqrt(tf.reduce_sum(tf.pow(bone_vector,2),axis=2,keep_dims=True))
                bones_logits.append(bone_length)
        bones_logits = tf.concat(bones_logits,axis=1)
    return bones_logits



def deep_prior(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    in_node = tf.reduce_max(in_node,axis=3,keep_dims=True)
    in_node = tf.slice(in_node,[0,0,0,0,2],[-1,-1,-1,-1,1])
    in_node = tf.squeeze(in_node,axis=3)
#    in_node_max = tf.reduce_max(in_node,axis=(1,2,3),keep_dims=True) 
#    in_node = (tf.div(in_node,in_node_max) - 0.5)*2
    ch_in = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("input") as SCOPE:
        conv1_w, conv1_b = CONV2D([5,5,ch_in,32])
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
    with tf.variable_scope("residuals") as SCOPE:
        current = cell2D_res(c1,      3, 32, 64, mode_node, 2, 'r1')
        current = cell2D_res(current, 3, 64, 64, mode_node, 1, 'r2')
        current = cell2D_res(current, 3, 64, 128, mode_node, 2, 'r3')
        current = cell2D_res(current, 3, 128, 128, mode_node, 1, 'r4')
        current = cell2D_res(current, 3, 128, 256, mode_node, 2, 'r5')
        current = cell2D_res(current, 3, 256, 256, mode_node, 1, 'r6')
        current = cell2D_res(current, 3, 256, 512, mode_node, 2, 'r7')
        current = BatchNorm(current,mode_node,SCOPE)
        current = tf.nn.relu(current)
    with tf.variable_scope("fully") as SCOPE:
        features = tf.reshape(current,(batch_size,-1))
#        bones = tf.stop_gradient(tf.squeeze(bone_logits(in_node2),-1))
#        features = tf.concat((features,bones),axis=1)
        current = mydropout(mode_node,cell1D(features,1024, mode_node, SCOPE='fc1', with_act=True, with_bn=False) ,1.0)
        current = mydropout(mode_node,cell1D(current ,1024, mode_node, SCOPE='fc2', with_act=True, with_bn=False) ,1.0)
        current = cell1D(current,63, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
        marks = tf.tanh(tf.reshape(current,(batch_size,3,21)))
#        marks = tf.reshape(current,(batch_size,3,21))
        marks_norm = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy") as SCOPE:
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
        pred_cloud = tf.transpose(marks,(0,2,1))
    return logits, pred_cloud

def deep_prior6(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    in_node = tf.reduce_max(in_node,axis=3,keep_dims=True)
#    in_node = tf.slice(in_node,[0,0,0,0,0],[-1,-1,-1,-1,3])
    in_node = tf.squeeze(in_node,axis=3)*2-1
#    in_node_max = tf.reduce_max(in_node,axis=(1,2,3),keep_dims=True) 
#    in_node = (tf.div(in_node,in_node_max) - 0.5)*2
    ch_in = in_node.get_shape().as_list()[-1]
    k_size = 1
    with tf.variable_scope("input") as SCOPE:
        conv1_w, conv1_b = CONV2D([k_size,k_size,ch_in,32])
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
    with tf.variable_scope("residuals") as SCOPE:
        current = cell2D_res(c1,      k_size, 32, 64, mode_node, 1, 'r1')
        current = tf.nn.max_pool(current,[1,2,2,1],[1,2,2,1],padding='SAME')
        current = cell2D_res(current, k_size, 64, 64, mode_node, 1, 'r2')
        current = cell2D_res(current, k_size, 64, 128, mode_node, 1, 'r3')
        current = tf.nn.max_pool(current,[1,2,2,1],[1,2,2,1],padding='SAME')
        current = cell2D_res(current, k_size, 128, 128, mode_node, 1, 'r4')
        current = cell2D_res(current, k_size, 128, 256, mode_node, 1, 'r5')
        current = tf.nn.max_pool(current,[1,2,2,1],[1,2,2,1],padding='SAME')
        current = cell2D_res(current, k_size, 256, 256, mode_node, 1, 'r6')
        current = cell2D_res(current, k_size, 256, 1024, mode_node, 1, 'r7')
        current = tf.nn.max_pool(current,[1,2,2,1],[1,2,2,1],padding='SAME')
        current = BatchNorm(current,mode_node,SCOPE)
        current = tf.nn.max_pool(current,[1,6,6,1],[1,1,1,1],padding='SAME')
    with tf.variable_scope("fully") as SCOPE:
        features = tf.reshape(current,(batch_size,-1))
        current = cell1D(features,63, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
        marks = tf.tanh(tf.reshape(current,(batch_size,3,21)))
        pred_cloud = tf.transpose(marks,(0,2,1))
#        z_bias = tf.slice(tf.reduce_mean(in_node2,axis=1,keep_dims=True),[0,0,2],[-1,-1,1])
#        z_bias = tf.concat((z_bias*0,z_bias*0,z_bias),axis=2)
#        pred_cloud = pred_cloud+z_bias
        marks_norm = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy") as SCOPE:
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
    return logits, pred_cloud






def deep_prior4(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    in_node = tf.reduce_max(in_node,axis=3,keep_dims=True)
    in_node = tf.slice(in_node,[0,0,0,0,2],[-1,-1,-1,-1,1])
    in_node = tf.squeeze(in_node,axis=3)
#    in_node_max = tf.reduce_max(in_node,axis=(1,2,3),keep_dims=True) 
#    in_node = (tf.div(in_node,in_node_max) - 0.5)*2
    ch_in = in_node.get_shape().as_list()[-1]
    num_params = 63
    with tf.variable_scope("input") as SCOPE:
        conv1_w, conv1_b = CONV2D([5,5,ch_in,32])
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
    with tf.variable_scope("residuals") as SCOPE:
        current = cell2D_res(c1,      3, 32, 64, mode_node, 2, 'r1')
        current = cell2D_res(current, 3, 64, 64, mode_node, 1, 'r2')
        current = cell2D_res(current, 3, 64, 128, mode_node, 2, 'r3')
        current = cell2D_res(current, 3, 128, 128, mode_node, 1, 'r4')
        current = cell2D_res(current, 3, 128, 256, mode_node, 2, 'r5')
        current = cell2D_res(current, 3, 256, 256, mode_node, 1, 'r6')
        current = cell2D_res(current, 3, 256, 512, mode_node, 2, 'r7')
        current = BatchNorm(current,mode_node,SCOPE)
        current = tf.nn.avg_pool(current,[1,6,6,1],[1,1,1,1],padding='SAME')
    with tf.variable_scope("fully") as SCOPE:
        features = tf.reshape(current,(batch_size,-1))
        current = cell1D(features,num_params, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
        marks = tf.tanh(tf.reshape(current,(batch_size,3,21)))
        pred_cloud = tf.transpose(marks,(0,2,1))
        marks_norm = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy") as SCOPE:
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
    return logits, pred_cloud



def deep_prior4_IG(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    in_node = tf.reduce_max(in_node,axis=3,keep_dims=True)
    in_node = tf.slice(in_node,[0,0,0,0,2],[-1,-1,-1,-1,1])
    in_node = tf.squeeze(in_node,axis=3)
    ch_in = in_node.get_shape().as_list()[-1]

    with tf.variable_scope("Input") as SCOPE:
        conv1_w, conv1_b = CONV2D([5,5,ch_in,32])
        c1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        c1 = tf.nn.bias_add(c1, conv1_b)
    with tf.variable_scope("Residuals") as SCOPE:
        current = cell2D_res(c1,      3, 32, 64, mode_node, 2, 'r1')
        current = cell2D_res(current, 3, 64, 64, mode_node, 1, 'r2')
        current = cell2D_res(current, 3, 64, 128, mode_node, 2, 'r3')
        current = cell2D_res(current, 3, 128, 128, mode_node, 1, 'r4')
        current = cell2D_res(current, 3, 128, 256, mode_node, 2, 'r5')
        current = cell2D_res(current, 3, 256, 256, mode_node, 1, 'r6')
        current = cell2D_res(current, 3, 256, 512, mode_node, 2, 'r7')
        current = BatchNorm(current,mode_node,SCOPE)
        current = tf.nn.avg_pool(current,[1,6,6,1],[1,1,1,1],padding='SAME')
    with tf.variable_scope("Features") as SCOPE:
        features = tf.reshape(current,(batch_size,-1))

    with tf.variable_scope("Kinematics") as SCOPE:
        num_joints = 21
        KL         = KINETIC.Kinetic(batch_size, num_joints = num_joints)
        with tf.variable_scope("Bones_GT") as SCOPE:
            bones_gt = bone_logits(in_node2)  
            bones_gt = tf.transpose(bones_gt,(0,2,1))
            bones_gt_scale = tf.div(bones_gt,tf.slice(KL.model['Bones'],[0,0,1],[-1,1,-1]))
            bones_gt_scale = tf.concat((tf.zeros((batch_size,1,1)),bones_gt_scale),axis=2)
            
        pose_limits_scale = 180*np.ones((1,3,21),dtype=np.float32)
        # X axis
        pose_limits_scale[:,0,(4,8,12,16,20)] = 5   # Tip bones
        pose_limits_scale[:,0,(3,7,11,15,19)] = 5   #
        pose_limits_scale[:,0,(2,6,10,14,18)] = 40  # Most of the rotational freedom
        pose_limits_scale[:,0,(1,5, 9,13,17)] = 30  # Metacarpal
        # Z axis
        pose_limits_scale[:,2,(4,8,12,16,20)] = 2   # Tip bones
        pose_limits_scale[:,2,(3,7,11,15,19)] = 2
        pose_limits_scale[:,2,(2,6,10,14,18)] = 50  # Most of the rotational freedom
        pose_limits_scale[:,2,(1,5, 9,13,17)] = 30  # Metacarpal
        # Y axis
        pose_limits_scale[:,1,(4,8,12,16,20)] = 80 # Tip bones
        pose_limits_scale[:,1,(3,7,11,15,19)] = 80
        pose_limits_scale[:,1,(2,6,10,14,18)] = 80 # Most of the rotational freedom
        pose_limits_scale[:,1,(1,5, 9,13,17)] = 50  # Metacarpal
        # Wrist
        pose_limits_scale[:,:,0] = 190
        
        pose_limits_scale_tf = tf.constant(pose_limits_scale/180.,dtype=tf.float32) 
        pose   = pose_limits_scale_tf*tf.tanh(tf.reshape(cell1D(features,63, mode_node, SCOPE='Pose', with_act=False, with_bn=False),(batch_size,3,21)) )
#        bones  = 1. + 0.5*tf.tanh(tf.expand_dims(cell1D(features,21, mode_node, SCOPE='Bones', with_act=False, with_bn=False),1))
        camera = tf.tanh(tf.expand_dims(cell1D(features,3, mode_node, SCOPE='Camera', with_act=False, with_bn=False),-1))
        pred_cloud,_    = KL.apply_trans(bones_gt_scale,pose,camera)
        marks           = tf.transpose(pred_cloud,(0,2,1))
        marks_norm      = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy") as SCOPE:
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
    return logits, pred_cloud




def fc(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    with tf.variable_scope('fully'):
        current = cell1D(in_node,512, mode_node, SCOPE='fc1', with_act=True, with_bn=False)
        current = cell1D(current ,256, mode_node, SCOPE='fc2', with_act=True, with_bn=False)
        current = cell1D(current,63, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
        marks = tf.tanh(tf.reshape(current,(batch_size,3,21)))
        marks_norm = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy"):
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
        pred_cloud = tf.transpose(marks,(0,2,1))
    return logits, pred_cloud


def fc_IG(in_node,in_node2, mode_node, batch_size, grid_size, grid_size_gt):
    with tf.variable_scope('fully'):
        current = cell1D(in_node,512, mode_node, SCOPE='fc1', with_act=True, with_bn=False)
        features = cell1D(current ,512, mode_node, SCOPE='fc2', with_act=True, with_bn=False)
    with tf.variable_scope("Kinematics"):
        num_joints = 21
        pose_limits_init = np.ones((1,3,21),dtype=np.float32)
        pose_limits_init[:,2,:] = 0.3*pose_limits_init[:,2,:] #Z rotations
        pose_limits_init[:,1,:] = 0.7*pose_limits_init[:,1,:] # 
        pose_limits_init[:,0,:] = 0.2*pose_limits_init[:,0,:]
        pose_limits_init[:,:,0] = 1
        pose_limits = tf.constant(pose_limits_init,dtype=tf.float32)        
        bones  = 0.5*tf.tanh(tf.expand_dims(cell1D(features,21, mode_node, SCOPE='Bones', with_act=False, with_bn=False),1))+1
        pose   = pose_limits*tf.tanh(tf.reshape(cell1D(features,63, mode_node, SCOPE='Pose', with_act=False, with_bn=False),(batch_size,3,21)) )
        camera = tf.tanh(tf.expand_dims(cell1D(features,3, mode_node, SCOPE='Camera', with_act=False, with_bn=False),-1))
        KL                = KINETIC.Kinetic(batch_size, num_joints = num_joints)
        pred_cloud,_    = KL.apply_trans(bones,pose,camera)
        marks           = tf.transpose(pred_cloud,(0,2,1))
        marks_norm = (marks/2+0.5)*(grid_size_gt-1)
    with tf.variable_scope("xentropy"):
        gt_meshgrid = np.meshgrid(np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt),np.linspace(0, grid_size_gt-1, grid_size_gt))
        grid_x = (gt_meshgrid[1]).astype(np.float32)
        grid_y = (gt_meshgrid[0]).astype(np.float32)
        grid_z = (gt_meshgrid[2]).astype(np.float32)
        grid_x_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='x', initializer = grid_x,trainable = False, dtype='float32'),0),-1)
        grid_y_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='y', initializer = grid_y,trainable = False, dtype='float32'),0),-1)
        grid_z_tf = tf.expand_dims(tf.expand_dims(tf.get_variable(name='z', initializer = grid_z,trainable = False, dtype='float32'),0),-1)
        pred_x = grid_x_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,0,0],[-1,1,-1]),1),1)
        pred_y = grid_y_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,1,0],[-1,1,-1]),1),1)
        pred_z = grid_z_tf - tf.expand_dims(tf.expand_dims(tf.slice(marks_norm,[0,2,0],[-1,1,-1]),1),1)
        logits = -1*(tf.pow(pred_x,2) + tf.pow(pred_y,2) + tf.pow(pred_z,2))
        init_scale = np.zeros((1,1,1,1,21),dtype='float32')
        init_bias = np.zeros((1,1,1,1,21),dtype='float32')
#        init_scale = np.zeros((1),dtype='float32')
#        init_bias = np.zeros((1),dtype='float32')
        scale = tf.get_variable(name='scale', initializer = init_scale,trainable = True, dtype='float32')
        bias  = tf.get_variable(name='bias', initializer = init_bias,trainable = True, dtype='float32')
        logits = logits*scale + bias
        pred_cloud = tf.transpose(marks,(0,2,1))
    return logits, pred_cloud

