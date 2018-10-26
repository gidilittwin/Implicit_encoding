import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
#from pointnet.models import pointnet_cls as POINT
from model_ops import cell1D,CONV3D,cell3D_res,cell2D_res,cell_deconv_3D,cell3D_res_regular,cell3D,cell3D_res_deconv,CONV2D,BatchNorm,cell3D_regular,cell3D_res_deconv_regular
from model_ops import ravel_index,unravel_index,cell2D, cell2D_gated ,conv_layer_etai, reg_etai



   
    #%% POINT NETS 
class POINTNET(object):
    def __init__(self, num_points=2048,grid_size=40,batch_size=16,rep_size=128):
        self.num_points = num_points
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.rep_size = rep_size
        self.grid_shape    = tf.constant([self.batch_size,self.grid_size,self.grid_size,self.grid_size],dtype=tf.int64)
        self.voxel_shape = tf.constant([self.batch_size,self.grid_size,self.grid_size,self.grid_size,rep_size],dtype=tf.int32)
        self.batch_idx_np = np.expand_dims(np.meshgrid(np.arange(0, self.batch_size, 1),np.arange(0, self.num_points, 1))[0].transpose(),axis=2)
        self.batch_idx = tf.reshape(tf.convert_to_tensor(self.batch_idx_np.astype(np.float32)),(self.batch_size*self.num_points,1))

    def build(self,inputs,mode_node,vox_params):
        # POINTS
        point_cloud = inputs[0]
        # Voxels
        if self.rep_size==128:
            
            
            
            
#            # BEST ETAI (elu) 87.5%  relu (87.3% )  train 96.7%
#            activation = tf.nn.elu
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
##            point_cloud_rot,rot_mat      = self.transform_points(point_cloud,mode_node,scope='T1')
##            T1                    = tf.expand_dims(point_cloud_rot,-1)
##            mlp1                  = self.mlp_etai(point_cloud_rot,[64,64],mode_node,act_type=activation, eps=eps, scope='mlp1')
##            T2                    = self.transform_features(mlp1,mode_node,scope='T2')
#            mlp2                  = self.mlp_etai(point_cloud_rot,[64,128,1024],mode_node,scope='mlp1',prob=1,act_type=activation, eps_init=vox_params['eps'])
#            pooled                = tf.reduce_max(mlp2,axis=1,keep_dims=True)
#            mlp3                  = self.mlp_etai(pooled,[512,256],mode_node,scope='mlp2',prob=[0.8,0.8],act_type=activation, eps_init=vox_params['eps'])
#            features_vox          = tf.squeeze(mlp3,axis=(1,2))
#            features_vox          = cell1D(features_vox,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
#            voxels                = features_vox



##            # Etai baseline 87.86%   train 97.8%
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
##            point_cloud_rot,rot_mat      = self.transform_points(point_cloud,mode_node,scope='T1')
##            T1                    = tf.expand_dims(point_cloud_rot,-1)
##            mlp1                  = self.mlp_etai(point_cloud_rot,[64,64],mode_node,act_type=activation, eps=eps, scope='mlp1')
##            T2                    = self.transform_features(mlp1,mode_node,scope='T2')
#            mlp2                  = self.mlp(point_cloud_rot,[64,128,1024],mode_node,scope='mlp2')
#            pooled                = tf.reduce_max(mlp2,axis=1,keep_dims=True)
#            mlp3                  = self.mlp(pooled,[512,256],mode_node,scope='mlp3',prob=[0.8,0.8])
#            features_vox          = tf.squeeze(mlp3,axis=(1,2))
#            features_vox          = cell1D(features_vox,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
#            voxels                = features_vox
            
            
            
            
            
            
            
            
            
            
            
            
            
            
           # BEST yet 88.63 97.5
            act_type = tf.nn.relu
            scale_ratio = 0.9
            point_cloud       = tf.expand_dims(point_cloud,-1)
            
            rot_mat           = tf.expand_dims(self.rotations(),axis=1)
            point_cloud_tiled = tf.tile(point_cloud,(1,1,1,3))
            point_cloud_rot   = tf.reduce_sum(point_cloud_tiled*rot_mat,axis=2)            
            point_cloud_rot   = tf.expand_dims(point_cloud_rot,-1)
            scale = tf.random_uniform((self.batch_size,1,1,1),scale_ratio,1/scale_ratio)
            point_cloud_rot_scaled  = point_cloud_rot*scale

            with tf.variable_scope('main_branch') as scope:
                mlp1                  = self.permute_eq(point_cloud,[128,256,512,1024],mode_node,scope='mlp1',act_type=act_type)
                pooled_1              = tf.reduce_max(mlp1,axis=1,keep_dims=True)
                mlp2                  = self.mlp(pooled_1,[512,256],mode_node,scope='mlp2',prob=[0.6,0.5],act_type=act_type)
                
                scope.reuse_variables()
                mlp1_                 = self.permute_eq(point_cloud_rot_scaled,[128,256,512,1024],mode_node,scope='mlp1',act_type=act_type)
                pooled_1_             = tf.reduce_max(mlp1_,axis=1,keep_dims=True)            
#                mlp2_                 = self.mlp(pooled_1_,[512,256],mode_node,scope='mlp2',prob=[0.6,1],act_type=act_type)

                diff = self.correlation_loss(pooled_1, pooled_1_)
                tf.add_to_collection('rot_inv',diff)    
#                diff2 = self.correlation_loss(mlp2, mlp2_)
#                tf.add_to_collection('rot_inv',diff2)   
                
            features_vox          = tf.squeeze(mlp2,axis=(1,2))
            features_vox          = cell1D(features_vox,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
            voxels = features_vox
            count = features_vox          
            
            
            
            
            

##            # BEST yet 87.3 96.2
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
#            mlp1                  = self.permute_eq(point_cloud_rot,[128,256,1024],mode_node,scope='mlp1',act_type='relu')
#            pooled_1              = tf.reduce_max(mlp1,axis=1,keep_dims=True)
#            mlp2                  = self.mlp(pooled_1,[512,256],mode_node,scope='mlp2',prob=[0.9,0.9],act_type='relu')
#            features_vox          = tf.squeeze(mlp2,axis=(1,2))
#            features_vox          = cell1D(features_vox,40, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
#            voxels = features_vox
#            count = features_vox
            

##            # BEST yet 85 90
#            set_size = 3
#            num_centers = set_size*set_size*set_size
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
#            mlp1                  = self.permute_eq(point_cloud_rot,[128,256,1024],mode_node,scope='mlp1',act_type='relu')
#            pooled_1              = tf.reduce_max(mlp1,axis=1,keep_dims=True)
#            mlp2                  = self.mlp(pooled_1,[512,512],mode_node,scope='mlp2',prob=[0.9,0.9],act_type='relu')
#            centers          = tf.squeeze(mlp2,axis=(1,2))
#            centers          = cell1D(centers,num_centers*3, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
#            centers = tf.reshape(centers,(self.batch_size,1,3,num_centers))
#            point_cloud_aug  = point_cloud_rot-centers
#            features_vox = []
#            with tf.variable_scope('mini_nets'): #, reuse=tf.AUTO_REUSE
#                for ii in np.arange(0,num_centers):
#                    with tf.variable_scope('mini_net_'+str(ii)):
#                        points          = tf.slice(point_cloud_aug,[0,0,0,ii],[-1,-1,-1,1])
#                        points_         = tf.pow(points,2)
#                        points          = tf.concat((points,points_),axis=2)                    
#                        mlp1_mini       = self.permute_eq(points,[256,256],mode_node,scope='mlp1_mini',act_type='relu')
#                        pooled_1_mini   = tf.reduce_max(mlp1_mini,axis=1,keep_dims=True)
#                        mlp2_mini       = self.mlp(pooled_1_mini,[128],mode_node,scope='mlp2_mini',prob=1,act_type='relu')            
#                        features_vox.append(mlp2_mini)
#            features_vox = tf.reshape(tf.concat(features_vox,axis=1),(self.batch_size,set_size,set_size,set_size,-1) )       
#            voxels = centers
#            count = centers

            
            
            
# #            # BEST yet 87 92
#            set_size = 3
#            num_centers = set_size*set_size*set_size
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
#            mlp1                  = self.permute_eq(point_cloud_rot,[128,256,1024],mode_node,scope='mlp1',act_type='relu')
#            pooled_1              = tf.reduce_max(mlp1,axis=1,keep_dims=True)
#            mlp2                  = self.mlp(pooled_1,[512,512],mode_node,scope='mlp2',prob=[0.9,0.9],act_type='relu')
#            centers          = tf.squeeze(mlp2,axis=(1,2))
#            centers          = cell1D(centers,num_centers*3, mode_node, SCOPE='logits', with_act=False, with_bn=False)             
#            centers = 3*tf.tanh(tf.reshape(centers,(self.batch_size,1,3,num_centers)))
#            point_cloud_aug  = point_cloud_rot-centers
#            features_vox = []
#            with tf.variable_scope('mini_nets'): #, reuse=tf.AUTO_REUSE
#                for ii in np.arange(0,num_centers):
#                    with tf.variable_scope('mini_net_'+str(ii)):
#                        points          = tf.slice(point_cloud_aug,[0,0,0,ii],[-1,-1,-1,1])
#                        r               = tf.reduce_sum(tf.pow(points,2),axis=2,keep_dims=True)
#                        points          = tf.concat((points,r),axis=2)
#                        mlp1_mini       = self.mlp(points,[54,128,512],mode_node,scope='mlp1_mini',prob=1,act_type='relu')
#                        pooled_1_mini   = tf.reduce_max(mlp1_mini,axis=1,keep_dims=True)
#                        mlp2_mini       = self.mlp(pooled_1_mini,[256],mode_node,scope='mlp2_mini',prob=1,act_type='relu')            
#                        features_vox.append(mlp2_mini)
#            features_vox = tf.reshape(tf.concat(features_vox,axis=1),(self.batch_size,set_size,set_size,set_size,-1) )       
#            voxels = centers
#            count = centers

            
#            point_cloud_rot       = tf.expand_dims(point_cloud,-1)
#            mlp                   = self.mlp(point_cloud_rot,[64,128,512,4096],mode_node,scope='mlp',prob=1,act_type='relu')
#            mlp                   = tf.reduce_max(mlp,axis=1,keep_dims=True)
#            features_vox          = tf.squeeze(mlp,axis=(1,2))
#            voxels                = tf.reshape(features_vox,(self.batch_size,self.grid_size,self.grid_size))*self.grid_size
#            count                 = voxels            
            

        else:
            point_cloud_rot = point_cloud
            indices, idx, count, count_normalized, point_cloud_4d = self.get_centers(point_cloud_rot) 
            features_vox = tf.cast(indices,tf.float32)/(self.grid_size-1)
            features_vox = tf.slice(features_vox,[0,2],[-1,1])
            features_vox = tf.expand_dims(tf.cast(count,dtype=tf.float32),-1)
            features_vox = features_vox/features_vox
            voxels = tf.scatter_nd(indices, features_vox, self.voxel_shape)
            voxels = tf.stop_gradient(voxels)
        
        batch_ = []
        batch_.append(point_cloud)
        batch_.append(point_cloud_rot) 
        batch_.append(features_vox) 
        batch_.append(voxels) 
        return batch_

    def correlation_loss(self,x,y):
        length = tf.cast(tf.shape(x)[3],tf.float32)
        original_loss =   length*tf.reduce_sum(x*y,-1) - (tf.reduce_sum(x,-1) * tf.reduce_sum(y,-1))
        divisor = tf.sqrt(
          (length * tf.reduce_sum(tf.square(x),-1) - tf.square(tf.reduce_sum(x,-1))) *
          (length * tf.reduce_sum(tf.square(y),-1) - tf.square(tf.reduce_sum(y,-1)))
        )
        original_loss = tf.reduce_mean(tf.truediv(original_loss, divisor))
        
        original_loss = 1-tf.pow(original_loss,2)
        return original_loss


    def segment_pool(self,point_cloud,current,grid_size,scope=None):
        with tf.variable_scope(scope):
            point_cloud = (point_cloud+1)/2*grid_size
            point_cloud_4d = tf.concat((self.batch_idx,tf.reshape(point_cloud,(-1,3))),axis=1)
            centers = tf.round(point_cloud_4d)
            point_cloud_idx = tf.cast(centers,tf.int64)
            grid_shape    = tf.constant([self.batch_size,grid_size,grid_size,grid_size],dtype=tf.int64)
            point_cloud_lin_idx = tf.squeeze(ravel_index(point_cloud_idx,grid_shape))
            vox_idx, idx, count = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
            vox_idx = tf.cast(vox_idx,dtype = tf.int32)
            vox_mult_idx = tf.transpose(unravel_index(vox_idx,grid_shape),(1,0))
#            batch_idx = tf.squeeze(tf.slice(vox_mult_idx,(0,0),(-1,1)))
            sh = current.get_shape().as_list()
            features_vox = tf.unsorted_segment_max(tf.reshape(current,(-1,sh[-1])),idx,tf.reduce_max(idx)+1)
            features = tf.gather(features_vox,idx,axis=0)
            features = tf.reshape(features,sh)
#            max_point_per_vol = tf.segment_max(count,batch_idx)
#            max_point_per_vol = tf.gather(max_point_per_vol,batch_idx)
#            count_normalized = tf.divide(count,max_point_per_vol)
        return features,features_vox,vox_mult_idx,count

    def transform_points(self,point_cloud,mode_node,scope=None):
        # Get transformation
        with tf.variable_scope(scope):
            rot_mat = self.point_transformer(point_cloud, mode_node)
            rot_mat_ = tf.expand_dims(rot_mat,axis=1)
            # Apply transformation
            point_cloud_tiled = tf.tile(tf.expand_dims(point_cloud,-1),(1,1,1,3))
            point_cloud_rot = tf.reduce_sum(point_cloud_tiled*rot_mat_,axis=2)
        return point_cloud_rot,rot_mat
    
    def transform_features(self,current,mode_node,scope=None):
        with tf.variable_scope(scope):
            sh = current.get_shape().as_list()
            T = self.feature_transformer(current, sh[-1], mode_node)
            T_ = tf.expand_dims(T,axis=1)
            # Apply transformation
            mlp_tiled = tf.tile(tf.transpose(current,(0,1,3,2)),(1,1,1,sh[-1]))
            mlp_rot = tf.reduce_sum(mlp_tiled*T_,axis=2,keep_dims=True)
        return  mlp_rot

    def mlp(self,current,out_size,mode_node,scope=None, prob = 1, act_type='relu'):
        with tf.variable_scope(scope):
            for layer in np.arange(0,len(out_size)):
                sh = current.get_shape().as_list()
                current = cell2D(current, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=True, act_type=act_type)
                if prob!=1:
                    current = self.mydropout(current,mode_node,prob[layer])
        return current


    def mlp_etai(self,current,out_size,mode_node,scope=None, prob = 1, act_type=tf.nn.elu,eps_init=0.3):
        with tf.variable_scope(scope):
            for layer in np.arange(0,len(out_size)):
                sh = current.get_shape().as_list()
                current = cell2D(current, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=False, act=False)
                with tf.variable_scope('reg_'+str(layer)):
                    reg_etai(current,10,eps_init=eps_init) 
                current = act_type(current)
                if prob!=1:
                    current = self.mydropout(current,mode_node,prob[layer])
        return current
    
    
    
    def gated_mlp(self,current,out_size,mode_node,scope=None, prob = 1, act_type='relu',gate=None):
        with tf.variable_scope(scope):
            for layer in np.arange(0,len(out_size)):
                sh = current.get_shape().as_list()
                current = cell2D_gated(current, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=True, act_type=act_type,gate=gate)
                if prob!=1:
                    current = self.mydropout(current,mode_node,prob[layer])
        return current
    
    def permute_eq(self,current,out_size,mode_node,scope=None, act_type=tf.nn.relu, alpha=1.):
        with tf.variable_scope(scope):
            for layer in np.arange(0,len(out_size)):
                sh = current.get_shape().as_list()
                pooled  = tf.reduce_max(current,axis=1,keep_dims=True)
                w1 = tf.get_variable('w1_'+str(layer),initializer = [0.],trainable = True, dtype='float32')
                current = cell2D(current-w1*pooled, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=False, act=False)
                current = BatchNorm(current,mode_node,scope=str(layer))
                current = act_type(current)
        return current

    def permute_res(self,current,out_size,mode_node,scope=None, act_type='relu', alpha=1.):
        with tf.variable_scope(scope):
            for layer in np.arange(0,len(out_size)):
                sh = current.get_shape().as_list()
                in_node = current
                with tf.variable_scope('stage_1'):
                    current = BatchNorm(current,mode_node,scope=str(layer))
                    current = tf.nn.relu(current)
                    current = cell2D(current, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=False, act=False)
                    pooled  = tf.reduce_max(current,axis=1,keep_dims=True)
                    w1 = tf.get_variable('w1_'+str(layer),initializer = [1.],trainable = True, dtype='float32')
                    current = current+w1*pooled
                with tf.variable_scope('stage_2'):
                    current = BatchNorm(current,mode_node,scope=str(layer))
                    current = tf.nn.relu(current)
                    current = cell2D(current, 1, sh[2], sh[-1] , out_size[layer], mode_node, 1, 'pointconv'+str(layer), padding='VALID', bn=False, act=False)
                    pooled2  = tf.reduce_max(current,axis=1,keep_dims=True)
                    w2 = tf.get_variable('w2_'+str(layer),initializer = [1.],trainable = True, dtype='float32')
                    current = current+w2*pooled2
                current = current + in_node
        return current
        
        
        
    def max_pool_augment(self,current,layers,scope=None):
        with tf.variable_scope(scope):
            net_ = tf.reduce_max(current,axis=1,keep_dims=True)
            aug_net = tf.concat((tf.tile(net_,(1,self.num_points,1,1)),layers),axis=-1)        
        return aug_net
                
    def point_net(self,point_cloud,mode_node):
        end_points = []
        with tf.variable_scope('MLP1'):
            point_cloud = tf.expand_dims(point_cloud,-1)
            mlp1 = cell2D(point_cloud, 1, 3, 1 , 64, mode_node, 1, 'pointconv1', padding='VALID', bn=True)
            mlp1 = cell2D(mlp1,             1, 1, 64, 64, mode_node, 1, 'pointconv2', padding='VALID', bn=True)
        # Get transformation
        with tf.variable_scope('feature_transformations'):
            T = self.feature_transformer(mlp1, 64, mode_node)
            T_ = tf.expand_dims(T,axis=1)
            # Apply transformation
            mlp_tiled = tf.tile(tf.transpose(mlp1,(0,1,3,2)),(1,1,1,64))
            mlp_rot = tf.reduce_sum(mlp_tiled*T_,axis=2,keep_dims=True)
        with tf.variable_scope('MLP2'):
            mlp2 = cell2D(mlp_rot, 1, 1, 64, 64, mode_node, 1, 'pointconv1', padding='VALID', bn=True)
            mlp2_ = cell2D(mlp2,     1, 1, 64, 128, mode_node, 1, 'pointconv2', padding='VALID', bn=True)
            mlp2 = cell2D(mlp2_,     1, 1, 128, 1024, mode_node, 1, 'pointconv3', padding='VALID', bn=True)
        with tf.variable_scope('Pooling'):
            net_ = tf.reduce_max(mlp2,axis=1,keep_dims=True)
        with tf.variable_scope('Augment_points'):
            aug_net = tf.concat((tf.tile(net_,(1,self.num_points,1,1)),mlp2_,mlp_rot,mlp1),axis=-1)
        with tf.variable_scope('MLP3'):
            mlp3 = cell2D(aug_net, 1, 1, aug_net.get_shape().as_list()[3], 512, mode_node, 1, 'pointconv1', padding='VALID', bn=True)
            mlp3 = cell2D(mlp3,     1, 1, 512, 256, mode_node, 1, 'pointconv2', padding='VALID', bn=True)
            features = cell2D(mlp3,     1, 1, 256, 128, mode_node, 1, 'pointconv3', padding='VALID', bn=True)
#        with tf.variable_scope('FullyConnected'):
#            current = cell1D(features,512, mode_node, SCOPE='fc1', with_act=True, with_bn=False)
#            current = cell1D(current ,256, mode_node, SCOPE='fc2', with_act=True, with_bn=False)
#            current = cell1D(current,63, mode_node, SCOPE='logits', with_act=False, with_bn=False)  
#            marks = tf.tanh(tf.reshape(current,(self.batch_size,3,21)))
#        end_points.append(rot_mat)
        return tf.squeeze(features,axis=2), end_points
        

    
    def model_2(self,point_cloud_4d, idx, count, count_normalized, grid_size, mode_node, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        point_cloud = tf.slice(point_cloud_4d,[0,1],[-1,-1])
        point_cloud = point_cloud/(grid_size-1)
        voxel_point_count = tf.cast(tf.expand_dims(tf.expand_dims(tf.expand_dims(count,1),2),3),tf.float32)
        voxel_point_count_norm = tf.cast(tf.expand_dims(tf.expand_dims(tf.expand_dims(count_normalized,1),2),3),tf.float32)
        end_points = {}
        point_cloud = tf.expand_dims(tf.expand_dims(point_cloud,1),-1)
        # AUGMENT points by centered points [x,y,z, x_c,y_c,z_c][1]
        centers = tf.divide(tf.unsorted_segment_sum(point_cloud,idx,tf.reduce_max(idx)+1),voxel_point_count)
        final = tf.concat((tf.transpose(centers,(0,1,3,2)),voxel_point_count_norm),axis=-1)
        final = tf.squeeze(final,axis=(1,2))
        return final, end_points
        

    
    def point_transformer(self,point_cloud, mode_node, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        with tf.variable_scope('point_transformer'):
            point_cloud = tf.expand_dims(point_cloud,-1)     
            # TRANSFORM to feature space []
            net = cell2D(point_cloud, 1, 3, 1, 64, mode_node, 1, 'pointconv1', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 64, 128, mode_node, 1, 'pointconv2', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 128, 1024, mode_node, 1, 'pointconv3', padding='VALID', bn=True)
            # AUGMENT features 
            net_ = tf.reduce_max(net,axis=1,keep_dims=True)
            # MLP 2
            net = cell2D(net_, 1, 1, 1024, 512, mode_node, 1, 'pointconv4', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 512, 256, mode_node, 1, 'pointconv5', padding='VALID', bn=True)
            # FINAL MAXPOOL
            features = tf.squeeze(net,axis=(1,2))
            params = cell1D(features,1, mode_node, SCOPE='pred', with_act=False, with_bn=False)
#            rot_mat = self.rotation_matrix(params)
            rot_mat = self.rotation_matrix_1d(params)
        return rot_mat



    def feature_transformer(self,feature_cloud, K,  mode_node, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        with tf.variable_scope('feature_transformer'):
            # TRANSFORM to feature space []
            net = cell2D(feature_cloud, 1, 1, K, 64, mode_node, 1, 'pointconv1', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 64, 128, mode_node, 1, 'pointconv2', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 128, 1024, mode_node, 1, 'pointconv3', padding='VALID', bn=True)
            # AUGMENT features 
            net_ = tf.reduce_max(net,axis=1,keep_dims=True)
            # MLP 2
            net = cell2D(net_, 1, 1, 1024, 512, mode_node, 1, 'pointconv4', padding='VALID', bn=True)
            net = cell2D(net, 1, 1, 512, 256, mode_node, 1, 'pointconv5', padding='VALID', bn=True)
            # FINAL MAXPOOL
            features = tf.squeeze(net,axis=(1,2))
            with tf.variable_scope('transform_feat'):
                weights = tf.get_variable('weights', [256, K*K],
                                          initializer=tf.constant_initializer(0.0),
                                          dtype=tf.float32)
                biases = tf.get_variable('biases', [K*K],
                                         initializer=tf.constant_initializer(0.0),
                                         dtype=tf.float32)
                biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(features, weights)
            transform = tf.nn.bias_add(transform, biases) 
            transform = tf.reshape(transform,(self.batch_size,K,K))           
        return transform

    def rotation_matrix(self,params):
        axis = tf.slice(params,[0,0],[-1,3])
        theta = tf.slice(params,[0,3],[-1,1])
        axis = axis/tf.sqrt(tf.reduce_sum(tf.pow(axis,2),axis=1,keep_dims=True))
        a = tf.cos(theta/2.0)
        a_sin = -axis*tf.sin(theta/2.0)
        b = tf.slice(a_sin,[0,0],[-1,1])
        c = tf.slice(a_sin,[0,1],[-1,1])
        d = tf.slice(a_sin,[0,2],[-1,1])
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        rot_mat = tf.stack([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
        rot_mat = tf.transpose(tf.squeeze(rot_mat,axis=3),(2,0,1))
        return rot_mat

    def rotation_matrix_1d(self,theta):

        cosval = tf.cos(theta*2*np.pi)
        sinval = tf.sin(theta*2*np.pi)
        zero = cosval*0
        one = cosval/cosval
        rot_mat = tf.stack([[cosval, zero, sinval],
                         [zero, one, zero],
                         [-sinval,zero, cosval]])
        rot_mat = tf.transpose(tf.squeeze(rot_mat,axis=3),(2,0,1))
        return rot_mat
    
    def get_centers(self,point_cloud):
        point_cloud = (point_cloud+1)/2*self.grid_size
        point_cloud_4d = tf.concat((self.batch_idx,tf.reshape(point_cloud,(-1,3))),axis=1)
        point_cloud_idx = tf.cast(tf.round(point_cloud_4d),tf.int64)
        point_cloud_lin_idx = tf.squeeze(ravel_index(point_cloud_idx,self.grid_shape))
        vox_idx, idx, count = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
        vox_idx = tf.cast(vox_idx,dtype = tf.int32)
        vox_mult_idx = tf.transpose(unravel_index(vox_idx,self.grid_shape),(1,0))
        batch_idx = tf.squeeze(tf.slice(vox_mult_idx,(0,0),(-1,1)))
        max_point_per_vol = tf.segment_max(count,batch_idx)
        max_point_per_vol = tf.gather(max_point_per_vol,batch_idx)
        count_normalized = tf.divide(count,max_point_per_vol)
        return vox_mult_idx, idx, count, count_normalized, point_cloud_4d

    def mydropout(self, x, mode_node, prob):
      # TODO: test if this is a tensor scalar of value 1.0
        return tf.cond(mode_node, lambda: tf.nn.dropout(x, prob), lambda: x)
    
    def rotations(self):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotation_angle = tf.random_uniform((self.batch_size,1,1),minval=0,maxval=2*np.pi) 
        cosval = tf.cos(rotation_angle)
        sinval = tf.sin(rotation_angle)
        zeros = tf.zeros(cosval.shape,dtype=tf.float32)
        ones  = tf.ones(cosval.shape,dtype=tf.float32)

        rotation_matrix1 = tf.concat((cosval,zeros,sinval),axis=2)
        rotation_matrix2 = tf.concat((zeros,ones,zeros),axis=2)
        rotation_matrix3 = tf.concat((-sinval,zeros,cosval),axis=2)
        rotation_tensor = tf.concat((rotation_matrix1,rotation_matrix2,rotation_matrix3),axis=1)
#        rotation_matrix = np.array([[cosval, 0, sinval],
#                                        [0, 1, 0],
#                                        [-sinval, 0, cosval]])
        
        return rotation_tensor