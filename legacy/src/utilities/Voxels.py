import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import glob
import tensorflow as tf


    
class Voxels(object):
    def __init__(self , canvas_size=64,grid_size = 32, batch_size=32, num_points=10000):
        
        self.canvas_size = canvas_size
        self.grid_size   = grid_size
        self.batch_size  = batch_size
        self.num_points  = num_points
        self.grid_shape  = tf.constant([self.batch_size,self.grid_size,self.grid_size,self.grid_size],dtype=tf.int64)
        self.voxel_shape = tf.constant([self.batch_size,self.grid_size,self.grid_size,self.grid_size,1],dtype=tf.int32)
        self.batch_idx_np= np.expand_dims(np.meshgrid(np.arange(0, self.batch_size, 1),np.arange(0, self.num_points, 1))[0].transpose(),axis=2)
        self.batch_idx   = tf.reshape(tf.convert_to_tensor(self.batch_idx_np.astype(np.float32)),(self.batch_size*self.num_points,1))

  
    
    def voxelize(self,point_cloud,field):
        
        indices, idx, count, count_normalized, point_cloud_4d = self.get_centers(point_cloud) 
#        features_vox = tf.cast(indices,tf.float32)/(self.grid_size-1)
#        features_vox = tf.slice(features_vox,[0,2],[-1,1])
        features_vox = tf.expand_dims(tf.cast(count,dtype=tf.float32),-1)
        features_vox = features_vox/features_vox
#        idx = tf.unique(idx)
#        features_vox = tf.gather(tf.constant(field),idx[0],axis=0)
        voxels = tf.scatter_nd(indices, features_vox, self.voxel_shape)
        return(voxels)
    
    def get_centers(self,point_cloud):
        point_cloud = (point_cloud+1)/2*self.grid_size
        point_cloud_4d = tf.concat((self.batch_idx,tf.reshape(point_cloud,(-1,3))),axis=1)
        point_cloud_idx = tf.cast(tf.round(point_cloud_4d),tf.int64)
        point_cloud_lin_idx = tf.squeeze(Voxels.ravel_index(point_cloud_idx,self.grid_shape))
        vox_idx, idx, count = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
        vox_idx = tf.cast(vox_idx,dtype = tf.int32)
        vox_mult_idx = tf.transpose(Voxels.unravel_index(vox_idx,self.grid_shape),(1,0))
        batch_idx = tf.squeeze(tf.slice(vox_mult_idx,(0,0),(-1,1)))
        max_point_per_vol = tf.segment_max(count,batch_idx)
        max_point_per_vol = tf.gather(max_point_per_vol,batch_idx)
        count_normalized = tf.divide(count,max_point_per_vol)
        return vox_mult_idx, idx, count, count_normalized, point_cloud_4d
    
    
    
    
    
    
    @staticmethod
    def unravel_index(indices, shape):
    #    indices = tf.transpose(indices,(1,0))
        indices = tf.expand_dims(indices, 0)
        shape = tf.expand_dims(shape, 1)
        shape = tf.cast(shape, tf.float32)
        strides = tf.cumprod(shape, reverse=True)
        strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
        strides = tf.cast(strides, tf.int32)
        strides_shifted = tf.cast(strides_shifted, tf.int32)
        def even():
            rem = indices - (indices // strides) * strides
            return rem // strides_shifted
        def odd():
            div = indices // strides_shifted
            return div - (div // strides) * strides
        rank = tf.rank(shape)
        return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)
    
    @staticmethod
    def ravel_index(bxyz, shape):
        b = tf.slice(bxyz,[0,0],[-1,1])
        x = tf.slice(bxyz,[0,1],[-1,1])
        y = tf.slice(bxyz,[0,2],[-1,1])
        z = tf.slice(bxyz,[0,3],[-1,1])
        return z + shape[3]*y + (shape[2] * shape[1])*x + (shape[3] * shape[2] * shape[1])*b
    
    
    
    