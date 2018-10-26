import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import glob
import tensorflow as tf


    
class Rasterizer(object):
    def __init__(self , canvas_size=64,grid_size = 32):
        
        self.canvas_size = canvas_size
        self.grid_size   = grid_size

    def rasterize_np(self,mesh):
        x = np.linspace(0,self.canvas_size-1,self.canvas_size)
        y = np.linspace(0,self.canvas_size-1,self.canvas_size)
        grid = np.meshgrid(x,y)
        grid = np.stack(grid,axis=2)
        grid = (np.reshape(grid,(-1,2))/self.canvas_size)*2-1
        image = np.zeros((self.canvas_size*self.canvas_size,1),dtype=np.float32)
        for ii in range(self.canvas_size*self.canvas_size):
            image[ii] = self.inside_check(mesh,grid[ii,:])
            print(ii)
        image = np.reshape(image,(self.canvas_size,self.canvas_size,1))
        return image


      
        
    def inside_check_np(self,mesh,P):
        A = mesh['vertices'][mesh['faces'][:,0],:]
        B = mesh['vertices'][mesh['faces'][:,1],:]
        C = mesh['vertices'][mesh['faces'][:,2],:]
        eps = 1e-8
        B_  = B[:,0:2] - A[:,0:2]
        C_  = C[:,0:2] - A[:,0:2]
        P_  = P - A[:,0:2]
        D   = B_[:,0]*C_[:,1] - C_[:,0]*B_[:,1]
        W_a = (P_[:,0]*(B_[:,1] - C_[:,1]) + P_[:,1]*(C_[:,0] - B_[:,0]) + D)/(D+eps)
        W_b = (P_[:,0]*C_[:,1] - P_[:,1]*C_[:,0])/(D+eps)
        W_c = (P_[:,1]*B_[:,0] - P_[:,0]*B_[:,1])/(D+eps)
        conds     = np.stack([W_a>0 ,W_a<1,W_b>0 ,W_b<1,W_c>0 ,W_c<1],axis=1)
        is_inside = np.all(conds,axis=1)
        values    = W_a*A[:,2]+W_b*B[:,2]+W_c*C[:,2]
        values    = values[is_inside]
        if values.shape[0]==0:
            pixel_value = 0.
        else:
            pixel_value = np.max(values)
        return pixel_value
    

    
    
    
    def rasterize_tf(self,mesh):
        with tf.name_scope('raycast'):

            x     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            y     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            grid = tf.meshgrid(x,y)
            grid = tf.stack(grid,axis=2)
            grid_list = tf.reshape(grid,(-1,2))
            
            A = tf.gather(mesh['vertices'],mesh['faces'][:,0],axis=0)
            B = tf.gather(mesh['vertices'],mesh['faces'][:,1],axis=0)
            C = tf.gather(mesh['vertices'],mesh['faces'][:,2],axis=0)
            B_norm  = B[:,0:2] - A[:,0:2]
            C_norm  = C[:,0:2] - A[:,0:2]
            D_norm   = B_norm[:,0]*C_norm[:,1] - C_norm[:,0]*B_norm[:,1]
            with tf.name_scope('indexing'):
                mapping = tf.map_fn(lambda x: self.get_closest_face(x,A,B,C,B_norm,C_norm,D_norm),grid_list,dtype=(tf.int64, tf.bool),back_prop=False,infer_shape=True)
            
            
#            tf.gradients(mapping[0], A)
            
            
            # Interpolate values
            indices = mapping[0]
            is_mesh = mapping[1]
            is_mesh = tf.reshape(is_mesh,(self.canvas_size,self.canvas_size))
            A_ = tf.reshape(tf.gather(tf.reshape(A,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.canvas_size,self.canvas_size,3))
            P_ = grid - A_[:,:,0:2]
            B_ = tf.reshape(tf.gather(tf.reshape(B,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.canvas_size,self.canvas_size,3))
            C_ = tf.reshape(tf.gather(tf.reshape(C,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.canvas_size,self.canvas_size,3))
            B_norm_ = tf.reshape(tf.gather(B_norm,indices,axis=0),(self.canvas_size,self.canvas_size,2))
            C_norm_ = tf.reshape(tf.gather(C_norm,indices,axis=0),(self.canvas_size,self.canvas_size,2))
            D_norm_ = tf.reshape(tf.gather(D_norm,indices,axis=0),(self.canvas_size,self.canvas_size))
            W_a = (P_[:,:,0]*(B_norm_[:,:,1] - C_norm_[:,:,1]) + P_[:,:,1]*(C_norm_[:,:,0] - B_norm_[:,:,0]) + D_norm_)/(D_norm_+1e-8)
            W_b = (P_[:,:,0]* C_norm_[:,:,1] - P_[:,:,1]*C_norm_[:,:,0])/(D_norm_+1e-8)
            W_c = (P_[:,:,1]* B_norm_[:,:,0] - P_[:,:,0]*B_norm_[:,:,1])/(D_norm_+1e-8)
            values = W_a*A_[:,:,2]+W_b*B_[:,:,2]+W_c*C_[:,:,2]
            
            background = tf.zeros(values.shape,dtype=tf.float32)
            image      = tf.where(is_mesh,(values+1)/2,background)
            image      = tf.expand_dims(tf.transpose(image,(1,0)),-1)


        
#        image = tf.map_fn(lambda x: self.inside_check_tf(x,mesh),grid,dtype=tf.float64)
#        image = tf.reshape(image,(self.canvas_size,self.canvas_size,1))
        return image        
       
        
    def get_closest_face(self,P,A,B,C,B_,C_,D):
        eps = 1e-8
        P_  = P - A[:,0:2]
        W_a = (P_[:,0]*(B_[:,1] - C_[:,1]) + P_[:,1]*(C_[:,0] - B_[:,0]) + D)/(D+eps)
        W_b = (P_[:,0]*C_[:,1] - P_[:,1]*C_[:,0])/(D+eps)
        W_c = (P_[:,1]*B_[:,0] - P_[:,0]*B_[:,1])/(D+eps)
        conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=1)
        is_inside = tf.reduce_all(conds,axis=1)
        is_mesh   = tf.reduce_any(is_inside,axis=0)
        values    = W_a*A[:,2]+W_b*B[:,2]+W_c*C[:,2]
        values    = (values+1)/2
        background        = tf.ones(values.shape,dtype=tf.float32)
        values_normalized = tf.where(is_inside,values,background)
        indices           = tf.argmin(values_normalized,axis=0)
        return indices, is_mesh
    
    
    
    
    
    
    
    
    
    def inside_check_tf(self,P,mesh):
        A = tf.gather(mesh['vertices'],mesh['faces'][:,0],axis=0)
        B = tf.gather(mesh['vertices'],mesh['faces'][:,1],axis=0)
        C = tf.gather(mesh['vertices'],mesh['faces'][:,2],axis=0)
        eps = 1e-8
        B_  = B[:,0:2] - A[:,0:2]
        C_  = C[:,0:2] - A[:,0:2]
        P_  = P - A[:,0:2]
        D   = B_[:,0]*C_[:,1] - C_[:,0]*B_[:,1]
        W_a = (P_[:,0]*(B_[:,1] - C_[:,1]) + P_[:,1]*(C_[:,0] - B_[:,0]) + D)/(D+eps)
        W_b = (P_[:,0]*C_[:,1] - P_[:,1]*C_[:,0])/(D+eps)
        W_c = (P_[:,1]*B_[:,0] - P_[:,0]*B_[:,1])/(D+eps)
        conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=1)

        is_inside = tf.reduce_all(conds,axis=1)
        values    = W_a*A[:,2]+W_b*B[:,2]+W_c*C[:,2]
        values = (values+1)/2
        values_normalized = values*tf.cast(is_inside,tf.float64)
        pixel_value = tf.reduce_max(values_normalized,keep_dims=True)
#        pixel_value = tf.concat((P,pixel_value),axis=0)
        return pixel_value
    
    
    
    
    
    
    
    def augment_data(self,mesh,params):
        with tf.name_scope('data_augmentation'):
            augmentations    = Rasterizer.get_aug_params(params)
            mesh['vertices'] = self.apply_augmentations(mesh['vertices'],augmentations)
        return mesh    
    
    def augment_data_var(self,mesh,params):
        with tf.name_scope('data_augmentation'):
            augmentations    = Rasterizer.get_aug_params_var(params)
            mesh['vertices'] = self.apply_augmentations(mesh['vertices'],augmentations)
        return mesh      
    
    
    def apply_augmentations(self,point_cloud,augmentations):
        # Rotations     
        point_cloud = tf.matmul(point_cloud,augmentations['rotations'])
        # Scale
        point_cloud = point_cloud * augmentations['scale']
        # Translations
        point_cloud = point_cloud + augmentations['shift']
        return point_cloud
    
    
    
    
    @staticmethod
    def get_aug_params_var(params):
        batch_size = 1
        # Scale
        scale_factor = params['scale']+1
        # Translations
        voxel_shift = params['trans']
        # Rot        
        rot_mat    = Rasterizer.rotation_matrix(params['rot'])
        augmentations = {}
        augmentations['scale'] = tf.squeeze(scale_factor,axis=0)
        augmentations['shift'] = tf.squeeze(voxel_shift,axis=0)
        augmentations['rotations'] = tf.squeeze(rot_mat,axis=0)
        return augmentations    




    @staticmethod
    def get_aug_params(params):
        batch_size = 1
        # Scale
        scale_factor = tf.random_uniform((batch_size,1,1),minval=1-params['scale'],maxval=1+params['scale'],dtype=tf.float32)
        # Translations
        voxel_shift = tf.random_uniform((batch_size,1,1),minval=-params['trans'],maxval=params['trans'],dtype=tf.float32)
        # Rot
#        rot_params = tf.concat((tf.zeros((batch_size,2),dtype = tf.float32),tf.ones((batch_size,1),dtype = tf.float32)),axis=1)
        axis       = tf.random_normal((1,3),
                                        mean=0.0,
                                        stddev=1.0,
                                        dtype=tf.float32)
        theta      = tf.random_uniform((1,1),minval=-params['rot'],maxval=params['rot'],dtype=tf.float32)
        rot_params = tf.concat((axis,theta),axis=1)
        rot_mat    = Rasterizer.rotation_matrix(rot_params)
        augmentations = {}
        augmentations['scale'] = tf.squeeze(scale_factor,axis=0)
        augmentations['shift'] = tf.squeeze(voxel_shift,axis=0)
        augmentations['rotations'] = tf.squeeze(rot_mat,axis=0)
        return augmentations






    @staticmethod
    def rotation_matrix(params):
        axis = tf.slice(params,[0,0],[-1,3])
        theta = tf.slice(params,[0,3],[-1,1])*np.pi
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
    
  