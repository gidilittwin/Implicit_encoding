import numpy as np
import tensorflow as tf
from model_ops import unravel_index,ravel_index_bxy, ravel_xy

def get_closest_face_np(P,A,B,C,B_,C_,D):
    eps = 1e-8
    P_  = P - A[:,:,0:2]
    W_a = (P_[:,:,0]*(B_[:,:,1] - C_[:,:,1]) + P_[:,:,1]*(C_[:,:,0] - B_[:,:,0]) + D)/(D+eps)
    W_b = (P_[:,:,0]*C_[:,:,1] - P_[:,:,1]*C_[:,:,0])/(D+eps)
    W_c = (P_[:,:,1]*B_[:,:,0] - P_[:,:,0]*B_[:,:,1])/(D+eps)
    conds     = np.stack([np.greater(W_a,0) ,np.less(W_a,1),np.greater(W_b,0) ,np.less(W_b,1),np.greater(W_c,0) ,np.less(W_c,1)],axis=2)
    is_inside = np.all(conds,axis=2)
    values    = W_a*A[:,:,2]+W_b*B[:,:,2]+W_c*C[:,:,2]
    values    = (values - 4)*is_inside.astype(np.float32)
    indices   = np.argmin(values,axis=1)    
    return indices

 
def get_closest_face(P,A,B,C,B_,C_,D):
    eps = 1e-8
    P_  = P-A[:,:,0:2]
    W_a = (P_[:,:,0]*(B_[:,:,1] - C_[:,:,1]) + P_[:,:,1]*(C_[:,:,0] - B_[:,:,0]) + D)/(D+eps)
    W_b = (P_[:,:,0]*C_[:,:,1] - P_[:,:,1]*C_[:,:,0])/(D+eps)
    W_c = (P_[:,:,1]*B_[:,:,0] - P_[:,:,0]*B_[:,:,1])/(D+eps)
    conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=2)
    is_inside = tf.reduce_all(conds,axis=2)
    values    = W_a*A[:,:,2]+W_b*B[:,:,2]+W_c*C[:,:,2]
    values    = (values - 4)*tf.cast(is_inside,tf.float32)
    indices   = tf.argmin(values,axis=1)
    return indices

def fast_check(P,IN_):
    min_ = P - tf.reduce_min(IN_,axis=3)
    max_ = tf.reduce_max(IN_,axis=3) - P
    conds     = tf.concat([tf.greater(min_,0),tf.greater(max_,0)],axis=2)
    is_inside = tf.reduce_all(conds,axis=2)
    return is_inside

def is_inside(P,A,B_,C_,D):
    eps = 1e-8
    P_  = P - A[:,:,0:2]
    W_a = (P_[:,:,0]*(B_[:,:,1] - C_[:,:,1]) + P_[:,:,1]*(C_[:,:,0] - B_[:,:,0]) + D)/(D+eps)
    W_b = (P_[:,:,0]*C_[:,:,1] - P_[:,:,1]*C_[:,:,0])/(D+eps)
    W_c = (P_[:,:,1]*B_[:,:,0] - P_[:,:,0]*B_[:,:,1])/(D+eps)
    conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=2)
    is_inside = tf.reduce_all(conds,axis=2)
    return is_inside




class Raycast(object):
    def __init__(self, batch_size, canvas_size=88):
        self.canvas_size = canvas_size
        self.batch_size  = batch_size
        self.minValue   = tf.convert_to_tensor(0.)
        self.maxValue   = tf.convert_to_tensor(1.)
        self.num_pixels = self.canvas_size*self.canvas_size

    def rasterize(self,mesh):
       with tf.name_scope('raycast'):
            
            x     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            y     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            grid  = tf.meshgrid(x,y)
            grid  = tf.stack(grid,axis=2)
            grid_list    = tf.reshape(grid,(-1,1,1,2))
            grid_batched = tf.expand_dims(grid,0)
            
            # Mesh faces
            faces    = mesh.faces
            vertices = mesh.vertices_t
            background_face     = tf.constant([[4023,4024,4025]],dtype=tf.int64)
            background_vertices = tf.tile(tf.constant([[[-2,-2,1],[-2,10,1],[10,-2,1]]],dtype=tf.float32),(self.batch_size,1,1))
            faces    = tf.concat((faces,background_face),axis=0)
            vertices = tf.concat((vertices,background_vertices),axis=1)
            
            # Get indeices of closest intersection per pixel in a non differentiable way (map_fn is memory consuming when back_prop=True so
            # we find the face index per pixel and interpolate again once per pixel)
            A   = tf.gather(vertices,faces[:,0],axis=1)
            B   = tf.gather(vertices,faces[:,1],axis=1)
            C   = tf.gather(vertices,faces[:,2],axis=1)
            B_norm  = B[:,:,0:2] - A[:,:,0:2]
            C_norm  = C[:,:,0:2] - A[:,:,0:2] 
            D_norm  = B_norm[:,:,0]*C_norm[:,:,1] - C_norm[:,:,0]*B_norm[:,:,1]

            indices = tf.map_fn(lambda x: get_closest_face(x,A,B,C,B_norm,C_norm,D_norm),grid_list,dtype=tf.int64,back_prop=False,parallel_iterations=10,infer_shape=True,swap_memory=True)

            indices       = tf.stop_gradient(tf.transpose(indices,(1,0)))
            indices_image = tf.reshape(indices,(self.batch_size,self.canvas_size,self.canvas_size))
            
            # Interpolate values
            A_ = tf.reshape(tf.gather(tf.reshape(A,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size,3))
            P_ = grid_batched - A_[:,:,:,0:2]
            B_ = tf.reshape(tf.gather(tf.reshape(B,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size,3))
            C_ = tf.reshape(tf.gather(tf.reshape(C,(-1,3)),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size,3))
            B_norm_ = tf.reshape(tf.gather(tf.reshape(B_norm,(-1,2)),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size,2))
            C_norm_ = tf.reshape(tf.gather(tf.reshape(C_norm,(-1,2)),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size,2))
            D_norm_ = tf.reshape(tf.gather(tf.reshape(D_norm,(-1, )),tf.reshape(indices,(-1,)),axis=0),(self.batch_size,self.canvas_size,self.canvas_size))
            W_a = (P_[:,:,:,0]*(B_norm_[:,:,:,1] - C_norm_[:,:,:,1]) + P_[:,:,:,1]*(C_norm_[:,:,:,0] - B_norm_[:,:,:,0]) + D_norm_)/(D_norm_+1e-8)
            W_b = (P_[:,:,:,0]* C_norm_[:,:,:,1] - P_[:,:,:,1]*C_norm_[:,:,:,0])/(D_norm_+1e-8)
            W_c = (P_[:,:,:,1]* B_norm_[:,:,:,0] - P_[:,:,:,0]*B_norm_[:,:,:,1])/(D_norm_+1e-8)
            values = W_a*A_[:,:,:,2]+W_b*B_[:,:,:,2]+W_c*C_[:,:,:,2]
            image = (values+1)/2
            background = tf.zeros(image.shape,dtype=tf.float32)
            image = tf.where(tf.equal(indices_image,faces.get_shape().as_list()[0]-1),background,image)
            image = tf.expand_dims(tf.transpose(image,(0,2,1)),-1)
            return image




class Forward_map(object):
    def __init__(self, batch_size, canvas_size=88):
        self.canvas_size = canvas_size
        self.batch_size  = batch_size
        self.minValue   = tf.convert_to_tensor(0.)
        self.maxValue   = tf.convert_to_tensor(1.)
        self.num_pixels = self.canvas_size*self.canvas_size

    def rasterize(self,mesh):
        with tf.name_scope('raycast'):
            
            x     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            y     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            idx   = np.expand_dims(np.expand_dims(np.linspace(0,self.batch_size-1,self.batch_size,dtype=np.float32),1),1)
            idx   = tf.constant(np.tile(idx,(1,8017,1)))
            grid  = tf.meshgrid(x,y)
            grid  = tf.stack(grid[0:2],axis=2)
            grid_list    = tf.reshape(grid,(-1,1,1,2))
            grid_batched = tf.expand_dims(grid,0)            
            
            # Mesh faces
            faces    = mesh.faces
            vertices = mesh.vertices_t
            background_face     = tf.constant([[4023,4024,4025]],dtype=tf.int64)
            background_vertices = tf.tile(tf.constant([[[-2,-2,1],[-2,10,1],[10,-2,1]]],dtype=tf.float32),(self.batch_size,1,1))
            faces    = tf.concat((faces,background_face),axis=0)
            vertices = tf.concat((vertices,background_vertices),axis=1)
            
            # Get indeices of closest intersection per pixel in a non differentiable way (map_fn is memory consuming when back_prop=True so
            # we find the face index per pixel and interpolate again once per pixel)
            A   = tf.gather(vertices,faces[:,0],axis=1)
            B   = tf.gather(vertices,faces[:,1],axis=1)
            C   = tf.gather(vertices,faces[:,2],axis=1)
            A = (tf.reshape(A,(-1,3))+1)/2*self.canvas_size
            B = (tf.reshape(B,(-1,3))+1)/2*self.canvas_size
            C = (tf.reshape(C,(-1,3))+1)/2*self.canvas_size
            IDX =tf.reshape(idx,(-1,1))
            
            IN = (A,B,C,IDX)
#            IN = (tf.stack((A,B_,C_,idx),axis=2)+1)/2*self.canvas_size
            
#            with tf.variable_scope('render',reuse=tf.AUTO_REUSE):
#                10000*tf.get_variable('Z_buffer',initializer = np.ones(IN.get_shape().as_list(),dtype = np.int64))
            size_   = (self.batch_size,self.canvas_size,self.canvas_size)
            Z_buffer = tf.constant(10000*np.ones(size_,dtype=np.float32))
            body    = lambda x: bounding_pixels(x,size_,Z_buffer)
            indices = tf.map_fn(body,IN,dtype=tf.float32,back_prop=False,parallel_iterations=10,infer_shape=True,swap_memory=True)
        return image





class Raycast_bb(object):
    def __init__(self, batch_size, canvas_size=88):
        self.canvas_size = canvas_size
        self.batch_size  = batch_size
        self.minValue   = tf.convert_to_tensor(0.)
        self.maxValue   = tf.convert_to_tensor(1.)
        self.num_pixels = self.canvas_size*self.canvas_size

    def rasterize(self,mesh):
        with tf.name_scope('raycast'):
            
            x     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            y     = np.linspace(-1,1,self.canvas_size,dtype=np.float32)
            grid  = tf.meshgrid(x,y)
            grid  = tf.stack(grid,axis=2)
            grid_list    = tf.reshape(grid,(-1,1,1,2))
            
            # Mesh faces
            faces    = mesh.faces
            vertices = mesh.vertices_t
            background_face     = tf.constant([[4023,4024,4025]],dtype=tf.int64)
            background_vertices = tf.tile(tf.constant([[[-2,-2,1],[-2,10,1],[10,-2,1]]],dtype=tf.float32),(self.batch_size,1,1))
            faces    = tf.concat((faces,background_face),axis=0)
            vertices = tf.concat((vertices,background_vertices),axis=1)
            
            # Get indeices of closest intersection per pixel in a non differentiable way (map_fn is memory consuming when back_prop=True so
            # we find the face index per pixel and interpolate again once per pixel)
            A   = tf.gather(vertices,faces[:,0],axis=1)
            B   = tf.gather(vertices,faces[:,1],axis=1)
            C   = tf.gather(vertices,faces[:,2],axis=1)

            # get boolean map of posible intersections (bounding box based)
            IN_ = tf.stack((A[:,:,0:2],B[:,:,0:2],C[:,:,0:2]),axis=3)
            to_test       = tf.map_fn(lambda x: fast_check(x,IN_),
                                    grid_list,dtype=tf.bool,back_prop=False,parallel_iterations=10,infer_shape=True,swap_memory=True)
            test_indices  = tf.where(tf.equal(to_test, True))
            grid_indices  = tf.gather(grid_list,test_indices[:,0],axis=0)
            P_t = tf.squeeze(grid_indices,axis=(1,2))
            A_t = tf.gather_nd(A,test_indices[:,1:3])
            B_t = tf.gather_nd(B,test_indices[:,1:3])
            C_t = tf.gather_nd(C,test_indices[:,1:3])
            P_t_ = P_t - A_t[:,0:2]
            B_t_ = B_t[:,0:2] - A_t[:,0:2]
            C_t_ = C_t[:,0:2] - A_t[:,0:2]
            D_t_ = B_t_[:,0:1]*C_t_[:,1:2] - C_t_[:,0:1]*B_t_[:,1:2]
            W_a = (P_t_[:,0:1]*(B_t_[:,1:2] - C_t_[:,1:2]) + P_t_[:,1:2]*(C_t_[:,0:1] - B_t_[:,0:1]) + D_t_)/(D_t_+1e-8)
            W_b = (P_t_[:,0:1]* C_t_[:,1:2] - P_t_[:,1:2]*C_t_[:,0:1])/(D_t_+1e-8)
            W_c = (P_t_[:,1:2]* B_t_[:,0:1] - P_t_[:,0:1]*B_t_[:,1:2])/(D_t_+1e-8)
            
            # Remove non intersecting rays
            conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=-1)
            is_inside = tf.reduce_all(conds,axis=-1)[:,0]
            is_inside = tf.squeeze(tf.where(tf.equal(is_inside, True)),axis=1)
            W_a_inside = tf.gather(W_a,is_inside,axis=0)
            W_b_inside = tf.gather(W_b,is_inside,axis=0)
            W_c_inside = tf.gather(W_c,is_inside,axis=0)
            A_t_inside = tf.gather(A_t[:,2:3],is_inside,axis=0)
            B_t_inside = tf.gather(B_t[:,2:3],is_inside,axis=0)
            C_t_inside = tf.gather(C_t[:,2:3],is_inside,axis=0)
            values        = W_a_inside*A_t_inside + W_b_inside*B_t_inside+W_c_inside*C_t_inside
            batch_indices = tf.gather(test_indices[:,0:2],is_inside,axis=0)
            lin_indices   = ravel_xy(batch_indices)

            # Argmin on pixels
            vox_idx, idx, count = tf.unique_with_counts(lin_indices[:,0],out_idx=tf.int32)
            image = -1*tf.unsorted_segment_max(-1*values,idx,self.num_pixels*self.batch_size)
            image = tf.transpose(tf.reshape(image,(self.canvas_size,self.canvas_size,self.batch_size,1)),(2,1,0,3))
            background = tf.equal(image,0)
            
        return image




    
    

        
#    def interpolate(self,P,A,B,C,B_,C_,D):
#        eps = 1e-8
#        P_  = P - A[:,:,0:2]
#        W_a = (P_[:,:,0]*(B_[:,:,1] - C_[:,:,1]) + P_[:,:,1]*(C_[:,:,0] - B_[:,:,0]) + D)/(D+eps)
#        W_b = (P_[:,:,0]*C_[:,:,1] - P_[:,:,1]*C_[:,:,0])/(D+eps)
#        W_c = (P_[:,:,1]*B_[:,:,0] - P_[:,:,0]*B_[:,:,1])/(D+eps)
#        conds     = tf.stack([tf.greater(W_a,0) ,tf.less(W_a,1),tf.greater(W_b,0) ,tf.less(W_b,1),tf.greater(W_c,0) ,tf.less(W_c,1)],axis=2)
#        is_inside = tf.reduce_all(conds,axis=2)
#        values    = W_a*A[:,:,2]+W_b*B[:,:,2]+W_c*C[:,:,2]
#        values    = (values+1)/2
#        background        = tf.ones(values.shape,dtype=tf.float32)
#        values_normalized = tf.where(is_inside,values,background)
#        pixel_value       = tf.reduce_min(values_normalized,axis=1,keep_dims=True)
#        return pixel_value
    




class Forward_map_sample(object):
    def __init__(self, batch_size, canvas_size=88):
        self.canvas_size = canvas_size
        self.batch_size  = batch_size
        self.minValue   = tf.convert_to_tensor(0.)
        self.maxValue   = tf.convert_to_tensor(1.)
    

    def rasterize(self,mesh):
        samples_tri  = mesh.faces_up
        vertices_t   = mesh.vertices_t
        faces_values = tf.gather(vertices_t,samples_tri,axis=1) # [batch, samples, edge, xyz ]
        # projection of normals on Z axis
        V1 = tf.gather(faces_values,1,axis=2) - tf.gather(faces_values,0,axis=2)
        V2 = tf.gather(faces_values,2,axis=2) - tf.gather(faces_values,0,axis=2)
        V1x,V1y,V1z = tf.split(V1,[1,1,1],axis=2)
        V2x,V2y,V2z = tf.split(V2,[1,1,1],axis=2)
        normal_z = tf.reshape(V1x*V2y - V1y*V2x,(-1,))
        # Up sampling the mesh
        num_faces    = samples_tri.get_shape().as_list()[0]
        r1   = tf.random_uniform((self.batch_size,num_faces,1,1),dtype=tf.float32)
        r2   = tf.random_uniform((self.batch_size,num_faces,1,1),dtype=tf.float32)
        rand_samps = tf.concat(((1-tf.sqrt(r1)),(tf.sqrt(r1)*(1-r2)),(r2*tf.sqrt(r1))),axis=2)
        vertices_MC  = tf.reduce_sum(faces_values*rand_samps,axis=2)
        num_vertices = num_faces
        # Re-project
        batch_idx_np = np.expand_dims(np.meshgrid(np.arange(0, self.batch_size, 1),np.arange(0, num_vertices, 1))[0].transpose(),axis=2)
        batch_idx = tf.reshape(tf.convert_to_tensor(batch_idx_np.astype(np.float32)),(self.batch_size*num_vertices,1))
        vertices_trans_values = tf.reshape(vertices_MC,(-1,3))
        vertices_trans_values = (vertices_trans_values/2+0.5)
        image_mask = tf.logical_or(tf.reduce_any(tf.less_equal(vertices_trans_values,self.minValue),axis=1),
                                   tf.reduce_any(tf.greater_equal(vertices_trans_values,self.maxValue),axis=1))        
        image_mask = tf.logical_or(image_mask,tf.greater_equal(normal_z,0))
        image_mask = tf.tile(tf.expand_dims(image_mask,axis=-1),(1,3))
        
        
        vertices_trans_values  = tf.where(image_mask,tf.zeros(image_mask.get_shape().as_list(),dtype=tf.float32),vertices_trans_values)
#        vertices_up            = tf.reshape(vertices_trans_values,(self.batch_size,-1,3))*2-1
        vertices_trans_indices_4d = tf.concat((batch_idx,vertices_trans_values*(self.canvas_size-1)),axis=1)
        vertices_trans_indices_4d = tf.cast(vertices_trans_indices_4d,tf.int64)
        point_cloud_lin_idx       = tf.squeeze(ravel_index_bxy(tf.slice(vertices_trans_indices_4d,[0,0],[-1,3]),(self.batch_size,self.canvas_size,self.canvas_size)))
        vox_idx, idx, count       = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
#        indices = tf.transpose(unravel_index(vox_idx,(self.batch_size,self.canvas_size,self.canvas_size),Type=tf.int64),(1,0))
        values_z = -1*tf.unsorted_segment_max(tf.slice(-1*vertices_trans_values,[0,2],[-1,1]),idx,tf.reduce_max(idx)+1)




        # get XYZ values corresponding to values_z
        max_values   = tf.squeeze(tf.gather(values_z,idx,axis=0),axis=1)
        max_xyz_idx  = tf.squeeze(tf.where(tf.equal(vertices_trans_values[:,2],max_values)),axis=1)
        values_xyz     = tf.gather(vertices_trans_values    ,max_xyz_idx,axis=0)
        values_xyz_idx = tf.gather(vertices_trans_indices_4d,max_xyz_idx,axis=0)
        image  = tf.scatter_nd(values_xyz_idx[:,0:3], values_xyz, (self.batch_size,self.canvas_size,self.canvas_size,3)) 
        
        # Semantic labeling image
        semantics = tf.tile(tf.expand_dims(mesh.semantics_up,axis=0),(self.batch_size,1,1))
        semantics = tf.reshape(semantics,(-1,3))
        semantics = tf.where(image_mask,tf.zeros(image_mask.get_shape().as_list(),dtype=tf.int64),semantics)
        semantics_visible = tf.gather(semantics, max_xyz_idx,axis=0)
        image_semantics   = tf.scatter_nd(values_xyz_idx[:,0:3], semantics_visible, (self.batch_size,self.canvas_size,self.canvas_size,3)) 

        return image, image_semantics















    