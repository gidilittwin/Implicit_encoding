import numpy as np
import tensorflow as tf
from model_ops import ravel_index,unravel_index,ravel_index_bxy
from kinematic_model import model as MODEL

class Kinetic(object):
    def __init__(self , batch_size, grid_size=96, box_limit=300, num_joints=21):
        self.batch_size = batch_size
        self.box_limit  = box_limit
        self.grid_size  = grid_size
        self.num_joints = num_joints
        mark_init       = np.array([[[0.],[0.],[0.],[1.]]])
        mark_init       = np.tile(mark_init,(batch_size,1,1))
        self.marks_init = tf.constant(mark_init,dtype=tf.float32)
#        self.model      = MODEL.V2(self.batch_size,self.box_limit)
        self.model      = MODEL.V3(self.batch_size,self.box_limit)
        self.MT         = self.get_local_trans(self.model['Bones'],self.model['Pose'],representation='euler')
        self.meshObj    = MODEL.objItem(self.box_limit)                
        self.grid_shape = tf.constant([self.batch_size,self.grid_size,self.grid_size,self.grid_size],dtype=tf.int64)
        self.minValue   = tf.convert_to_tensor(0.)
        self.maxValue   = tf.convert_to_tensor(1.)


        
        
    def translation(self,params):
        I = tf.eye(4,num_columns=4,batch_shape=[self.batch_size],dtype=tf.float32,name=None)
        trans = tf.concat((tf.slice(I,[0,0,0],[-1,3,3]),params),axis=2)
        trans = tf.concat((trans,tf.slice(I,[0,3,0],[-1,1,-1])),axis=1)
        trans_inv = tf.concat((tf.slice(I,[0,0,0],[-1,3,3]),-1*params),axis=2)
        trans_inv = tf.concat((trans_inv,tf.slice(I,[0,3,0],[-1,1,-1])),axis=1)        
        return trans,trans_inv


    
    def rotation(self,params):
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
        zero = 0*aa
        ones = aa/aa
        rot_mat = tf.stack([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), zero],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), zero],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, zero],
                            [zero      ,zero       ,zero          , ones]])
        rot_mat = tf.transpose(tf.squeeze(rot_mat,axis=3),(2,0,1))
        rot_mat_inv = tf.transpose(rot_mat,(0,2,1))
        return rot_mat,rot_mat_inv

    def rotation_rodrigues(self,theta, name=None):
        """
        Theta is N x 3
        """
        with tf.name_scope(name, "batch_rodrigues", [theta]):
            theta = theta*np.pi
            angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1),-1)
            axis = tf.div(theta, angle)
            a = tf.cos(angle/2.0)
            a_sin = -axis*tf.sin(angle/2.0)
            b = tf.slice(a_sin,[0,0],[-1,1])
            c = tf.slice(a_sin,[0,1],[-1,1])
            d = tf.slice(a_sin,[0,2],[-1,1])
            aa, bb, cc, dd = a*a, b*b, c*c, d*d
            bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
#            zero = 0*aa
#            ones = aa/aa
            zero = tf.zeros(aa.shape,dtype=tf.float32)
            ones = tf.ones(aa.shape,dtype=tf.float32)            
            rot_mat = tf.stack([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), zero],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), zero],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, zero],
                                [zero      ,zero       ,zero          , ones]])
            rot_mat = tf.transpose(tf.squeeze(rot_mat,axis=3),(2,0,1))
            rot_mat_inv = tf.transpose(rot_mat,(0,2,1))
            return rot_mat,rot_mat_inv


    def rotation_euler(self,theta, name=None):
        """
        Theta is N x 3
        """        
        theta = (theta*np.pi+ 1e-8)
        aa, bb, cc = tf.split(tf.expand_dims(theta,1), [1,1,1],2)
#        zero = 0*aa
#        ones = aa/aa
        zero = tf.zeros(aa.shape,dtype=tf.float32)
        ones = tf.ones(aa.shape,dtype=tf.float32)
        rot_mat = tf.concat([
                    tf.concat([tf.cos(bb)*tf.cos(cc), tf.cos(cc)*tf.sin(aa)*tf.sin(bb) - tf.cos(aa)*tf.sin(cc), tf.sin(aa)*tf.sin(cc) + tf.cos(aa)*tf.sin(bb)*tf.cos(cc),zero],2),
                    tf.concat([tf.cos(bb)*tf.sin(cc), tf.cos(aa)*tf.cos(cc) + tf.sin(aa)*tf.sin(bb)*tf.sin(cc), -tf.sin(aa)*tf.cos(cc) + tf.cos(aa)*tf.sin(bb)*tf.sin(cc),zero], 2),
                    tf.concat([-tf.sin(bb), tf.sin(aa)*tf.cos(bb), tf.cos(aa)*tf.cos(bb),zero], 2),
                    tf.concat([zero      ,zero       ,zero          , ones], 2)
                    ],1)
        rot_mat_inv = tf.transpose(rot_mat,(0,2,1))
        return rot_mat,rot_mat_inv
        
        
    def get_local_trans(self,bone_node,pose_node,representation='rodrigues'):
        local_trans = []
        for idx in np.arange(0,self.num_joints):
            trans = {}
            Bones = bone_node
            bone = tf.gather(Bones,[idx],axis=-1)
            pose = tf.squeeze(tf.gather(pose_node,[idx],axis=-1),-1)
            T,T_i = self.translation(bone)
            if representation=='rodrigues':
                R,R_i = self.rotation_rodrigues(pose)
            elif representation=='euler':
                R,R_i = self.rotation_euler(pose)
            trans['T'] = T
            trans['R'] = R
            trans['T_inv'] = T_i
            trans['R_inv'] = R_i            
            local_trans.append(trans)
        return local_trans

    def get_global_trans(self,LT):
        transforms_inv = []
        transforms = []
        for joint in np.arange(0,self.num_joints):
            current     = tf.eye(4,batch_shape=[self.batch_size],dtype=tf.float32)
            current_inv = tf.eye(4,batch_shape=[self.batch_size],dtype=tf.float32)
            node = joint
            while node!=-1:
                current_inv = tf.matmul(tf.matmul(current_inv,self.MT[node]['T_inv']),self.MT[node]['R_inv']) 
                current     = tf.matmul(LT[node]['R'],tf.matmul(self.MT[node]['R'],tf.matmul(LT[node]['T'],current)))
                node = self.model['topology'][0,node]
            transforms_inv.append(current_inv) 
            transforms.append(current)  
        return transforms,transforms_inv
        
    def apply_trans(self,features,meshify=False):

               
        batch_ = {}
        # Initial Conditions
        bone_node = features['bone_scale']*self.model['Bones']
        camera_offset = np.zeros((1,3,1),dtype=np.float32)
        camera_offset[0,1,0] = -0.6
        camera_offset[0,2,0] = 0.15
        camera_offset_tf = tf.constant(camera_offset,dtype=tf.float32) 
        camera_trans_node = features['camera_trans'] + camera_offset_tf
        pose_offset = np.zeros((1,3,self.num_joints),dtype=np.float32)
        pose_offset_tf = tf.constant(pose_offset,dtype=tf.float32) 
        pose_diff_node = features['pose_delta'] + pose_offset_tf

        # Get local transformations
        LT = self.get_local_trans(bone_node,pose_diff_node,representation='rodrigues')
        # Get global transformations
        transforms,transforms_inv = self.get_global_trans(LT)
        # Apply to landmarks
        marks_ = self.augment_landmarks(transforms,camera_trans_node)
        # Apply to mesh
        vertices_trans     = self.augment_mesh(transforms,transforms_inv,camera_trans_node,features['blendshapes'])
        voxels,vertices_MC = self.orthographic_reprojection(vertices_trans)

        batch_['landmarks']     = marks_
        batch_['mesh_vertices'] = vertices_trans
        batch_['reprojected']   = voxels   
#        batch_['LT']            = LT   
#        batch_['model_v2']      = self.model.pop('topology')
        return batch_








    def augment_mesh(self,transforms,transforms_inv,camera_trans_node,blendshapes_node):
        transforms_mesh = []
        vertices = self.meshObj.vertices
        vertices_shape   = vertices.get_shape().as_list()
        vertices_blended = tf.matmul(blendshapes_node,tf.reshape(vertices,(vertices_shape[0],-1)))
        vertices_blended = tf.reshape(vertices_blended,(self.batch_size,vertices_shape[1],vertices_shape[2]))
        
        for vertex in np.arange(0,self.meshObj.num_v):
            blend = self.meshObj.blend[vertex]
            trans = tf.zeros((self.batch_size,4,4),dtype=tf.float32)
            for bb in np.arange(0,len(blend)):
                node        = blend[bb]['joint']
                weight      = blend[bb]['weight']
                current     = transforms[node]
                current_inv = transforms_inv[node]
                trans = trans + tf.matmul(current,current_inv)*weight 
            trans = tf.expand_dims(trans,1)
            transforms_mesh.append(trans)
#            print(vertex)
        transforms_mesh = tf.concat(transforms_mesh,axis=1) 
        vertices_tiled = tf.tile(tf.expand_dims(vertices_blended,2),(1,1,4,1))
        vertices_trans = tf.reduce_sum(transforms_mesh*vertices_tiled,axis=-1) 
        vertices_trans = tf.slice(vertices_trans,[0,0,0],[-1,-1,3]) + tf.transpose(camera_trans_node,(0,2,1),name='Mesh_output')
        return vertices_trans



    def augment_landmarks(self,transforms,camera_trans_node):
        marks_trans = []
        for joint in np.arange(0,self.num_joints):
            current = tf.matmul(transforms[joint],self.marks_init)
            marks_trans.append(current) 
        marks_ = tf.slice(tf.concat(marks_trans,-1),[0,0,0],[-1,3,-1]) + camera_trans_node
        marks_ = tf.transpose(marks_,(0,2,1),name='Landmarks_output')
        return marks_





    def normalize_UVD(self,recon_params, norm_uvd, name='normalized_UVD_to_UVD'):
        '''
        Grabs the cropped normalized uvd parameters obtained from Yaron's preprocessing and computes original image UVD
        :param recon_parameters: tensor with shape [batch_size, 7]  representing reconstruction parameters given by Yaron's scripts
        :param norm_uvd: tensor with shape [batch_size, num_points, 3] representing normalized UVD coordinates
        :param name: Name
        :return:
        '''
        u = tf.gather(norm_uvd, indices=0, axis=-1)  # shape [batch_size, num_points]
        v = tf.gather(norm_uvd, indices=1, axis=-1)  # shape [batch_size, num_points]
        d = tf.gather(norm_uvd, indices=2, axis=-1)  # shape [batch_size, num_points]
        
        u -= tf.gather(recon_params, indices=[0], axis=-1)
        v -= tf.gather(recon_params, indices=[1], axis=-1)
        d -= tf.gather(recon_params, indices=[2], axis=-1)  
        u *= tf.gather(recon_params, indices=[3], axis=-1)
        v *= tf.gather(recon_params, indices=[4], axis=-1)
      
        d = (d / tf.gather(recon_params, indices=[6], axis=-1))*2 -1.
        u = (u / tf.gather(recon_params, indices=[5], axis=-1))*2 -1.
        v = (v / tf.gather(recon_params, indices=[5], axis=-1))*2 -1.
        
        uvd = tf.stack([u, v, d], axis=-1)
    
        return uvd

#    def camera_projection(self,vertices,vertices_uvd):
#        # Project
#        batch_idx_np = np.expand_dims(np.meshgrid(np.arange(0, self.batch_size, 1),np.arange(0, self.meshObj.num_v, 1))[0].transpose(),axis=2)
#        batch_idx = tf.reshape(tf.convert_to_tensor(batch_idx_np.astype(np.float32)),(self.batch_size*self.meshObj.num_v,1))
#        vertices_values  = tf.reshape(vertices,(-1,3))
#        vertices_indices = tf.reshape(vertices_uvd,(-1,3))
#        vertices_values     = (vertices_values/2+0.5)
#        vertices_indices    = (vertices_indices/2+0.5)*(self.grid_size-1)
#        image_mask = tf.logical_and(tf.logical_not(tf.reduce_any(tf.less_equal(vertices_indices,self.minValue),axis=1)),
#                                         tf.logical_not(tf.reduce_any(tf.greater_equal(vertices_indices,self.maxValue),axis=1)))
#        batch_idx             = tf.boolean_mask(batch_idx,image_mask)            
#        vertices_indices      = tf.boolean_mask(vertices_indices,image_mask) 
#        vertices_values       = tf.boolean_mask(vertices_values,image_mask) 
#        vertices_indices_4d = tf.concat((batch_idx,vertices_indices),axis=1)
#        indices, idx, count, count_normalized = self.get_centers(vertices_indices_4d)
#
#        vertices_indices_4d = tf.cast(tf.round(vertices_indices_4d),tf.int64)
#        count       = tf.expand_dims(tf.gather(count,idx),-1)
#        voxels      = tf.scatter_nd(vertices_indices_4d, -1*vertices_values/tf.cast(count,tf.float32), (self.batch_size,self.grid_size,self.grid_size,self.grid_size,3))
#        voxels      = -1*tf.reduce_min(voxels,axis=3) 
#        return voxels
    

    def orthographic_reprojection(self,vertices_trans):
        
        samples = self.meshObj.faces_MC
        samples = np.concatenate((samples[:,(0,1,2)],samples[:,(0,2,3)]),axis=0)
        samples_tri = tf.constant(samples)
        faces_values = tf.gather(vertices_trans,samples_tri,axis=1) # [batch, samples, edge, xyz ]
        
        
        V1 = tf.gather(faces_values,1,axis=2) - tf.gather(faces_values,0,axis=2)
        V2 = tf.gather(faces_values,2,axis=2) - tf.gather(faces_values,0,axis=2)
        V1x,V1y,V1z = tf.split(V1,[1,1,1],axis=2)
        V2x,V2y,V2z = tf.split(V2,[1,1,1],axis=2)
        normal_z = tf.reshape(V1x*V2y - V1y*V2x,(-1,))
        
        num_faces    = samples_tri.get_shape().as_list()[0]
        r1   = tf.random_uniform((self.batch_size,num_faces,1,1),dtype=tf.float32)
        r2   = tf.random_uniform((self.batch_size,num_faces,1,1),dtype=tf.float32)
        rand_samps = tf.concat(((1-tf.sqrt(r1)),(tf.sqrt(r1)*(1-r2)),(r2*tf.sqrt(r1))),axis=2)
        vertices_MC  = tf.reduce_sum(faces_values*rand_samps,axis=2)
        num_vertices = num_faces


        # Project
        batch_idx_np = np.expand_dims(np.meshgrid(np.arange(0, self.batch_size, 1),np.arange(0, num_vertices, 1))[0].transpose(),axis=2)
        batch_idx = tf.reshape(tf.convert_to_tensor(batch_idx_np.astype(np.float32)),(self.batch_size*num_vertices,1))
        vertices_trans_values = tf.reshape(vertices_MC,(-1,3))
        vertices_trans_values = (vertices_trans_values/2+0.5)
        image_mask = tf.logical_or(tf.reduce_any(tf.less_equal(vertices_trans_values,self.minValue),axis=1),
                                   tf.reduce_any(tf.greater_equal(vertices_trans_values,self.maxValue),axis=1))        
        
        image_mask = tf.logical_or(image_mask,tf.greater_equal(normal_z,0))
        image_mask = tf.tile(tf.expand_dims(image_mask,axis=-1),(1,3))
        vertices_trans_values  = tf.where(image_mask,0*tf.cast(image_mask,tf.float32),vertices_trans_values)

        vertices_trans_indices_4d = tf.concat((batch_idx,vertices_trans_values*(self.grid_size-1)),axis=1)
        vertices_trans_indices_4d = tf.cast(vertices_trans_indices_4d,tf.int64)
        point_cloud_lin_idx = tf.squeeze(ravel_index_bxy(tf.slice(vertices_trans_indices_4d,[0,0],[-1,3]),(self.batch_size,self.grid_size,self.grid_size)))
        vox_idx, idx, count = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
        indices = tf.transpose(unravel_index(vox_idx,(self.batch_size,self.grid_size,self.grid_size),Type=tf.int64),(1,0))
        values = -1*tf.unsorted_segment_max(tf.slice(-1*vertices_trans_values,[0,2],[-1,1]),idx,tf.reduce_max(idx)+1)
        voxels = tf.scatter_nd(indices, values, (self.batch_size,self.grid_size,self.grid_size,1))        

        return voxels,vertices_MC






    def get_centers(self,point_cloud_4d):
        point_cloud_idx = tf.cast(tf.round(point_cloud_4d),tf.int64)
        point_cloud_lin_idx = tf.squeeze(ravel_index(point_cloud_idx,self.grid_shape))
        vox_idx, idx, count = tf.unique_with_counts(point_cloud_lin_idx,out_idx=tf.int32)
        vox_idx = tf.cast(vox_idx,dtype = tf.int32)
        vox_mult_idx = tf.transpose(unravel_index(vox_idx,self.grid_shape),(1,0))
        batch_idx = tf.squeeze(tf.slice(vox_mult_idx,(0,0),(-1,1)))
        max_point_per_vol = tf.segment_max(count,batch_idx)
        max_point_per_vol = tf.gather(max_point_per_vol,batch_idx)
        count_normalized = tf.divide(count,max_point_per_vol)
        return vox_mult_idx, idx, count, count_normalized


    def normXYZ_to_normUVD(self,model,example):
        xyz = model['landmarks']
        xyz = (xyz+1.)/2
        xyz = xyz*self.box_limit +  (example['centers_xyz'] - self.box_limit/2)
        uvd = self.XYZ_to_UVD(tf.expand_dims(example['camera'],0), xyz)
        uvd = self.normalize_UVD(tf.expand_dims(example['recon_params'],0), uvd)
        uvd = tf.identity(uvd,name='Landmarks_uvd_output')
        model['landmarks_uvd'] = uvd
        xyz = model['mesh_vertices']
        xyz = (xyz+1.)/2
        xyz = xyz*self.box_limit +  (example['centers_xyz'] - self.box_limit/2)
        uvd = self.XYZ_to_UVD(tf.expand_dims(example['camera'],0), xyz)
        uvd = self.normalize_UVD(tf.expand_dims(example['recon_params'],0), uvd)
        uvd = tf.identity(uvd,name='Mesh_uvd_output')
        model['mesh_vertices_uvd'] = uvd        
        return model

    def XYZ_to_UVD(self,camera, xyz, name='xyz_to_uvd'):
        '''
        Projct XYZ to UVD point
        :param camera: tensor of shape [batch_size,3,3]
        :param xyz: tensor of shape [batch_size, num_points, 3]
        :return: tensor of uvd with shape [batch_size, num_points, 3]
        '''
        uvd = tf.matmul(xyz, camera, transpose_b=True)
        d = tf.gather(uvd, indices=[2], axis=-1)
        uv = tf.gather(uvd, indices=[0, 1], axis=-1) / d
        uvd = tf.concat([uv, d], axis=-1)
        return uvd

#    def XYZ_to_UVD(self,camera, uvd, name='uvd_to_xyz'):
#        '''
#        Project UVD to XYZ point
#        :param camera: tensor of shape [batch_size,3,3]
#        :param xyz: tensor of shape [batch_size, num_points, 3]
#        :return:
#        '''
#        xy = tf.gather(uvd, indices=[0, 1], axis=-1)
#        z = tf.gather(uvd, indices=[2], axis=-1)
#        fx = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=0, axis=1), indices=0, axis=1),axis=1),axis=1)
#        fy = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=1, axis=1), indices=1, axis=1),axis=1),axis=1)
#        cx = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=0, axis=1), indices=2, axis=1),axis=1),axis=1)
#        cy = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=1, axis=1), indices=2, axis=1),axis=1),axis=1)
#        x = tf.gather(xy, indices=[0], axis=-1)*fx/z + cx
#        y = tf.gather(xy, indices=[1], axis=-1)*fy/z + cy
#        xyz = tf.concat((x, y, z), axis=-1)
#        return xyz