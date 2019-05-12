import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import glob
import tensorflow as tf
import src.utilities.binvox_rw as binvox_rw
import scipy.ndimage as ndi
from scipy import misc
from skimage import measure
import scipy.io
import skimage.transform 





    
class ShapeNet(object):
    def __init__(self , path_, mesh_path_, files, rand, batch_size=16, grid_size=32, levelset=0.0, num_samples=1000, list_=['02691156'],rec_mode=False,shuffle_rgb=False,reduce=1):
        self.path_      = path_
        self.mesh_path_ = mesh_path_
        self.grid_size = grid_size
        self.binvox_32  = '/model.binvox'
        self.binvox_128 = '/models/model_normalized.solid.binvox'
        self.binvox_256 = '/model.mat'
        self.reduce=reduce
        if grid_size != 256:
            self.train_paths = self.getBenchmark(files,list_)  
        else:
            self.train_paths = self.getBenchmarkICCV(files,list_)  
#        self.train_paths = self.train_paths[0:8]  
#        self.train_paths = self.getModelPaths(list_=list_)
        self.train_size  = len(self.train_paths)
        self.train_files,self.train_image_files, self.cam_params ,self.train_classes = self.getModelFiles(list_)
        self.obj_map = {}
        self.obj_id = np.arange(0,len(self.train_files))
        self.epoch = 0
        self.rand = rand
        self.reset()
        self.batch_size = batch_size
        self.batch=np.linspace(0, batch_size-1, batch_size)
        self.x   = np.linspace(-1, 1, grid_size)
        self.y   = np.linspace(-1, 1, grid_size)
        self.z   = np.linspace(-1, 1, grid_size)
        self.grid = np.stack(np.meshgrid(self.x,self.y,self.z),axis=-1)
        self.levelset    = levelset
        self.rec_mode    = rec_mode
        self.num_samples = num_samples/len(self.levelset)
        self.shuffle_rgb = shuffle_rgb
        if self.grid_size==36:
            self.string_len = len(self.binvox_32)-1
        elif self.grid_size==132:
            self.string_len = len(self.binvox_128)-8
        elif self.grid_size==256:
            self.string_len = 9
        
        

    def getModelPaths(self,list_):
        paths = []
        for i in range(len(list_)):
            prefix = self.path_ + list_[i] +'/'
            paths_cat = glob.glob(os.path.join(prefix, '*'))
            paths_cat = paths_cat[0:1]
            paths = paths+paths_cat
        return  paths  

    def getBenchmark(self,files,list_):
        paths = []
        paths.append('')
        with open(files, 'r') as file:
            all_lines = file.readlines() 
            for line in all_lines:
                cat = line[19:27]
                if cat in list_:
                    name = line[28:-8]
#                    path = self.path_+ cat+'/'+name +'/'
                    path = cat+'/'+name +'/'
                    if paths[-1]!=path:
                        paths.append(path)
        return  paths[1:]  

#    def getBenchmarkICCV(self,files,list_):
#        paths = os.listdir(self.path_ +'_imgs/') 
#        return  paths
    
    def getBenchmarkICCV(self,files,list_):
        paths = []
        for cat in list_:
            file_ = self.path_+'_'+cat+'.txt'
            with open(file_, 'r') as f:
                data = f.readlines()
                data_ = [cat+'/'+s[0:-5] for s in data]
            paths = paths+data_
        return  paths 




    def getModelFiles(self,list_):
        paths = self.train_paths
        vox_files   = []
        image_files = []
        cam_params  = []
        classes    = []
        classes_     = {}
        for idx,key in enumerate(list_):
            classes_[key] = idx
        
        for i in range(len(paths)):
            prefix = paths[i]
            if self.grid_size==36:
                vox_file  = self.path_+prefix+self.binvox_32
                images = glob.glob(os.path.join(self.path_,prefix, 'rendering/*.png'))
                meta = np.loadtxt(self.path_+ prefix+'/rendering/rendering_metadata.txt')    
                first_slash = prefix.find('/')
                cat = prefix[0:first_slash]
                class_ = classes_[cat]
            elif self.grid_size==132:
                vox_file = self.mesh_path_+prefix+self.binvox_128
                images   = glob.glob(os.path.join(self.path_,prefix, 'rendering/*.png'))
                meta     = np.loadtxt(self.path_+ prefix+'/rendering/rendering_metadata.txt')                
            elif self.grid_size==256:
                last_slash = self.path_.rfind('/')
                vox_file = self.path_[0:last_slash]+'/modelBlockedVoxels256/'+prefix+'.mat'    
                images   = glob.glob(os.path.join(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+prefix ,'*.png'))
                images = [s for s in images if 'render_' in s]
                first_slash = prefix.find('/')
                cat = prefix[0:first_slash]
                class_ = classes_[cat]                
                meta     = ''
            classes.append(class_)
            cam_params.append(meta)
            vox_files.append(vox_file)
            image_files.append(images)
#            image_files.append([images[0]])
        return  vox_files, image_files, cam_params, classes
    
    
    def unit(self,v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm    
    
    def camera_info(self,param):
        theta = np.deg2rad(param[0])
        phi = np.deg2rad(param[1])
        camY = param[3]*np.sin(phi)
        temp = param[3]*np.cos(phi)
        camX = temp * np.cos(theta)    
        camZ = temp * np.sin(theta)        
        cam_pos = np.array([camX, camY, camZ])        
        axisZ = cam_pos.copy()
        axisY = np.array([0,1,0])
        axisX = np.cross(axisY, axisZ)
        axisY = np.cross(axisZ, axisX)
        cam_mat = np.array([self.unit(axisX), self.unit(axisY), self.unit(axisZ)])
        return cam_mat, cam_pos

    
    def reset(self):
        if self.rand==False:
            self.train_idx = np.arange(self.train_size)
        else:
            self.train_idx = np.random.permutation(self.train_size)
        self.train_step = 0  
        self.epoch+=1
        
    def get_batch(self,type_):
        size = self.batch_size
        if self.train_step+size>self.train_size:
            self.reset()
        indexes = np.arange(self.train_step,self.train_step+size)    
        indexes = indexes%self.train_size
        indexes = self.train_idx[indexes]
        self.train_step = self.train_step+size
        files         = [self.train_files[j] for j in indexes]
        train_classes = [self.train_classes[j] for j in indexes]
        image_files   = [self.train_image_files[j] for j in indexes]
        train_paths   = [self.train_paths[j] for j in indexes]
        train_id      = [self.obj_id[j] for j in indexes]
        image_id    = []
        voxels      = []
        sdf         = []
        images      = []
        camera_mat  = []
        camera_pose = []        
        alpha       = []
        vertices    = []
        last_slash  = self.path_.rfind('/')
        for j in range(size):
            if self.grid_size!=256:
                with open(files[j], 'rb') as f:
                    m1 = binvox_rw.read_as_3d_array(f)
                    voxels_b = m1.data
                    voxels_b = np.pad(voxels_b, pad_width=2,mode='constant', constant_values=False)
            else:
#                m1 = scipy.io.loadmat(files[j])
                m1 = np.load(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+train_paths[j]+'/voxels0.npz')
                voxels_b = m1['voxels']   
                voxels_b = np.transpose(voxels_b,(0,2,1))
            voxels_ = -1.0*voxels_b.astype(np.float32)+0.5
            sdf_    = voxels_
            sdf.append(sdf_) 
            voxels.append(voxels_b)
#            inner_volume       = voxels_
#            outer_volume       = np.logical_not(voxels_)
#            sdf_o = ndi.distance_transform_edt(outer_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#            sdf_i = ndi.distance_transform_edt(inner_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#            sdf_                 = (sdf_o - sdf_i)/(self.grid_size-1)*2  
            
            Verts = []
            for ll in range(len(self.levelset)):
                if self.grid_size!=256:
                    verts = np.load(files[j][0:-self.string_len]+'verts'+str(ll)+'.npy')
                else:
                    verts   = np.load(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+train_paths[j]+'/verts0.npy')
                    verts = verts[:,(0,2,1)]
                num_points = verts.shape[0]
                arr_ = np.arange(0,num_points)
                perms = np.random.choice(arr_,self.num_samples)
                verts_sampled = verts[perms,:]
                Verts.append(verts_sampled[:,(2,0,1)])
            vertices.append(np.stack(Verts,axis=-1))
            if self.rand==True: 
                image_file_rand = np.random.randint(0,len(image_files[j]))   
            else:
                image_file_rand = (self.epoch-1) % len(image_files[j])
            image_id.append(image_file_rand)
            with open(image_files[j][image_file_rand], 'rb') as f:
                image = misc.imread(f).astype(np.float32)
                rgb   = image[:,:,0:3]
                alph  = image[:,:,3:4]
                if self.shuffle_rgb and self.rand==False:
                     rgb = rgb[:,:,np.random.permutation(3)]
                images.append(np.concatenate((rgb,alph),axis=-1))
                alpha.append(image[:,:,3:4])
            if self.grid_size!=256:
                params = self.cam_params[j][image_file_rand,:]
                cam_mat,cam_pose = self.camera_info(params)
                camera_mat.append(cam_mat)
                camera_pose.append(cam_pose)
        voxels   = np.transpose(np.stack(voxels,axis=0),(0,1,3,2))
        sdf      = np.transpose(np.stack(sdf,axis=0),(0,1,3,2))
        image_id = np.stack(image_id,axis=0)
        images   = np.stack(images,axis=0)
        alpha    = np.stack(alpha,axis=0)  
        classes  = np.stack(train_classes,axis=0)  
        ids      = np.stack(train_id,axis=0)  
        if self.grid_size!=256:
            camera_mat    = np.stack(camera_mat,axis=0)  
            camera_pose   = np.stack(camera_pose,axis=0)          
        vertices = np.stack(vertices,axis=0)
        return {'classes':classes,'ids':ids,'voxels':voxels,'sdf':sdf,'indexes':np.expand_dims(indexes,axis=1),'images':images,'alpha':alpha,'vertices':vertices,'camera_mat':camera_mat,'camera_pose':camera_pose}

    def get_batch_multi(self,type_):
        size = 1
        if self.train_step+size>self.train_size:
            self.reset()
        indexes = self.train_step  
        indexes = indexes%self.train_size
        indexes = self.train_idx[indexes]
        self.train_step = self.train_step+size
        files         = [self.train_files[indexes]]
        train_classes = [self.train_classes[indexes]]
        image_files   = [self.train_image_files[indexes]]
        train_paths   = [self.train_paths[indexes]]
        train_id      = [self.obj_id[indexes]]
        voxels      = []
        sdf         = []
        images      = []
        alpha       = []
        vertices    = []
        last_slash  = self.path_.rfind('/')
        j=0
        if self.grid_size!=256:
            with open(files[j], 'rb') as f:
                m1 = binvox_rw.read_as_3d_array(f)
                voxels_b = m1.data
                voxels_b = np.pad(voxels_b, pad_width=2,mode='constant', constant_values=False)
        else:
            m1 = np.load(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+train_paths[j]+'/voxels0.npz')
            voxels_b = m1['voxels']   
            voxels_b = np.transpose(voxels_b,(0,2,1))
            
        if self.reduce!=1:
            voxels_b = measure.block_reduce(voxels_b, (self.reduce,self.reduce,self.reduce), np.max)
#            voxels_b = 1.0*voxels_b>0.5
            
        voxels_ = -1.0*voxels_b.astype(np.float32)+0.5
        sdf_    = voxels_
        sdf.append(sdf_) 
        voxels.append(voxels_b)
        Verts = []
        for ll in range(len(self.levelset)):
            if self.grid_size!=256:
                verts = np.load(files[j][0:-self.string_len]+'verts'+str(ll)+'.npy')
            else:
                verts   = np.load(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+train_paths[j]+'/verts0.npy')
                verts = verts[:,(0,2,1)]
            num_points = verts.shape[0]
            arr_ = np.arange(0,num_points)
            perms = np.random.choice(arr_,self.num_samples)
            verts_sampled = verts[perms,:]
            Verts.append(verts_sampled[:,(2,0,1)])
        vertices.append(np.stack(Verts,axis=-1))
        perm = np.random.permutation(len(image_files[0]))        
        for im in range(self.batch_size):
            with open(image_files[0][perm[im]], 'rb') as f:
                image = (misc.imread(f).astype(np.float32)/255.)*2-1.
                image = ((skimage.transform.resize(image, (137,137))+1.)/2.*255.).astype(np.float32)
                rgb   = image[:,:,0:3]
                alph  = image[:,:,3:4]
#                if self.shuffle_rgb and self.rand!=False:
#                     rgb = rgb[:,:,np.random.permutation(3)]
                images.append(np.concatenate((rgb,alph),axis=-1))
                alpha.append(image[:,:,3:4])

        voxels   = np.tile(np.transpose(np.stack(voxels,axis=0),(0,1,3,2)),(self.batch_size,1,1,1))
        sdf      = np.tile(np.transpose(np.stack(sdf,axis=0),(0,1,3,2)),(self.batch_size,1,1,1))
        images   = np.stack(images,axis=0)
        alpha    = np.stack(alpha,axis=0)  
        classes  = np.tile(np.stack(train_classes,axis=0) ,(self.batch_size,1)) 
        ids      = np.tile(np.stack(train_id,axis=0)  ,(self.batch_size,1))         
        vertices = np.tile(np.stack(vertices,axis=0) ,(self.batch_size,1,1,1))         
        return {'classes':classes,'ids':ids,'voxels':voxels,'sdf':sdf,'indexes':np.expand_dims(indexes,axis=1),'images':images,'alpha':alpha,'vertices':vertices}





    def preprocess(self,type_):
        size = self.batch_size
        if self.train_step+size>self.train_size:
            self.reset()
        indexes = np.arange(self.train_step,self.train_step+size)    
        indexes = indexes%self.train_size
        indexes = self.train_idx[indexes]
        self.train_step = self.train_step+size
        files = [self.train_files[j] for j in indexes]
        for j in range(size):
            try:
                with open(files[j], 'rb') as f:
                    m1 = binvox_rw.read_as_3d_array(f)
                    voxels_ = m1.data
#                    voxels_ = np.pad(voxels_, pad_width=2,mode='constant', constant_values=False)
#                    for ll in range(len(self.levelset)):
#                        verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_,0.5)
            except:
                print('voxel file:' + files[j] + ' is missing')  
                aa.aa=1
                voxels_ = []
                verts   = np.zeros(shape=(10,3),dtype=np.float32)
#            np.save(files[j][0:-self.string_len]+'verts'+str(ll)+'.npy',verts)
#            np.save(files[j][0:-self.string_len]+'faces'+str(ll)+'.npy',faces)
#            np.save(files[j][0:-self.string_len]+'normals'+str(ll)+'.npy',normals)
#        return {'voxels':voxels_,'vertices':verts}



    def preprocess_iccv(self,type_):
        size = self.batch_size
        if self.train_step+size>self.train_size:
            self.reset()
        indexes = np.arange(self.train_step,self.train_step+size)    
        indexes = indexes%self.train_size
        indexes = self.train_idx[indexes]
        ii,jj,kk = np.meshgrid(np.arange(0,16),np.arange(0,16),np.arange(0,16))
        ii = np.reshape(ii,(-1))*16
        jj = np.reshape(jj,(-1))*16
        kk = np.reshape(kk,(-1))*16
        idx = np.arange(0,4096)
        self.train_step = self.train_step+size
        files = [self.train_files[j] for j in indexes]
        paths = [self.train_paths[j] for j in indexes]
        last_slash = self.path_.rfind('/')
        for j in range(size):
            try:
                m1     = scipy.io.loadmat(files[j])
                grid   = np.reshape((m1['bi']-1).astype(np.int64),-1)
                blocks = m1['b'].astype(np.bool)
                voxels = np.zeros((256,256,256),dtype=np.bool)
                voxels_ = blocks[grid,:,:,:]
                for bb in idx:
                    voxels[jj[bb]:jj[bb]+16,ii[bb]:ii[bb]+16,kk[bb]:kk[bb]+16] = voxels_[idx[bb],:,:,:]
                voxels = np.transpose(voxels,(0,2,1))
                verts, faces, normals, values = measure.marching_cubes_lewiner(voxels,0.5)
            except:
                print('voxel file:' + files[j] + ' is missing')  
                voxels  = np.zeros((256,256,256),dtype=np.bool)
                verts   = np.zeros(shape=(10,3),dtype=np.float32)
                np.save(self.path_[0:last_slash]+'/blenderRenderPreprocess/'+paths[j]+'/verts0'+'.npy',verts)
                np.savez_compressed( self.path_[0:last_slash]+'/blenderRenderPreprocess/'+paths[j]+'/voxels0.npz', voxels=voxels)
#        return {'voxels':[],'vertices':[]}
        return {'voxels':voxels,'vertices':verts}


    def process_batch(self,batch,config):
        samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(1,config.global_points,3)).astype(dtype=np.float32)
#        samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
#        samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)
        samples_xyz_np       = np.tile(samples_xyz_np,(self.batch_size,1,1))
#        samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(BATCH_SIZE,global_points,3))

#        vertices             = np.concatenate((batch['vertices'][:,:,:,0],batch['vertices'][:,:,:,1]),axis=1)/(self.grid_size-1)*2-1
        vertices             = batch['vertices'][:,:,:,0]/(self.grid_size-1)*2-1
        gaussian_noise       = np.random.normal(loc=0.0,scale=config.noise_scale,size=vertices.shape).astype(np.float32)
        vertices             = np.clip((vertices+gaussian_noise),-1.0,1.0)
        
        samples_xyz_np       = np.concatenate((samples_xyz_np,vertices),axis=1)
        samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(self.grid_size-1))).astype(np.int32)
    #    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)
        batch_idx            = np.tile(np.reshape(np.arange(0,self.batch_size,dtype=np.int32),(self.batch_size,1,1)),(1,self.num_samples+config.global_points,1))
        samples_ijk_np       = np.reshape(np.concatenate((batch_idx,samples_ijk_np),axis=-1),(self.batch_size*(self.num_samples+config.global_points),4))
        samples_sdf_np       = np.reshape(batch['sdf'][samples_ijk_np[:,0],samples_ijk_np[:,2],samples_ijk_np[:,1],samples_ijk_np[:,3]],(self.batch_size,self.num_samples+config.global_points,1))
        
        images = batch['images']
        if config.rgba==0:
            images = images[:,:,:,0:3]
        
        return {'samples_xyz_np':samples_xyz_np,'samples_sdf_np':samples_sdf_np,'images':images}







    def convert2np(self,type_,up_samp):
        list_ = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
        for i in range(10):
            files_path = self.path_ + list_[i] + '/' + type_ + '/'
            print(str(i))
            for file_name in glob.glob(files_path+'*.off'):

                with open(file_name, 'r') as file:
                    all_lines = file.readlines() 
                    splits  = all_lines[1].split()
                    num_vert = int(splits[0])
                    num_faces = int(splits[1])
                    vertices = (all_lines[2:2+num_vert])
                    vertices = np.array(map(conv_v, vertices))
                    faces = (all_lines[2+num_vert:2+num_vert+num_faces])
                    faces = np.array(map(conv_f, faces)).astype(np.int64)
                    
                    V1 = vertices[faces[:,1]]-vertices[faces[:,0]]
                    V2 = vertices[faces[:,2]]-vertices[faces[:,0]]
                    area = np.linalg.norm(np.cross(V1,V2),axis=1)/2
                    distribution = area/np.sum(area)
                    rand_samps = np.random.choice(faces.shape[0], size=(up_samp,), replace=True, p=distribution)
                    faces_up = faces[rand_samps,:]                    
                    distribution_up  = distribution[rand_samps]                    

                    data = {}
                    data['vertices'] = vertices
                    data['faces']    = faces
                    data['faces_up']    = faces_up
                    data['faces_up_area']    = distribution_up
                    out_name = file_name.replace('/train/','/train_np/')
                    out_name = out_name.replace('.off','.npy')
                    save_file_to_log(data, out_name)
                    
    def convert2tf(self,batch):
       batch_tf = []
       for ii in np.arange(self.batch_size) :
           mesh = batch[ii]
           mesh_tf = {}
           for key, value in mesh.iteritems():
               mesh_tf[key] = tf.constant(value)
           batch_tf.append(mesh_tf)
           
       return  batch_tf
        
                
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.02, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def scale_point_cloud(batch_data, scale_ratio=0.8):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    scale = np.random.uniform(scale_ratio,1/scale_ratio,(B,1,1))
    scaled_data = batch_data*scale
    return scaled_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]




    
    
def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)







def generate_polys(BATCH_SIZE,grid_size):
    
    v0 = np.random.uniform(low=0.0, high=grid_size-1, size=( BATCH_SIZE,3,1))
    v1 = np.random.uniform(low=0.0, high=grid_size-1, size=( BATCH_SIZE,3,1))
    v2 = np.random.uniform(low=0.0, high=grid_size-1, size=( BATCH_SIZE,3,1))
    data = np.reshape(np.concatenate((v0,v1,v2),axis=2),(BATCH_SIZE,1,9))
    
    color = np.transpose(np.concatenate((v0[:,2:3,:],v1[:,2:3,:],v2[:,2:3,:]),axis=-1)   ,(0,2,1)) 
    X,Y = np.meshgrid(np.linspace(0,grid_size-1,grid_size),np.linspace(0,grid_size-1,grid_size))    
    pp = np.concatenate((np.reshape(X,(1,1,-1)) ,np.reshape(Y,(1,1,-1))),axis=1) 
    area =  (v2[:,0:1,:]-v0[:,0:1,:])*(v1[:,1:2,:]-v0[:,1:2,:]) - (v2[:,1:2,:]-v0[:,1:2,:])*(v1[:,0:1,:]-v0[:,0:1,:])  
    w0   =  (pp[:,0:1,:]-v1[:,0:1,:])*(v2[:,1:2,:]-v1[:,1:2,:]) - (pp[:,1:2,:]-v1[:,1:2,:])*(v2[:,0:1,:]-v1[:,0:1,:])  
    w1   =  (pp[:,0:1,:]-v2[:,0:1,:])*(v0[:,1:2,:]-v2[:,1:2,:]) - (pp[:,1:2,:]-v2[:,1:2,:])*(v0[:,0:1,:]-v2[:,0:1,:])  
    w2   =  (pp[:,0:1,:]-v0[:,0:1,:])*(v1[:,1:2,:]-v0[:,1:2,:]) - (pp[:,1:2,:]-v0[:,1:2,:])*(v1[:,0:1,:]-v0[:,0:1,:])  
    W = np.concatenate((w0,w1,w2),axis=1)/area
    images = np.reshape(np.sum(np.all(W>0,axis=1,keepdims=True)*W*color,1),(BATCH_SIZE,grid_size,grid_size))
    return data,images  
    
    
    