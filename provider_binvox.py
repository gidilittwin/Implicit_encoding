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






    
class ShapeNet(object):
    def __init__(self , path_,files,rand,batch_size=16,grid_size=32,levelset=0.0,num_samples=1000,list_=['02691156'],type_='train',rec_mode=False):
        self.path_ = path_
        
        

        self.train_paths = self.getBenchmark(files)            
#        self.train_paths = self.getModelPaths(type_,list_=list_)
        self.train_size  = len(self.train_paths)
        self.train_files,self.train_image_files = self.getModelFiles()
        self.epoch = 0
        self.rand = rand
        self.reset()
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.batch=np.linspace(0, batch_size-1, batch_size)
        self.x   = np.linspace(-1, 1, grid_size)
        self.y   = np.linspace(-1, 1, grid_size)
        self.z   = np.linspace(-1, 1, grid_size)
        self.grid = np.stack(np.meshgrid(self.x,self.y,self.z),axis=-1)
        self.levelset    = levelset
        self.rec_mode    = rec_mode
        self.num_samples = num_samples/len(self.levelset)

    def getModelPaths(self,type_,list_):
        paths = []
        for i in range(len(list_)):
            prefix = self.path_ + list_[i]+'/'+ type_ +'/'
            paths_cat = glob.glob(os.path.join(prefix, '*'))
#            paths_cat = paths_cat[0:100]
            paths = paths+paths_cat
        return  paths  

    def getBenchmark(self,files):
        paths = []
        paths.append('')
        with open(files, 'r') as file:
            all_lines = file.readlines() 
            for line in all_lines:
                cat = line[19:27]
                name = line[28:-8]
                path = self.path_+ cat+'/'+name +'/'
                if paths[-1]!=path:
                    paths.append(path)
        return  paths[1:]  
    
    def getModelFiles(self):
        vox_files = []
        image_files = []
        name = '/model.binvox'
        for i in range(len(self.train_paths)):
            prefix = self.train_paths[i]
            vox_file = prefix+name
            images = glob.glob(os.path.join(prefix, 'rendering/*.png'))
            vox_files.append(vox_file)
#            image_files.append([images[0]])
            image_files.append(images)
        return  vox_files,  image_files

 
    
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
        files = [self.train_files[j] for j in indexes]
        image_files = [self.train_image_files[j] for j in indexes]
        code   = np.zeros((self.batch_size,self.train_size),dtype=np.int64)
        voxels = []
#        sdf    = []
        images = []
        alpha = []
        vertices = []
        for j in range(size):
            with open(files[j], 'rb') as f:
                m1 = binvox_rw.read_as_3d_array(f)
                voxels_ = m1.data
                voxels_ = voxels_ + np.flip(voxels_,2)
                voxels_ = np.pad(voxels_, pad_width=2,mode='constant', constant_values=False)
                voxels_ = -1.0*voxels_.astype(np.float32)+0.5
                voxels.append(voxels_)
#                inner_volume       = voxels_
#                outer_volume       = np.logical_not(voxels_)
#                sdf_o, closest_point_o = ndi.distance_transform_edt(outer_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
#                sdf_i, closest_point_i = ndi.distance_transform_edt(inner_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
#                sdf_                 = (sdf_o - sdf_i)/(self.grid_size-1)*2  
                
                if self.rec_mode:
#                    np.save(files[j][0:-12]+'sdf.npy',sdf_)
                    for ll in range(len(self.levelset)):
                        verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_, self.levelset[ll])
                        np.save(files[j][0:-12]+'verts'+str(ll)+'.npy',verts)
                        np.save(files[j][0:-12]+'faces'+str(ll)+'.npy',faces)
                        np.save(files[j][0:-12]+'normals'+str(ll)+'.npy',normals)
                else:
                    Verts = []
                    for ll in range(len(self.levelset)):
                        verts = np.load(files[j][0:-12]+'verts'+str(ll)+'.npy')
                        num_points = verts.shape[0]
                        arr_ = np.arange(0,num_points)
                        perms = np.random.choice(arr_,self.num_samples)
                        verts_sampled = verts[perms,:]
                        Verts.append(verts_sampled[:,(2,0,1)])
                    vertices.append(np.stack(Verts,axis=-1))
#                sdf.append(sdf_) 
            if self.rand==False: 
                image_file_rand = np.random.randint(0,len(image_files[j]))   
            else:
                image_file_rand = (self.epoch-1) % len(image_files[j])

            with open(image_files[j][image_file_rand], 'rb') as f:
                image = misc.imread(f).astype(np.float32)
                images.append(image[:,:,0:3])
                alpha.append(image[:,:,3:4])

        voxels = np.transpose(np.stack(voxels,axis=0),(0,1,3,2))
#        sdf    = np.transpose(np.stack(sdf,axis=0),(0,1,3,2))
        images = np.stack(images,axis=0)
        alpha  = np.stack(alpha,axis=0)  
        if self.rec_mode==False:
            vertices = np.stack(vertices,axis=0)
        rows = np.arange(0,self.batch_size)
        code[rows,indexes] = 1

        return {'sdf':voxels,'code':code,'indexes':np.expand_dims(indexes,axis=1),'images':images,'alpha':alpha,'vertices':vertices}




    def process_batch(self,batch):
    #    samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(1,num_samples/10,3))
    #    samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(grid_size-1))).astype(np.int32)
    #    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)
    #    samples_xyz_np       = np.tile(samples_xyz_np,(BATCH_SIZE,1,1))
    #    samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(BATCH_SIZE,global_points,3))
    
    
    
    
#        vertices             = np.concatenate((batch['vertices'][:,:,:,0],batch['vertices'][:,:,:,1]),axis=1)/(self.grid_size-1)*2-1
        vertices             = batch['vertices'][:,:,:,0]/(self.grid_size-1)*2-1
        gaussian_noise       = np.random.normal(loc=0.0,scale=0.1,size=vertices.shape).astype(np.float32)
        samples_xyz_np       = np.clip((vertices+gaussian_noise),-1.0,1.0)
    #    samples_xyz_np       = np.concatenate((samples_xyz_np,vertices),axis=1)
        samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(self.grid_size-1))).astype(np.int32)
    #    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)
        batch_idx            = np.tile(np.reshape(np.arange(0,self.batch_size,dtype=np.int32),(self.batch_size,1,1)),(1,self.num_samples,1))
        samples_ijk_np       = np.reshape(np.concatenate((batch_idx,samples_ijk_np),axis=-1),(self.batch_size*(self.num_samples),4))
        samples_sdf_np       = np.reshape(batch['sdf'][samples_ijk_np[:,0],samples_ijk_np[:,2],samples_ijk_np[:,1],samples_ijk_np[:,3]],(self.batch_size,self.num_samples,1))

        return {'samples_xyz_np':samples_xyz_np,'samples_sdf_np':samples_sdf_np}







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
    
    
    