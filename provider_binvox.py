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
    def __init__(self , path_,rand,batch_size=16,grid_size=32,levelset=0.0,num_samples=1000,list_=['02691156'],type_='train',rec_mode=False,binvox='/model.binvox',obj=''):
        self.path_ = path_
        self.binvox = binvox
        self.obj    = obj
        self.train_paths = self.getModelPaths(type_,list_=list_)
        self.train_size  = len(self.train_paths)
        self.train_files,self.train_image_files = self.getModelFiles()
        self.rand = rand
        self.reset()
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.batch=np.linspace(0, batch_size-1, batch_size)
        self.x   = np.linspace(-1, 1, grid_size)
        self.y   = np.linspace(-1, 1, grid_size)
        self.z   = np.linspace(-1, 1, grid_size)
        self.grid = np.stack(np.meshgrid(self.x,self.y,self.z),axis=-1)
        self.levelset    = 0.00
        self.rec_mode    = rec_mode
        self.num_samples = num_samples

    def getModelPaths(self,type_,list_):
        paths = []
        for i in range(len(list_)):
            prefix = self.path_ + list_[i]+'/'+ type_ +'/'
            paths_cat = glob.glob(os.path.join(prefix, '*'))
            paths_cat = paths_cat[0:1]
            paths = paths+paths_cat
        return  paths  

    def getModelFiles(self):
        paths = self.train_paths
        vox_files   = []
        image_files = []
        obj_files   = []
        binvox_name = self.binvox
        obj_name    = self.obj
        for i in range(len(paths)):
            prefix = paths[i]
            vox_file = prefix+binvox_name
            obj_file = prefix+obj_name
            images = glob.glob(os.path.join(prefix, 'rendering/*.png'))
#            all_files = all_files+  [x + name for x in files]     
            vox_files.append(vox_file)
            image_files.append(images)
            obj_files.append(obj_file)
        return  vox_files,  image_files

 
    
    def reset(self):
        if self.rand==False:
            self.train_idx = np.arange(self.train_size)
        else:
            self.train_idx = np.random.permutation(self.train_size)
        self.train_step = 0    

        
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
        sdf    = []
        images = []
        alpha = []
        vertices = []
        for j in range(size):
            with open(files[j], 'rb') as f:
                m1 = binvox_rw.read_as_3d_array(f)
                voxels_ = m1.data
                voxels_ = voxels_ + np.flip(voxels_,2)
                voxels_ = np.pad(voxels_, pad_width=2,mode='constant', constant_values=False)
                voxels.append(voxels_)
                
                inner_volume       = voxels_
                outer_volume       = np.logical_not(voxels_)
                sdf_o, closest_point_o = ndi.distance_transform_edt(outer_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
                sdf_i, closest_point_i = ndi.distance_transform_edt(inner_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
                sdf_                 = (sdf_o - sdf_i)/(self.grid_size-1)*2
                if self.rec_mode:
                    verts, faces, normals, values = measure.marching_cubes_lewiner(sdf_, 0.0)
                    np.save(files[j][0:-12]+'verts.npy',verts)
                    np.save(files[j][0:-12]+'faces.npy',faces)
                    np.save(files[j][0:-12]+'normals.npy',normals)
#                else:
#                    verts = np.load(files[j][0:-12]+'verts.npy')
#                    num_points = verts.shape[0]
#                    arr_ = np.arange(0,num_points)
#                    perms = np.random.choice(arr_,self.num_samples)
#                    verts_sampled = verts[perms,:]
#                    vertices.append(verts_sampled[:,(2,0,1)])
                sdf.append(sdf_) 
                
            image_file_rand = np.random.randint(0,len(image_files[j]))   
            with open(image_files[j][image_file_rand], 'rb') as f:
                image = misc.imread(f).astype(np.float32)
                images.append(image[:,:,0:3])
                alpha.append(image[:,:,3:4])

        voxels = np.transpose(np.stack(voxels,axis=0),(0,1,3,2))
        sdf    = np.transpose(np.stack(sdf,axis=0),(0,1,3,2))
        images = np.stack(images,axis=0)
        alpha  = np.stack(alpha,axis=0)  
#        if self.rec_mode==False:
#            vertices = np.stack(vertices,axis=0)
        
        rows = np.arange(0,self.batch_size)
        code[rows,indexes] = 1

        return {'voxels':voxels,'sdf':sdf,'code':code,'indexes':np.expand_dims(indexes,axis=1),'images':images,'alpha':alpha}


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
    
    
    