import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import glob
import tensorflow as tf


# Download dataset for point cloud classification
#DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#    zipfile = os.path.basename(www)
#    os.system('wget %s; unzip %s' % (www, zipfile))
#    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#    os.system('rm %s' % (zipfile))
DATA_DIR = '/Users/gidilittwin/Dropbox/Thesis/ModelNet/Data/ModelNet40/'

def conv_v(mystr):
    return map(float, mystr.split()[:3])

def conv_f(mystr):
    return map(float, mystr.split()[1:4])

def save_file_to_log(struct, path):
#    if not os.path.exists(path):
#        os.makedirs(path)
    np.save(path,struct)
    
class ModelNet10(object):
    def __init__(self , path_,rand,batch_size=16,num_points=10000,multipier=1):
        self.path_ = path_
        self.train_files = self.getModelNet10Files('train_np')
        self.test_files  = self.getModelNet10Files('test_np')
        self.train_size  = len(self.train_files)
        self.test_size   =  len(self.test_files)      
        self.rand = rand
        self.reset()
        self.up_samp = num_points*multipier
        self.multipier = multipier
        self.batch_size = batch_size

    def getModelNet10Files(self,type_):
        list_ = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
        all_files = []
        for i in range(10):
            files_path = self.path_ + list_[i] + '/' + type_ + '/'
            for file in glob.glob(files_path+'*.npy'):
                all_files.append(file) 
        return  all_files   

    def reset(self):
        if self.rand==False:
            self.train_idx = np.arange(self.train_size)
            self.test_idx = np.arange(self.test_size)
        else:
            self.train_idx = np.random.permutation(self.train_size)
            self.test_idx = np.random.permutation(self.test_size)
        self.train_step = 0    
        self.test_step  = 0 
        
        
    def get_batch(self,type_):
        size = self.batch_size
        if type_=='train':
            indexes = np.arange(self.train_step,self.train_step+size)    
            indexes = indexes%self.train_size
            indexes = self.train_idx[indexes]
            self.train_step = self.train_step+size
            files = [self.train_files[j] for j in indexes]
        elif type_=='test':
            indexes = np.arange(self.test_step,self.test_step+size)    
            indexes = indexes%self.test_size
            indexes = self.test_idx[indexes]
            self.test_step = self.test_step+size
            files = [self.test_files[j] for j in indexes]
        arrays   = [np.load(files[j]) for j in range(size)]
        vertices = [arrays[j].item().get('vertices') for j in range(size)]
        faces    = [arrays[j].item().get('faces') for j in range(size)]
        faces_up         = [arrays[j].item().get('faces_up') for j in range(size)]
        faces_up_area    = [arrays[j].item().get('faces_up_area') for j in range(size)]
        out_arrays = []
        for ii in range(size):
            example = {}
            example['faces'] = faces[ii]
            example['vertices'] = vertices[ii]
            batch_var = np.sqrt(np.sum(np.power(example['vertices'],2),axis=1))
            batch_var = np.max(batch_var)
            example['vertices'] = example['vertices']/batch_var  
            example['vertices'] = example['vertices'].astype(np.float32)
            
            faces_up_mult= np.tile(faces_up[ii],(self.multipier,1))
            faces_values = example['vertices'][faces_up_mult]
            num_faces    = self.up_samp
            r1   = np.random.uniform(low=0.0, high=1.0,size=(num_faces,1,1))
            r2   = np.random.uniform(low=0.0, high=1.0,size=(num_faces,1,1))
            rand_samps = np.concatenate(((1-np.sqrt(r1)),(np.sqrt(r1)*(1-r2)),(r2*np.sqrt(r1))),axis=1)
            vertices_MC  = np.sum(faces_values*rand_samps,axis=1)
            example['vertices_up'] = vertices_MC
            
            V1 = faces_values[:,1,:] - faces_values[:,0,:]
            V2 = faces_values[:,2,:] - faces_values[:,0,:]
            normal = np.cross(V1,V2)
            example['vertices_up_normals'] = normal/np.linalg.norm(normal,axis=1,keepdims=True)
            

            out_arrays.append(example)
        return out_arrays


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
    
    
    
    