
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from provider_binvox import ShapeNet as ShapeNet 
import matplotlib.pyplot as plt
from src.utilities import mesh_handler as MESHPLOT
import scipy.ndimage as ndi
from skimage import measure
import tfrecords_handler as TFH
import argparse
import socket
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'exp_full_data')
    parser.add_argument('--model_params_path', type=str, default= './archs/architecture_2_no_bn_wide.json')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--grid_size', type=int,  default=30)
    parser.add_argument('--batch_size', type=int,  default=1)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--global_points', type=int,  default=10000)    
    parser.add_argument('--checkpoint_every', type=int,  default=10000)
    parser.add_argument('--categories', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
#    parser.add_argument('--categories', type=int,  default=["02691156"], help='number of point samples')
    parser.add_argument('--plot_every', type=int,  default=1000)
    parser.add_argument('--test_every', type=int,  default=10000)
    parser.add_argument('--learning_rate', type=float,  default=0.00005)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=False)
    if socket.gethostname() == 'gidi-To-be-filled-by-O-E-M':
        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/")
        parser.add_argument("--path_tf"         , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNet_TF30/")
        parser.add_argument("--mesh_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetHSP/")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNetRendering/")
        parser.add_argument("--path_tf"         , type=str, default="/private/home/wolf/gidishape/data/ShapeNet_TF30/")
        parser.add_argument("--mesh_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetHSP/")
    return parser.parse_args()
config = parse_args()




#%%





filepath     = '/media/gidi/SSD/Thesis/Data/Pascal3D/PASCAL3D.mat'
 
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
    
batches = {}
batches['images'] = np.transpose(arrays['image_train'],(3,2,1,0)).astype(np.float32)
batches['classes'] = np.expand_dims(np.transpose(arrays['model_train'][0,:]),-1).astype(np.int64)
batches['voxels'] = np.flip(np.transpose(np.reshape(arrays['model_train'][1:,:],(30,30,30,3734)),(3,1,2,0)),2).astype(np.bool)
for ii in range(0,batches['voxels'].shape[0]):
    batch = {}
    batch['images']  = 255*np.concatenate((np.tile(batches['images'][ii,:,:,:],(20,1,1,1)),np.ones((20,100,100,1),dtype=np.float32)),axis=-1)
    batch['classes'] = batches['classes'][ii:ii+1,:]
    batch['voxels']  = np.transpose(batches['voxels'][ii:ii+1,:,:,:],(0,2,1,3))
    batch['ids']  = np.array([[ii]])
    verts, faces, normals, values = measure.marching_cubes_lewiner(batch['voxels'][0,:,:,:],0.5)
    verts         = verts[:,(1,0,2)]
    arr_          = np.arange(0,verts.shape[0])
    perms         = np.random.choice(arr_,10000)
    verts_sampled = verts[perms,:]   
    batch['vertices'] = np.expand_dims(np.expand_dims(verts_sampled,0),-1)
    print(str(ii)+' /'+str(batches['voxels'].shape[0]))
    path =config.path_tf+'train/'
    TFH.dataset_builder_fn(path,batch,compress=True)  


batches = {}
batches['images'] = np.transpose(arrays['image_test'],(3,2,1,0)).astype(np.float32)
batches['classes'] = np.expand_dims(np.transpose(arrays['model_test'][0,:]),-1).astype(np.int64)
batches['voxels'] = np.flip(np.transpose(np.reshape(arrays['model_test'][1:,:],(30,30,30,arrays['model_test'].shape[-1])),(3,1,2,0)),2).astype(np.bool)
for ii in range(0,batches['voxels'].shape[0]):
    batch = {}
    batch['images']  = 255*np.concatenate((np.tile(batches['images'][ii,:,:,:],(20,1,1,1)),np.ones((20,100,100,1),dtype=np.float32)),axis=-1)
    batch['classes'] = batches['classes'][ii:ii+1,:]
    batch['voxels']  = np.transpose(batches['voxels'][ii:ii+1,:,:,:],(0,2,1,3))
    batch['ids']  = np.array([[ii]])
    verts, faces, normals, values = measure.marching_cubes_lewiner(batch['voxels'][0,:,:,:],0.5)
    verts         = verts[:,(1,0,2)]
    arr_          = np.arange(0,verts.shape[0])
    perms         = np.random.choice(arr_,10000)
    verts_sampled = verts[perms,:]   
    batch['vertices'] = np.expand_dims(np.expand_dims(verts_sampled,0),-1)
    print(str(ii)+' /'+str(batches['voxels'].shape[0]))
    path =config.path_tf+'test/'
    TFH.dataset_builder_fn(path,batch,compress=True)  




 
 

