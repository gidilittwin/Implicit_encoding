
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
        parser.add_argument("--path_tf"         , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNet_TF256_v3/")
        parser.add_argument("--mesh_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetHSP/")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNetRendering/")
        parser.add_argument("--path_tf"         , type=str, default="/private/home/wolf/gidishape/data/ShapeNet_TF256_v3/")
        parser.add_argument("--mesh_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetHSP/")
    return parser.parse_args()
config = parse_args()









filepath     = '/media/gidi/SSD/Thesis/Data/Pascal3D/PASCAL3D.mat'
 
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
    
batches = {}
batches['images'] = np.transpose(arrays['image_train'],(3,1,2,0))
batches['classes'] = np.expand_dims(np.transpose(arrays['model_train'][0,:]),-1)
batches['voxels'] = np.flip(np.transpose(np.reshape(arrays['model_train'][1:,:],(30,30,30,3734)),(3,1,2,0)),2)

for ii in range(0,batches['voxels'].shape[0]):

    batch = {}
    batch['images'] = np.tile(batches['images'][ii,:,:,:],(20,1,1,1))
    batch['classes'] = np.tile(batches['classes'][ii,:],(20,1,1,1))


    verts, faces, normals, values = measure.marching_cubes_lewiner(batches['voxels'][0,:,:,:],0.5)
    verts         = verts[:,(0,2,1)]
    arr_          = np.arange(0,verts.shape[0])
    perms         = np.random.choice(arr_,10000)
    verts_sampled = verts[perms,:]   



    print(str(ii)+' /'+str(batches['voxels'].shape[0]))
    path =config.path_tf+'test/'
    TFH.dataset_builder_fn(path,batch,compress=True)  



 

import matplotlib.pyplot as plt   

idx =45
psudo_sdf = batches['voxels'][idx,:,:,:]*1.0
verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
cubed0 = {'vertices':verts0/(config.grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(config.grid_size-1)*2-1}
MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')    

vertices             = verts_sampled[:,:,:]/(config.grid_size_v-1)*2-1
cubed = {'vertices':vertices[idx,:,:],'faces':faces0,'vertices_up':vertices[idx,:,:]}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  

vertices             = batch_['samples_xyz'][:,:,:]
cubed = {'vertices':vertices[idx,:,:],'faces':faces0,'vertices_up':vertices[idx,:,:]}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  


vertices             = batch_['samples_xyz'][idx,:,:]
vertices_on          = batch_['samples_sdf'][idx,:,:]<0.
vertices              = vertices*vertices_on
cubed = {'vertices':vertices,'faces':faces0,'vertices_up':vertices}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  



pic = batch_['images'][idx,:,:,0:3]
fig = plt.figure()
plt.imshow(pic)
    