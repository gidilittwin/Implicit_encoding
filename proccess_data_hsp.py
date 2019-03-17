
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from provider_binvox import ShapeNet as ShapeNet 
import matplotlib.pyplot as plt
from src.utilities import mesh_handler as MESHPLOT
import scipy.ndimage as ndi
from skimage import measure


class MOV_AVG(object):
    def __init__(self, size):
        self.list = []
        self.size = size
    def push(self,sample):
        if np.isnan(sample)!=True:
            self.list.append(sample)
            if len(self.list)>self.size:
                self.list.pop(0)
        return np.mean(np.array(self.list))
    def reset(self):
        self.list    = []
    def get(self):
        return np.mean(np.array(self.list))        


import argparse
import socket


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'exp_full_data')
    parser.add_argument('--model_params_path', type=str, default= './archs/architecture_2_no_bn_wide.json')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--grid_size', type=int,  default=256)
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
        parser.add_argument("--mesh_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetHSP/")
        parser.add_argument("--train_file"      , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/train_list.txt")
        parser.add_argument("--test_file"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/test_list.txt")
        parser.add_argument("--checkpoint_path" , type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/exp31(benchmark=57.4)/-196069")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNetRendering/")
        parser.add_argument("--mesh_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetHSP/")
        parser.add_argument("--train_file"      , type=str, default="/private/home/wolf/gidishape/train_list.txt")
        parser.add_argument("--test_file"       , type=str, default="/private/home/wolf/gidishape/test_list.txt")
        parser.add_argument("--checkpoint_path" , type=str, default="/private/home/wolf/gidishape/checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/private/home/wolf/gidishape/checkpoints/exp31(benchmark=57.4)/-196069")
    return parser.parse_args()
config = parse_args()


#02933112/ac7935b217aeb58e19ca41cdb396f36.mat
#04256520/31ae964a8a9a15e87934a0d24a61231.mat

#%%
rec_mode     = True
BATCH_SIZE   = 1
#SN_train     = ShapeNet(config.iccv_path+'train',config.mesh_path,
#                 files=[],
#                 rand=False,
#                 batch_size=BATCH_SIZE,
#                 grid_size=config.grid_size,
#                 levelset=[0.00],
#                 num_samples=config.num_samples,
#                 list_=config.categories,
#                 rec_mode=rec_mode)
#for ii in range(0,SN_train.train_size):
#    batch = SN_train.preprocess_iccv(type_='')
#    print(str(SN_train.train_step)+' /'+str(SN_train.train_size))



#SN_val     = ShapeNet(config.iccv_path+'val',config.mesh_path,
#                 files=[],
#                 rand=False,
#                 batch_size=BATCH_SIZE,
#                 grid_size=config.grid_size,
#                 levelset=[0.00],
#                 num_samples=config.num_samples,
#                 list_=config.categories,
#                 rec_mode=rec_mode)
#for ii in range(0,SN_val.train_size):
#    batch = SN_val.preprocess_iccv(type_='')
#    print(str(SN_val.train_step)+' /'+str(SN_val.train_size))
    

SN_test     = ShapeNet(config.iccv_path+'test',config.mesh_path,
                 files=[],
                 rand=False,
                 batch_size=BATCH_SIZE,
                 grid_size=config.grid_size,
                 levelset=[0.00],
                 num_samples=config.num_samples,
                 list_=config.categories,
                 rec_mode=rec_mode)
for ii in range(0,SN_test.train_size):
    batch = SN_test.preprocess_iccv(type_='')
    print(str(SN_test.train_step)+' /'+str(SN_test.train_size))
    
    
    
    
#%% Test Iterator
 
#if False:    
#    batch     = SN_train.get_batch(type_='')
#    size_     = SN_train.train_size
#    psudo_sdf = batch['sdf'][0,:,:,:]
#    verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
#    cubed0    = {'vertices':verts0/(config.grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(config.grid_size-1)*2-1}
#    MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')    
#    
#    vertices = batch['vertices'][:,:,:,0]/(config.grid_size-1)*2-1
#    cubed    = {'vertices':vertices[0,:,:],'faces':faces0,'vertices_up':vertices[0,:,:]}
#    MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud_up')  
#    
#    #batch = SN_train.get_batch(type_='')    
#    #pic = batch['images'][0,:,:,:]
#    #fig = plt.figure()
#    #plt.imshow(pic/255.)
#    
#    vox = (batch['sdf']+0.5).astype(np.bool)
#    inner_volume       = vox
#    outer_volume       = np.logical_not(vox)
#    sdf_o = ndi.distance_transform_edt(outer_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#    sdf_i = ndi.distance_transform_edt(inner_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#    sdf_                 = (sdf_o - sdf_i)
#    
#    vertices             = batch['vertices'][:,:,:,0]/(config.grid_size-1)*2-1
#    samples_xyz_np       = vertices
#    samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
#    batch_idx            = np.tile(np.reshape(np.arange(0,config.batch_size,dtype=np.int32),(config.batch_size,1,1)),(1,config.num_samples,1))
#    samples_ijk_np       = np.reshape(np.concatenate((batch_idx,samples_ijk_np),axis=-1),(config.batch_size*(config.num_samples),4))
#    samples_sdf_np       = np.reshape(sdf_[samples_ijk_np[:,0],samples_ijk_np[:,2],samples_ijk_np[:,1],samples_ijk_np[:,3]],(config.batch_size,config.num_samples,1))
#    
#    
#    
#    vox = batch['voxels'][0,:,:,:]
#    vox_slice = vox[:,:,128]
#    fig = plt.figure()
#    plt.imshow(vox_slice)
        
        
        
    
    
    