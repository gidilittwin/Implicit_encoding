
import os
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
import matplotlib.pyplot as plt
BATCH_SIZE = 1
import src.utilities.binvox_rw as binvox_rw
from provider_binvox import ShapeNet as ShapeNet 


grid_size   = 32
batch_size  = 16
canvas_size = grid_size
num_points  = 100000


path = '/Users/gidilittwin/Dropbox/Thesis/ModelNet/Data/ShapeNetVox32/'
SN = ShapeNet(path,rand=False,batch_size=16,grid_size=32,levelset=0.0)

batch = SN.get_batch(type_='train')






files = os.listdir('/Users/gidilittwin/Dropbox/Thesis/ModelNet/Data/ShapeNetVox32/')
with open('/Users/gidilittwin/Dropbox/Thesis/ModelNet/Data/ShapeNetVox32/02691156/1b7ac690067010e26b7bd17e458d0dcb/model.binvox', 'rb') as f:
    m1 = binvox_rw.read_as_3d_array(f)
   
    
    
    