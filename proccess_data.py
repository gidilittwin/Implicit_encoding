
import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
import matplotlib.pyplot as plt
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
from src.utilities import raytrace as RAY
import matplotlib.pyplot as plt
from src.utilities import iou_loss as IOU


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


MODEL_PARAMS = './model_params.json'
with open(MODEL_PARAMS, 'r') as f:
    config = json.load(f)







#%%

rand     = False
rec_mode = True
BATCH_SIZE = 1
SN_train       = ShapeNet(config['path'],
                 files=config['train_file'],
                 rand=rand,
                 batch_size=BATCH_SIZE,
                 grid_size=config['grid_size'],
                 levelset=[0.00],
                 num_samples=config['num_samples'],
                 list_=config['categories'],
                 rec_mode=rec_mode)
for ii in range(0,SN_train.train_size):
    batch = SN_train.get_batch(type_='')
    print(str(SN_train.train_step)+' /'+str(SN_train.train_size))

SN_test        = ShapeNet(config['path'],
                 files=config['test_file'],
                 rand=False,
                 batch_size=BATCH_SIZE,
                 grid_size=config['grid_size'],
                 levelset=[0.00],
                 num_samples=config['num_samples'],
                 list_=config['categories'],
                 rec_mode=False)
for ii in range(0,SN_test.train_size):
    batch = SN_test.get_batch(type_='')
    print(str(SN_test.train_step)+' /'+str(SN_test.train_size))

    







