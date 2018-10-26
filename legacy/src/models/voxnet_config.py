import numpy as np
import tensorflow as tf



def voxelnet_config(code):
    vox_params = {}
    if code==0:
        vox_params['filter']     = [128,128,256,256,512,1024]
        vox_params['downsample'] = [2,1,2,1,2,2]   
    elif code==1:
        vox_params['filter']     = [256,256,512,512,512,1024]
        vox_params['downsample'] = [2,2,2,2,1,1]
    elif code==2:
        vox_params['filter']     = [256,256,512,512,512,1024]
        vox_params['downsample'] = [1,2,1,2,2,2]
    elif code==3:
        vox_params['filter']     = [256,256,512,512,512,1024]
        vox_params['downsample'] = [2,1,2,1,2,2]
    elif code==4:
        vox_params['filter']     = [128,256,512,512,512,1024]
        vox_params['downsample'] = [2,2,2,1,2,2]
    
    return vox_params