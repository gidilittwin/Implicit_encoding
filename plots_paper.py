


import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
#from src.utilities import raytrace as RAY
import os
import argparse
import socket

with open('./classes.json', 'r') as f:
    classes2name = json.load(f)
#for ii,key in enumerate(config.categories):
#    classes2name[key]['id']=ii

 #%% IOU    

ious_per_class        = []
iou_per_class        = []
Features_per_class   = []
ids_per_class        = []
Features_per_class_N = []
names                = []
for cc in range(0,13):
    ious_per_class.append(ious_test[classes_test==cc])
    iou_per_class.append(np.mean(ious_test[classes_test==cc]))
#    Features_per_class.append(Features[Classes==cc,:])
    ids_per_class.append(ids_test[classes_test==cc])
#    perm = np.random.permutation(Features_per_class[-1].shape[0])
#    Features_per_class_N.append(Features_per_class[-1][perm[:500],:])
    names.append(classes2name[config.category_names[cc]]['name'])
mean_iou = np.mean(iou_per_class)    
print(mean_iou)





#%% CD    
classes = []    
cds = []
for ii in range(1300):
    print(ii)        
    feed_dict = {idx_node           :0,
             level_set          :0}   
  
    evals_target_, evals_function_, next_element_display_ = session.run([evals_target, evals_function, next_element_display],feed_dict=feed_dict) 

 
    
    field              = np.reshape(evals_function_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts1, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)

    
    field              = np.reshape(evals_target_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts2, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.)

        
    verts1_cd_tf          = tf.placeholder(tf.float32,shape=(1,None,3), name='cloud1')  
    verts2_cd_tf          = tf.placeholder(tf.float32,shape=(1,None,3), name='cloud2') 
    dist1, idx1, dist2, idx2 = CD.nn_distance_cpu(verts1_cd_tf,verts2_cd_tf)
    
    p1 = np.random.permutation(verts1.shape[0])    
    verts1_cd = verts1[p1,:]  /256.
    if verts1_cd.shape[0]>10000:
        verts1_cd = verts1_cd[0:10000,:]
    p2 = np.random.permutation(verts2.shape[0])    
    verts2_cd = verts2[p2,:]  /256.
    if verts2_cd.shape[0]>10000:
        verts2_cd = verts2_cd[0:10000,:]        
    feed_dict_cd = {verts1_cd_tf           :np.expand_dims(verts1_cd,0),
                    verts2_cd_tf           :np.expand_dims(verts2_cd,0)}   
    dist1_, idx1_, dist2_, idx2_  = session.run([dist1, idx1, dist2, idx2 ],feed_dict=feed_dict_cd) 
    
    cd = (np.mean(dist1_) + np.mean(dist2_))*1000
    cds.append(cd)
    classes.append(next_element_display_['classes'])

    cds_ = np.stack(cds)
    classes_ = np.stack(classes)[:,0,0]


    cd_per_class        = []
    cds_per_class        = []
    ids_per_class        = []
    for cc in range(0,13):
        cds_per_class.append(cds_[classes_==cc])
        cd_per_class.append(np.mean(cds_[classes_==cc]))

    mean_cd = np.mean(cd_per_class)    
    print(mean_cd)


    

















#%% TSNE    
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.manifold import TSNE
import matplotlib as mpl
 mpl.style.use('seaborn')
 
TSNE_Features   = np.concatenate(Features_per_class_N,0)
TSNE_Classes,_  = np.meshgrid(range(0,13),range(0,500))
TSNE_Classes    = np.reshape(np.transpose(TSNE_Classes),-1)

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(TSNE_Features)

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
plt.figure(figsize=(7, 5))
for i in range(0,13):
    plt.scatter(X_2d[TSNE_Classes == i, 0], X_2d[TSNE_Classes == i, 1], c=colors[i], alpha=0.5, label=names[i])
plt.legend( bbox_to_anchor=(1,1))
plt.subplots_adjust(right=0.5)
plt.show()










