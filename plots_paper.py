


import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
from src.utilities import raytrace as RAY
import os
import argparse
import socket

 
ious_per_class        = []
iou_per_class        = []
Features_per_class   = []
ids_per_class        = []
Features_per_class_N = []
names                = []
for cc in range(0,13):
    ious_per_class.append(ious[Classes==cc])
    iou_per_class.append(np.mean(ious[Classes==cc]))
    Features_per_class.append(Features[Classes==cc,:])
    ids_per_class.append(ids[Classes==cc])
    perm = np.random.permutation(Features_per_class[-1].shape[0])
    Features_per_class_N.append(Features_per_class[-1][perm[:500],:])
    names.append(classes2name[config.categories[cc]]['name'])
mean_iou = np.mean(iou_per_class)    
print(mean_iou)


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










