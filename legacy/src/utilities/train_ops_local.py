from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import time
import scipy.io
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

skeleton = np.array([[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]])



def plot_voxel(voxels,color,idx):

    voxel = np.copy(voxels[idx,:,:,:,:])
    size_ = voxel.shape[0]+1
    r, g, b = np.indices((size_, size_, size_)) / (size_-1)
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)
    
    V = np.sum(voxel,-1)
    V = V>0
    
    # define a sphere about [0.5, 0.5, 0.5]
#    sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2
    
    # combine the color components
    colors = np.zeros(V.shape + (3,))
    colors[..., 0] = rc
    colors[..., 1] = gc
    colors[..., 2] = bc
    
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(r, g, b, V,
              facecolors=colors,
              edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
              linewidth=0.5)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    
    plt.show()
    plt.pause(0.05)







def plot_xyz(I1,I2,idx):

    I1 = (I1[idx,:,:]).astype(np.int64)
    I2 = (I2[idx,:,:]).astype(np.int64)
    fig = plt.figure(1)
    plt.imshow(I1,origin='lower')
    plt.pause(0.05)
    fig = plt.figure(2)
    plt.imshow(I2,origin='lower')
    plt.pause(0.05)
   
   



def plot_cloud(inputs_nodes_,color,idx,centers=None):
    cloud = inputs_nodes_[idx,:,:]
    # Figure
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
#    ax.clear()
    ax.set(xlabel='x', ylabel='y', zlabel='z')

    # CLOUD
    ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],c='black',edgecolor=color,s=33)  
    ax.scatter(0,0,0,c='red',edgecolor='blue',s=33)
#    if np.isnan(centers) != True:
#    ax.scatter(centers[idx,0,0,:],centers[idx,0,1,:],centers[idx,0,2,:],c='red',edgecolor='black',s=33)  

    plt.pause(0.05)

