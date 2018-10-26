from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys


import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt




def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value
    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, num,rad,rot):
    phi = np.arange(0,2*np.pi,2*np.pi/num)+rot/2./np.pi
    rho = rad*np.ones((phi.shape))  
    vertices = np.flip((np.transpose(np.array(pol2cart(rho,phi)),(1,0))+shape[0]/2),axis=1)
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros
    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill
    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)
    # Set all values inside polygon to one
    base_array[fill] = 1
    return base_array

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    



def get_batch(N,s):
    labels = np.zeros((N,s,s,1),dtype = 'float32')
    data = np.zeros((N,3),dtype = 'float32')
    for i in range(N):
        num = np.random.uniform(3,6)
        rad = np.random.uniform(15,30)
        rot = np.random.uniform(-10,10)
        data[i,:] = ((num-4.5)/1.5,(rad-22.5)/7.5,rot/10)
        labels[i,:,:,0] = create_polygon([s,s], round(num),rad,rot)
    return {'parameters':data,'images':labels}





    
