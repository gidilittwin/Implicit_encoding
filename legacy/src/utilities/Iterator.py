from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import time
import scipy.io
from scipy import ndimage






class BinaryIterator:
  
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.epoch = 0
        self.num_images = images.shape[0]
        self.mean = np.mean(images,axis=0)
        self.std = np.std(images,axis=0)
#        self.reset_epoch()
        self.batch_size = batch_size
        
#    def reset_epoch(self):    
#        self.shuffle = np.random.permutation(self.num_images)
#        self.idx = 0


#    def get_batch(self,batch_size):
#        if self.idx==self.num_images:
#           self.reset_epoch() 
#           self.epoch+=1
#        batch = {}
#        start_idx = self.idx
#        end_idx = min(start_idx+batch_size,self.num_images)
#        batch['images'] = self.images[self.shuffle[start_idx:end_idx],:,:,:]
#        batch['labels'] = self.labels[self.shuffle[start_idx:end_idx]]
#        self.idx = end_idx
#        return batch        
        

    def augmentData(self,batch):
        batch_size = batch['images'].shape[0]
        bg_value = 0.
        Canvas = np.zeros((batch_size,36,36,1))
        for i in range(batch_size):
            # rotate the image with random degree
            image = batch['images'][i,:,:,:]
#            angle = np.random.randint(-15,15,1)
#            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            Canvas[:,4:32,4:32,:] = image
            idx1 = np.random.randint(0,6)
            idx2 = np.random.randint(0,6)
            batch['images'][i,:,:,:] = Canvas[i,idx1:(idx1+28),idx2:(idx2+28),:]
        return batch


    def __iter__(self):
        self.shuffle = np.random.permutation(self.num_images)
        self.idx = 0
        self.step = 0
        return self

    def next(self):
        if self.idx!=self.num_images:
            self.step+=1
            batch = {}
            start_idx = self.idx
            end_idx = min(start_idx+self.batch_size,self.num_images)
            batch['images'] = self.images[self.shuffle[start_idx:end_idx],:,:,:]
            batch['labels'] = self.labels[self.shuffle[start_idx:end_idx]]
            batch['step'] = self.step
            self.idx = end_idx
            return batch     
        else:
            raise StopIteration







