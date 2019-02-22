
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


path  = '/media/gidi/SSD/Thesis/Data/Checkpoints/Results/bunch1/'
type_ = 'iou_values_test.npy'

iou_test  = []
iou_train = []
acc_test  = []
acc_train = []

for ii in range(6):
    file_name = path+'architecture_'+str(ii+1)+'iou_values_test.npy'
    iou_test.append(np.load(file_name))

    file_name = path+'architecture_'+str(ii+1)+'accuracy_values_test.npy'
    acc_test.append(np.load(file_name))
    
    file_name = path+'architecture_'+str(ii+1)+'accuracy_values.npy'
    acc_train.append(np.load(file_name))
    
    file_name = path+'architecture_'+str(ii+1)+'iou_values.npy'
    iou_train.append(np.load(file_name))


    
    
    
plt.figure(1)
plt.title('test iou')
for ii in range(6):
    plt.plot(iou_test[ii])
plt.plot(iou_values_test)
plt.legend(( '1', '2', '3','4', '5', '6','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)
    
plt.figure(2)
plt.title('test acc')
for ii in range(6):
    plt.plot(acc_test[ii])    
plt.legend(( '1', '2', '3','4', '5', '6'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)    
    
plt.figure(3)
plt.title('train acc')
for ii in range(6):
    plt.plot(acc_train[ii])    
plt.legend(( '1', '2', '3','4', '5', '6'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)

plt.figure(4)
plt.title('train iou')
for ii in range(6):
    plt.plot(iou_train[ii])
plt.plot(iou_values)
plt.legend(( '1', '2', '3','4', '5', '6','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)








