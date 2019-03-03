
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


path  = '/media/gidi/SSD/Thesis/Data/Checkpoints/Results/batch5/'
type_ = 'iou_values_test.npy'

iou_test  = []
iou_train = []
acc_test  = []
acc_train = []
num_plots = 3

for ii in range(num_plots):
    file_name = path+'exp_full_data_noflip'+str(ii+1)+'w/iou_values_test.npy'
    iou_test.append(np.load(file_name))

    file_name = path+'exp_full_data_noflip'+str(ii+1)+'w/accuracy_values_test.npy'
    acc_test.append(np.load(file_name))
    
    file_name = path+'exp_full_data_noflip'+str(ii+1)+'w/accuracy_values.npy'
    acc_train.append(np.load(file_name))
    
    file_name = path+'exp_full_data_noflip'+str(ii+1)+'w/iou_values.npy'
    iou_train.append(np.load(file_name))



#for ii in range(num_plots):
#    file_name = path+'exp_full_data'+str(ii+1)+'w/iou_values_test.npy'
#    iou_test.append(np.load(file_name))
#
#    file_name = path+'exp_full_data'+str(ii+1)+'w/accuracy_values_test.npy'
#    acc_test.append(np.load(file_name))
#    
#    file_name = path+'exp_full_data'+str(ii+1)+'w/accuracy_values.npy'
#    acc_train.append(np.load(file_name))
#    
#    file_name = path+'exp_full_data'+str(ii+1)+'w/iou_values.npy'
#    iou_train.append(np.load(file_name))
    
  
#for ii in range(num_plots):
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/iou_values_test.npy'
#    iou_test.append(np.load(file_name))
#
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/accuracy_values_test.npy'
#    acc_test.append(np.load(file_name))
#    
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/accuracy_values.npy'
#    acc_train.append(np.load(file_name))
#    
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/iou_values.npy'
#    iou_train.append(np.load(file_name))    
#    
plt.figure(1)
plt.title('test iou')
for ii in range(num_plots):
    plt.plot(iou_test[ii])
plt.plot(iou_values_test)
plt.legend(( '1', '2', '3','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)
    
plt.figure(2)
plt.title('test acc')
for ii in range(num_plots):
    plt.plot(acc_test[ii])    
plt.plot(accuracy_values_test)    
plt.legend(( '1', '2', '3','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)    
    
plt.figure(3)
plt.title('train acc')
for ii in range(num_plots):
    plt.plot(acc_train[ii])  
plt.plot(accuracy_values)    
plt.legend(( '1', '2', '3','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)

plt.figure(4)
plt.title('train iou')
for ii in range(num_plots):
    plt.plot(iou_train[ii])
plt.plot(iou_values)
plt.legend(( '1', '2', '3','base'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=8)








