
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


path  = '/media/gidi/SSD/Thesis/Data/Checkpoints/Results/batch6/'
type_ = 'iou_values_test.npy'

iou_test  = []
iou_train = []
acc_test  = []
acc_train = []
num_plots = 64
strings   = []
test_epoch = 26

for ii in range(num_plots):
    file_name = path+'dnn_arch_exp'+str(ii+1)+'/iou_values_test.npy'
    iou_test.append(np.load(file_name))
    file_name = path+'dnn_arch_exp'+str(ii+1)+'/accuracy_values_test.npy'
    acc_test.append(np.load(file_name))
    file_name = path+'dnn_arch_exp'+str(ii+1)+'/accuracy_values.npy'
    acc_train.append(np.load(file_name))
    file_name = path+'dnn_arch_exp'+str(ii+1)+'/iou_values.npy'
    iou_train.append(np.load(file_name))
    strings.append(str(ii))
    



#for ii in range(num_plots):
#    file_name = path+'exp_full_data'+str(ii+1)+'w/iou_values_test.npy'
#    iou_test.append(np.load(file_name))
#    file_name = path+'exp_full_data'+str(ii+1)+'w/accuracy_values_test.npy'
#    acc_test.append(np.load(file_name))
#    file_name = path+'exp_full_data'+str(ii+1)+'w/accuracy_values.npy'
#    acc_train.append(np.load(file_name))
#    file_name = path+'exp_full_data'+str(ii+1)+'w/iou_values.npy'
#    iou_train.append(np.load(file_name))
#    strings.append(str(ii))
  
#for ii in range(num_plots):
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/iou_values_test.npy'
#    iou_test.append(np.load(file_name))
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/accuracy_values_test.npy'
#    acc_test.append(np.load(file_name))
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/accuracy_values.npy'
#    acc_train.append(np.load(file_name))
#    file_name = path+'exp_full_data_aug'+str(ii+1)+'w/iou_values.npy'
#    iou_train.append(np.load(file_name))    
#    strings.append(str(ii))
    

max_ = 0.0
for ii in range(num_plots):
    iou = iou_test[ii]
    if iou.shape[0]>test_epoch:
        value = iou[test_epoch]
        if value>max_:
            max_ = value
            max_ii = ii
   
strings.append('base')
 


    
plt.figure(1)
plt.title('test iou')
for ii in range(num_plots):
    plt.plot(iou_test[ii])
plt.plot(iou_values_test)
plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)
plt.text(test_epoch,max_,'winner=exp'+str(max_ii+1))
    
plt.figure(2)
plt.title('test acc')
for ii in range(num_plots):
    plt.plot(acc_test[ii])    
plt.plot(accuracy_values_test)    
plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)    
    
plt.figure(3)
plt.title('train acc')
for ii in range(num_plots):
    plt.plot(acc_train[ii])  
plt.plot(accuracy_values)    
plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)

plt.figure(4)
plt.title('train iou')
for ii in range(num_plots):
    plt.plot(iou_train[ii])
plt.plot(iou_values)
plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)








