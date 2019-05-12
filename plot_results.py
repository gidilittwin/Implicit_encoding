
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


path     = '/media/gidi/SSD/Thesis/Data/Checkpoints/Results/alpha9/'
baseline = 'study_dnn32_arch33'
accuracy_values     =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'+baseline+'/accuracy_values.npy')
accuracy_values_test=np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'+baseline+'/accuracy_values_test.npy')
iou_values          =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'+baseline+'/iou_values.npy')
iou_values_test     =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'+baseline+'/iou_values_test.npy')
loss_values         =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'+baseline+'/loss_values.npy')

iou_test  = []
iou_train = []
acc_test  = []
acc_train = []
num_plots = 200
strings   = []
strings_test   = []

postfix = ''


name = 'center_loss_exp'
name = 'multi_image_encode_exp'
#name = '256_pre_exp'
#name = 'fastrecords_256_v2_exp'
name = 'light_128_v2_'
name = 'light_256_v3_'
#name = 'study_dnn32_stage_v3_'

plot_idx =15


#postfix = '_stage3-256'
postfix = '1_256'
#postfix = '0_36'

for ii in range(num_plots):
    try:
        file_name = path+name+str(ii+1)+'/accuracy_values'+postfix+'.npy'
        acc_train.append(np.load(file_name))
        file_name = path+name+str(ii+1)+'/iou_values'+postfix+'.npy'
        iou_train.append(np.load(file_name))
        strings.append(str(ii+1))        
        
        file_name = path+name+str(ii+1)+'/iou_values_test'+postfix+'.npy'
        iou_test.append(np.load(file_name))
        file_name = path+name+str(ii+1)+'/accuracy_values_test'+postfix+'.npy'
        acc_test.append(np.load(file_name))
        strings_test.append(str(ii+1))        
    except:
        print('missing - ' + str(ii))



num_plots_alive = len(iou_test)
kill_list = []
#for ii in range(num_plots_alive):


plt.figure()
plt.title('train iou')
plt.plot(iou_values,'--')
for ii in range(len(iou_train)):
    plt.plot(iou_train[ii])
    plt.text(len(iou_train[ii])-1,iou_train[ii][-1],strings[ii])





plt.figure()
plt.title('test iou')
plt.plot(iou_values_test,'--')
for ii in range(num_plots_alive):
    if np.max(iou_test[ii])>0.0:
        plt.plot(iou_test[ii])
        plt.text(len(iou_test[ii])-1,iou_test[ii][-1],strings_test[ii])
    if np.max(iou_test[ii])<0.20:
        kill_list.append(strings_test[ii])


if True==False:
    
    ii=31
    plt.figure()
    plt.title('test iou')
    plt.plot(iou_values_test,'--')
    if np.max(iou_test[ii])>0.0:
        plt.plot(iou_test[ii])
        plt.text(len(iou_test[ii])-1,iou_test[ii][-1],strings_test[ii])

        
        
        
        
        
        
        