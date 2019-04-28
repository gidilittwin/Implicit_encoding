
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


path     = '/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records23/'
baseline = 'study_dnn32_arch33'
accuracy_values     =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records20/'+baseline+'/accuracy_values.npy')
accuracy_values_test=np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records20/'+baseline+'/accuracy_values_test.npy')
iou_values          =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records20/'+baseline+'/iou_values.npy')
iou_values_test     =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records20/'+baseline+'/iou_values_test.npy')
loss_values         =np.load('/media/gidi/SSD/Thesis/Data/Checkpoints/Results/records20/'+baseline+'/loss_values.npy')

iou_test  = []
iou_train = []
acc_test  = []
acc_train = []
num_plots = 200
strings   = []


name = 'archsweep_exp'
name = 'study_dnn256_v2_arch'
#name = 'study_dnn32_dropout'
#name = 'study_dnn256_cat'
name = 'study_dnn32_stage_v3_'

plot_idx =15



        
for ii in range(num_plots):
    try:
        file_name = path+name+str(ii+1)+'/accuracy_values.npy'
        acc_train.append(np.load(file_name))
        file_name = path+name+str(ii+1)+'/iou_values.npy'
        iou_train.append(np.load(file_name))
        strings.append(str(ii+1))        
        
        file_name = path+name+str(ii+1)+'/iou_values_test.npy'
        iou_test.append(np.load(file_name))
        file_name = path+name+str(ii+1)+'/accuracy_values_test.npy'
        acc_test.append(np.load(file_name))
    except:
        print('missing - ' + str(ii))



num_plots_alive = len(iou_test)
kill_list = []
#for ii in range(num_plots_alive):


plt.figure(4)
plt.title('train iou')
plt.plot(iou_values,'--')
for ii in range(len(iou_train)):
    plt.plot(iou_train[ii])
    plt.text(len(iou_train[ii])-1,iou_train[ii][-1],strings[ii])





plt.figure(ii)
plt.title('test iou')
plt.plot(iou_values_test,'--')
for ii in range(num_plots_alive):
    if np.max(iou_test[ii])>0.0:
        plt.plot(iou_test[ii])
        plt.text(len(iou_test[ii])-1,iou_test[ii][-1],strings[ii])
    if np.max(iou_test[ii])<0.20:
        kill_list.append(strings[ii])
#plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)

    
#plt.figure(2)
#plt.title('test acc')
#for ii in range(num_plots_alive):
#    plt.plot(acc_test[ii])    
#plt.plot(accuracy_values_test)    
#plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)    
#    
#plt.figure(3)
#plt.title('train acc')
#dead = []
#for ii in range(num_plots_alive):
#    plt.plot(acc_train[ii])  
#    if np.max(acc_train[ii])<0.75:
#        print(strings[ii])
#        dead.append(strings[ii])
#        
##plt.plot(accuracy_values)    
#plt.legend(strings,shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=5)
#






