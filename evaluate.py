
import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
import os
import argparse
import tfrecords_handler as TFH
import socket
from skimage import measure
from src.utilities.nn_distance import tf_nndistance_cpu as CD

# tar -czvf meta_functionals.tar.gz metafunctionals/ --exclude-vcs


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'light_128_v3_3')
    parser.add_argument('--model_params_path', type=str, default= './archs/resnet_5_light2.json')
    parser.add_argument('--padding', type=str, default= 'VALID')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--batch_size', type=int,  default=1)
    parser.add_argument('--beta1', type=float,  default=0.9)
    parser.add_argument('--dropout', type=float,  default=1.0)
    parser.add_argument('--stage', type=int,  default=1)
    parser.add_argument('--multi_image', type=int,  default=0)
    parser.add_argument('--multi_image_views', type=int,  default=4)
    parser.add_argument('--alpha', type=float,  default=0.003)
    parser.add_argument('--grid_size', type=int,  default=256)
    parser.add_argument('--grid_size_v', type=int,  default=256)
    parser.add_argument('--compression', type=int,  default=1)
    parser.add_argument('--pretrained', type=int,  default=0)
    
    parser.add_argument('--img_size', type=int,  default=[224,224])
    parser.add_argument('--im_per_obj', type=int,  default=20)
    parser.add_argument('--test_size', type=int,  default=20)
    parser.add_argument('--shuffle_size', type=int,  default=100)  
    parser.add_argument('--test_every', type=int,  default=20000)    
    parser.add_argument('--save_every', type=int,  default=10000) 
    parser.add_argument("--postfix"   , type=str, default="")
    parser.add_argument('--fast_eval', type=int,  default=0)    

    parser.add_argument('--eval_grid_scale', type=int,  default=1)
    parser.add_argument('--batch_norm', type=int,  default=0)
    parser.add_argument('--bn_l0', type=int,  default=0)
    parser.add_argument('--augment', type=int,  default=1)
    parser.add_argument('--rgba', type=int,  default=1)
    parser.add_argument('--symetric', type=int,  default=1)
    parser.add_argument('--radius', type=float,  default=0.1)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--global_points', type=int,  default=10000)    
    parser.add_argument('--noise_scale', type=float,  default=0.05)
#    parser.add_argument('--categories'      , type=str,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
    parser.add_argument('--categories'      , type=int,  default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='number of point samples')
#    parser.add_argument('--categories'      , type=int,  default=[4], help='number of point samples')

    parser.add_argument('--category_names', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
    parser.add_argument('--learning_rate', type=float,  default=0.0001)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=True)
    parser.add_argument('--plot_every', type=int,  default=1000)
    if socket.gethostname() == 'gidi-To-be-filled-by-O-E-M':
        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNet_TF")
        parser.add_argument("--checkpoint_path" , type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--pretrained_path",  type=str, default="/media/gidi/SSD/Thesis/Data/pretrained/")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNet_TF")
        parser.add_argument("--checkpoint_path" , type=str, default="/private/home/wolf/gidishape/checkpoints2/")
        parser.add_argument("--saved_model_path", type=str, default="/private/home/wolf/gidishape/checkpoints2/")   
        parser.add_argument("--pretrained_path",  type=str, default="/private/home/wolf/gidishape/pretrained/")

    return parser.parse_args()
config = parse_args()


config.grid_size_v = 256
config.img_size    = [224,224]
config.im_per_obj  = 20
config.test_size   = 1
config.shuffle_size= 100
config.test_every  = 10000
config.save_every  = 10000
config.compression = 1
#config.postfix     = str(config.stage)+'_'+str(config.grid_size)
config.postfix     = str(config.stage)+'_128'
#config.postfix     = '_stage2-32'

config.fast_eval   = 0
if config.grid_size==256:
    config.path        = config.path+str(config.grid_size)+'_v3/'
else:
    config.path        = config.path+str(config.grid_size)+'/'

if isinstance(config.categories, int):
    config.categories = [config.categories]

class MOV_AVG(object):
    def __init__(self, size):
        self.list = []
        self.size = size
    def push(self,sample):
        if np.isnan(sample)!=True:
            self.list.append(sample)
            if len(self.list)>self.size:
                self.list.pop(0)
        return np.mean(np.array(self.list))
    def reset(self):
        self.list    = []
    def get(self):
        return np.mean(np.array(self.list))        


MODEL_PARAMS = config.model_params_path
MODE_TRAIN = 0
MODE_TEST  = 1
with open(MODEL_PARAMS, 'r') as f:
    model_params = json.load(f)
config.model_params = model_params    
    
    
directory = config.checkpoint_path + config.experiment_name 
if not os.path.exists(directory):
    os.makedirs(directory)

with open('./classes.json', 'r') as f:
    classes2name = json.load(f)
#for ii,key in enumerate(config.categories):
#    classes2name[key]['id']=ii



print('#############################################################################################')
print('###############################  '+config.experiment_name+'   ################################################')
print('#############################################################################################')
print('levelset= ',str(config.levelset))

    

#%% Data iterators

test_iterator  = TFH.iterator(config.path+'test/',
                              config.test_size,
                              epochs=10000,
                              shuffle=False,
                              img_size=config.img_size[0],
                              im_per_obj=config.im_per_obj,
                              grid_size=config.grid_size,
                              num_samples=config.num_samples,
                              shuffle_size=config.shuffle_size,
                              categories = config.categories,
                              compression = config.compression)
    

idx_node          = tf.placeholder(tf.int32,shape=(), name='idx_node')  
level_set         = tf.placeholder(tf.float32,shape=(),   name='levelset')  
next_element_test = test_iterator.get_next()
next_batch_test = TFH.process_batch_test(next_element_test,idx_node,config)


grid_size_lr = config.grid_size
x            = np.linspace(-1, 1, grid_size_lr)
y            = np.linspace(-1, 1, grid_size_lr)
z            = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)




import matplotlib.pyplot as plt   
session = tf.Session()
session.run(tf.initialize_all_variables())
#session.run(mode_node.assign(False)) 
session.run(test_iterator.initializer)
batch,batch_ = session.run([next_element_test,next_batch_test],feed_dict={idx_node:0})
batch,batch_ = session.run([next_element_test,next_batch_test],feed_dict={idx_node:0})

idx =0
psudo_sdf = batch['voxels'][idx,:,:,:]*1.0
verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
cubed0 = {'vertices':verts0/(config.grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(config.grid_size-1)*2-1}
MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')    

vertices             = batch['vertices'][:,:,:]/(config.grid_size_v-1)*2-1
cubed = {'vertices':vertices[idx,:,:],'faces':faces0,'vertices_up':vertices[idx,:,:]}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  

vertices             = batch_['samples_xyz'][:,:,:]
cubed = {'vertices':vertices[idx,:,:],'faces':faces0,'vertices_up':vertices[idx,:,:]}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  


vertices             = batch_['samples_xyz'][idx,:,:]
vertices_on          = batch_['samples_sdf'][idx,:,:]<0.
vertices              = vertices*vertices_on
cubed = {'vertices':vertices,'faces':faces0,'vertices_up':vertices}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  



pic = batch_['images'][idx,:,:,0:3]
fig = plt.figure()
plt.imshow(pic)



#aa=batch_['samples_sdf'][0,1000:,:]
#samps=batch_['samples_xyz'][0,1000:,:]
#samples_ijk_np       = np.round(((samps+1)/2*(config.grid_size-1))).astype(np.int64)
#arr_                = np.split(samples_ijk_np,3,axis=-1)
#
#voxels = batch['voxels'][0,:,:,:]
#import scipy.ndimage as ndi
#inner_volume       = voxels
#outer_volume       = np.logical_not(voxels)
#sdf_o = ndi.distance_transform_edt(outer_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#sdf_i = ndi.distance_transform_edt(inner_volume, return_indices=False) #- ndi.distance_transform_edt(inner_volume)
#sdf_                 = (sdf_o - sdf_i)/(config.grid_size-1)*2  
#voxels_gathered      = sdf_[arr_[1],arr_[0],arr_[2]]
#samples_sdf_np       = -1.*voxels_gathered + 0.5








#%% Function wrappers     
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')

def g_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_shape(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function

def f_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)
    
def m_wrapper(ids,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.multiplexer(ids,args_)
        return CNN.regressor(current,args_)    

def f2_wrapper(image,args_):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_50(image,args_)
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_) 
    
    
def injection_wrapper(current,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_)
    


#%% Training graph 
def build_graph(next_batch,config,batch_size):
    images                = next_batch['images'] 
    samples_sdf           = next_batch['samples_sdf']  
    samples_xyz           = next_batch['samples_xyz']
    evals_target          = {}
    evals_target['x']     = samples_xyz
    evals_target['y']     = samples_sdf
    evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
    if config.pretrained:
        g_weights             = f2_wrapper(images,[mode_node,config])
    else:
        g_weights             = f_wrapper(images,[mode_node,config])
#    g_weights             = m_wrapper(next_batch['ids'] ,[mode_node,config])
    evals_function        = SF.sample_points_list(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)
#    evals_function        = SF.render_sil(evals_function,evals_target,config)
    
    labels                = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(batch_size,-1)),0.0),tf.int64)
    logits                = tf.reshape(evals_function['y'],(batch_size,-1,1)) #- levelset
    logits_iou            = tf.concat((logits-level_set,-logits+level_set),axis=-1)
    logits_ce             = tf.concat((logits,-logits),axis=-1)
    predictions           = tf.nn.softmax(logits_iou)
    correct_prediction    = tf.equal(tf.argmax(predictions, 2), labels)
    accuracy              = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    err                   = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss_class            = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_ce,name='cross-entropy'),axis=-1)
    loss                  = tf.reduce_mean(loss_class)
    if config.multi_image:
       center_loss = 0.5*tf.reduce_mean((tf.get_collection('embeddings')[0] - tf.get_collection('centers')[0]) **2)
       loss = loss + config.alpha*center_loss
    X                     = tf.cast(labels,tf.bool)
    Y                     = tf.cast(tf.argmax(predictions, 2),tf.bool)
    iou_image             = tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1)
    iou                   = tf.reduce_mean(iou_image)
    X_32 = tf.reshape(X,(1,config.grid_size,config.grid_size,config.grid_size,1))
    Y_32 = tf.reshape(Y,(1,config.grid_size,config.grid_size,config.grid_size,1))
    X_32 = tf.nn.max_pool3d(tf.cast(X_32,tf.float32),ksize=(1,8,8,8,1),strides=(1,8,8,8,1),padding='VALID')
    Y_32 = tf.nn.max_pool3d(tf.cast(Y_32,tf.float32),ksize=(1,8,8,8,1),strides=(1,8,8,8,1),padding='VALID')
    X_32 = tf.reshape(tf.greater(X_32,0.5),(1,-1))
    Y_32 = tf.reshape(tf.greater(Y_32,0.5),(1,-1))
    iou_32_image             = tf.reduce_sum(tf.cast(tf.logical_and(X_32,Y_32),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X_32,Y_32),tf.float32),axis=1)
    iou_32                   = tf.reduce_mean(iou_32_image)
    
    return {'loss':loss,'accuracy':accuracy,'err':err,'iou':iou,'iou_image':iou_image,'X':X,'Y':Y,'iou_32_image':iou_32_image,'iou_32':iou_32}

test_dict  = build_graph(next_batch_test,config,batch_size=config.test_size)
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver    = tf.train.Saver(var_list=all_vars)
loader   = tf.train.Saver(var_list=all_vars)
if config.pretrained:
    pretrained_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'resnet_v2_50')
    pretrained        = tf.train.Saver(var_list=pretrained_vars)



#%% Train loop
def ma_iou(ious_test,classes_test,config):
    ious_per_class       = []
    iou_per_class        = []
    names                = []
    for cc in range(0,13):
        ious_per_class.append(ious_test[classes_test==cc])
        iou_per_class.append(np.mean(ious_per_class))
        names.append(classes2name[config.categories[cc]]['name'])
    mean_iou = np.mean(iou_per_class)    
    print('Mean average IOU score is: ' + str(mean_iou) )   
    return mean_iou






    
session = tf.Session()
session.run(tf.initialize_all_variables())
loss_plot      = []
acc_plot       = []
iou_plot       = []
acc_plot_test  = []
iou_plot_test  = []
max_test_acc   = 0.
max_test_iou   = 0.

#loader.restore(session, directory+'/latest'+config.postfix+'-0')
loader.restore(session, directory+'/latest_stage2-32-0')
#loss_plot     = np.load(directory+'/loss_values'+config.postfix+'.npy')
#acc_plot      = np.load(directory+'/accuracy_values'+config.postfix+'.npy')  
#iou_plot      = np.load(directory+'/iou_values'+config.postfix+'.npy')      
#acc_plot_test = np.load(directory+'/accuracy_values_test'+config.postfix+'.npy') 
#iou_plot_test = np.load(directory+'/iou_values_test'+config.postfix+'.npy') 
#loss_plot     = np.split(loss_plot,loss_plot.shape[0])
#acc_plot      = np.split(acc_plot,acc_plot.shape[0])
#iou_plot      = np.split(iou_plot,iou_plot.shape[0])
#acc_plot_test = np.split(acc_plot_test,acc_plot_test.shape[0])
#iou_plot_test = np.split(iou_plot_test,iou_plot_test.shape[0])    
step           = 0


session.run(mode_node.assign(False)) 
acc_mov_test  = MOV_AVG(3000000) # exact mean
iou_mov_test  = MOV_AVG(3000000) # exact mean
iou_mov_test_32  = MOV_AVG(3000000) # exact mean

classes       = []
ids           = []  
ious          = []
ious32        = []
num_epochs = config.im_per_obj/config.test_size
for epoch_test in range(num_epochs):
    session.run(test_iterator.initializer)
    while True:
        try:
            feed_dict = {idx_node           :epoch_test%config.im_per_obj,
                         level_set          :config.levelset}  
            accuracy_t_ ,iou_t_, iou_image_t, iou_32t_, iou_image_32t, batch_ = session.run([test_dict['accuracy'],test_dict['iou'],test_dict['iou_image'],test_dict['iou_32'],test_dict['iou_32_image'], next_element_test],feed_dict=feed_dict) 
            acc_mov_avg_test = acc_mov_test.push(accuracy_t_)
            iou_mov_avg_test = iou_mov_test.push(iou_t_)
            iou_mov_avg_test_32 = iou_mov_test_32.push(iou_32t_)
            
            classes.append(np.tile(batch_['classes'],(config.test_size,1)))
            ids.append(np.tile(batch_['ids'],(config.test_size,1)))
            ious.append(iou_image_t) 
            ious32.append(iou_image_32t)   
#            print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,IOU: '+str(iou_mov_avg_test)+' ,IOU32: '+str(iou_mov_avg_test_32))
        except tf.errors.OutOfRangeError:
            print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,IOU: '+str(iou_mov_avg_test)+' ,IOU32: '+str(iou_mov_avg_test_32))
            break

print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,IOU: '+str(iou_mov_avg_test)+' ,IOU32: '+str(iou_mov_avg_test_32))    
classes  = np.concatenate(classes,axis=0)[:,0]
ious     = np.concatenate(ious,axis=0) 
ious32   = np.concatenate(ious32,axis=0) 
ids      = np.concatenate(ids,axis=0)[:,0]     

np.save(directory+'/classes_ls='+str(config.levelset)+'.npy',classes)
np.save(directory+'/ious_ls='+str(config.levelset)+'.npy',ious) 
np.save(directory+'/ious32_ls='+str(config.levelset)+'.npy',ious32)
np.save(directory+'/ids_ls='+str(config.levelset)+'.npy',ids)



if True==False:
    

    
    config.test_size=1
    config.batch_size=1
    idx_node          = tf.placeholder(tf.int32,shape=(), name='idx_node')  
    level_set         = tf.placeholder(tf.float32,shape=(),   name='levelset')  
    grid_size_lr = config.grid_size
    x            = np.linspace(-1, 1, grid_size_lr)
    y            = np.linspace(-1, 1, grid_size_lr)
    z            = np.linspace(-1, 1, grid_size_lr)
    xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)
    
    dispaly_iterator  = TFH.iterator(config.path+'test/',
                                  1,
                                  epochs=10000,
                                  shuffle=True,
                                  img_size=config.img_size[1],
                                  im_per_obj=config.im_per_obj,
                                  grid_size=config.grid_size,
                                  num_samples=config.num_samples,
                                  shuffle_size=config.shuffle_size,
#                                  categories = [0],
                                  categories = config.categories,
                                  compression = config.compression)

    next_element_display = dispaly_iterator.get_next()
    next_batch_display   = TFH.process_batch_test(next_element_display,idx_node,config)
    
    images                = next_batch_display['images'] 
    samples_sdf           = next_batch_display['samples_sdf']  
    samples_xyz           = next_batch_display['samples_xyz']
    evals_target          = {}
    evals_target['x']     = samples_xyz
    evals_target['y']     = samples_sdf
    evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
    g_weights             = f_wrapper(images,[mode_node,config])
    evals_function        = SF.sample_points_list(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [1,config.num_samples],samples=evals_target['x'] , use_samps=True)
    evals_function        = SF.render_sil(evals_function,evals_target,config)
    
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    loader   = tf.train.Saver(var_list=all_vars)
    
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    loader.restore(session, directory+'/latest'+config.postfix+'-0')
#    loader.restore(session, directory+'/latest_stage2-32-0')
    session.run(mode_node.assign(False)) 
    session.run(dispaly_iterator.initializer)
    feed_dict = {idx_node           :0,
                 level_set          :0}   

    evals_target_, evals_function_ = session.run([evals_target, evals_function],feed_dict=feed_dict) 

 
    
    field              = np.reshape(evals_function_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts1, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts1/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts1/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  
     
    
    
    field              = np.reshape(evals_target_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts2, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.)
        cubed_plot = {'vertices':verts2/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts2/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  
        
        
        
        
        

    