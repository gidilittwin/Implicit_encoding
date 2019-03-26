
import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
from src.utilities import raytrace as RAY
#import matplotlib.pyplot as plt
#from src.utilities import iou_loss as IOU
import os
import argparse
import socket
import tfrecords_handler as TFH

# 1) try HD grid for evaluations (we don't need to stick to 32..)     vvvvvvvvvvvvvvvv
# 2) Add batch norm to different blocks                                 vvvvvvvvvvvvvvv
# 3) set different levelsets for evaluations
# 4) Hourgalss?
# 5) different gaussin noise values         vvvvvvvvvvvvvvvvvvvvv
# 6) bring weighting back?
# 7) Flip augmentations, 
# 8) add silhouette to rgb for 32^3         vvvvvvvvvvvvvvvvvv
# 9) dropout??
# 9) lrelu/elu for cnn
# 10) more dnn layers
# 11) Spectral norm



def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'archsweep_exp79')
    parser.add_argument('--model_params_path', type=str, default= './archs/resnet_5.json')
    parser.add_argument('--padding', type=str, default= 'VALID')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--grid_size', type=int,  default=36)
    parser.add_argument('--img_size', type=int,  default=[137,137])
#    parser.add_argument('--grid_size', type=int,  default=256)
#    parser.add_argument('--img_size', type=int,  default=[224,224])  
    parser.add_argument('--eval_grid_scale', type=int,  default=1)
    parser.add_argument('--batch_size', type=int,  default=8)
    parser.add_argument('--test_size', type=int,  default=1)
    parser.add_argument('--multi_image', type=int,  default=0)
    parser.add_argument('--batch_norm', type=int,  default=1)
    parser.add_argument('--bn_l0', type=int,  default=0)
    parser.add_argument('--shuffle_rgb', type=int,  default=1)
    parser.add_argument('--rgba', type=int,  default=0)
    parser.add_argument('--symetric', type=int,  default=0)
    parser.add_argument('--radius', type=float,  default=0.1)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--global_points', type=int,  default=1000)    
    parser.add_argument('--noise_scale', type=float,  default=0.1)
    parser.add_argument('--checkpoint_every', type=int,  default=10000)
    parser.add_argument('--categories'      , type=str,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
#    parser.add_argument('--categories', type=str,  default="02691156", help='number of point samples')
    parser.add_argument('--plot_every', type=int,  default=1000)
    parser.add_argument('--test_every', type=int,  default=10000)
    parser.add_argument('--learning_rate', type=float,  default=0.00005)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=False)
    if socket.gethostname() == 'gidi-To-be-filled-by-O-E-M':
#        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/")
        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNet_TF/")
        parser.add_argument("--mesh_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetHSP/")
        parser.add_argument("--train_file"      , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/train_list.txt")
        parser.add_argument("--test_file"       , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/test_list.txt")
        parser.add_argument("--checkpoint_path" , type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNetRendering/")
        parser.add_argument("--mesh_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetMesh/ShapeNetCore.v2/")
        parser.add_argument("--iccv_path"       , type=str, default="/private/home/wolf/gidishape/data/ShapeNetHSP/")
        parser.add_argument("--train_file"      , type=str, default="/private/home/wolf/gidishape/train_list.txt")
        parser.add_argument("--test_file"       , type=str, default="/private/home/wolf/gidishape/test_list.txt")
        parser.add_argument("--checkpoint_path" , type=str, default="/private/home/wolf/gidishape/checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/private/home/wolf/gidisharchsweep_exp60ape/checkpoints/")
    return parser.parse_args()
config = parse_args()
print('#############################################################################################')
print('###############################  '+config.experiment_name+'   ################################################')
print('#############################################################################################')




if isinstance(config.categories, basestring):
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
for ii,key in enumerate(config.categories):
    classes2name[key]['id']=ii
    

if config.rgba:
    color_channels = 4
else:
    color_channels = 3


#%%
    
if config.grid_size==36:
    train_iterator = TFH.iterator(config.path+'train/',config.batch_size,epochs=10000,shuffle=True)
    test_iterator  = TFH.iterator(config.path+'test/',config.test_size,epochs=10000,shuffle=False)
    run_mode       = tf.Variable(0, name='run_node',dtype=tf.int32)
    mode_node      = tf.equal(run_mode,MODE_TRAIN, name='mode_node')
    idx_node       = tf.placeholder(tf.int32,shape=(), name='idx_node')  
    level_set      = tf.placeholder(tf.float32,shape=(),   name='levelset')  
    tf.add_to_collection('istrainvar',mode_node)
    next_element = tf.case([(tf.equal(MODE_TRAIN, run_mode), (lambda: train_iterator.get_next()  )),
                            (tf.equal(MODE_TEST,  run_mode), (lambda: test_iterator.get_next() ))],
                                                     default=(lambda: test_iterator.get_next() ) )
    next_batch      = TFH.process_batch_train(next_element,idx_node,config)
    next_batch_test = TFH.process_batch_test(next_element,idx_node,config)

   

elif config.grid_size==256:
    SN_train     = ShapeNet(config.iccv_path+'train',config.mesh_path,
                     files=[],
                     rand=True,
                     batch_size=config.batch_size,
                     grid_size=config.grid_size,
                     levelset=[0.00],
                     num_samples=config.num_samples,
                     list_=config.categories,
                     rec_mode=False)
    
    
    SN_val     = ShapeNet(config.iccv_path+'test',config.mesh_path,
                     files=[],
                     rand=False,
                     batch_size=config.batch_size,
                     grid_size=config.grid_size,
                     levelset=[0.00],
                     num_samples=config.num_samples,
                     list_=config.categories,
                     rec_mode=False)    
 


grid_size_lr = config.grid_size
x            = np.linspace(-1, 1, grid_size_lr)
y            = np.linspace(-1, 1, grid_size_lr)
z            = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)



#%% Function wrappers   
  
   
def function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_sdf2(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function


def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)

def injection_wrapper(current,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_)
    
    
def mpx_function_wrapper(encoding,args_):
    with tf.variable_scope('multiplexer_model',reuse=tf.AUTO_REUSE):
        return CNN.multiplexer(encoding,args_[0])







#%% Sampling in XYZ domain  

def build_graph(next_batch,config,batch_size):
    images                = next_batch['images'] 
    samples_sdf           = next_batch['samples_sdf']  
    samples_xyz           = next_batch['samples_xyz']
    evals_target          = {}
    evals_target['x']     = samples_xyz
    evals_target['y']     = samples_sdf
    evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
    g_weights             = CNN_function_wrapper(images,[mode_node,config])
    evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)
    labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(batch_size,-1)),0.0),tf.int64)
#    labels_float       = tf.cast(labels,tf.float32)
    logits             = tf.reshape(evals_function['y'],(batch_size,-1,1)) #- levelset
    logits_iou         = tf.concat((logits-level_set,-logits+level_set),axis=-1)
    logits_ce          = tf.concat((logits,-logits),axis=-1)
    predictions        = tf.nn.softmax(logits_iou)
    correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    delta_y            = tf.square(evals_function['y']-evals_target['y'])
#    jacobian           = evals_function['dydx']
    norm               = evals_function['dydx_norm']
#    norm_loss          = tf.reduce_mean((evals_function['dydx_norm'] - 1.0)**2)
    sample_w           = tf.squeeze(tf.exp(-(evals_target['y']-config.levelset)**2/config.radius),axis=-1)
    loss_class         = tf.reduce_mean(sample_w*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_ce,name='cross-entropy'),axis=-1)
    loss_class         = loss_class/tf.reduce_mean(sample_w,axis=-1)
    loss               = tf.reduce_mean(loss_class )
    X                  = tf.cast(labels,tf.bool)
    Y                  = tf.cast(tf.argmax(predictions, 2),tf.bool)
    iou_image          = tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1)
    iou                = tf.reduce_mean(iou_image)
    features           = tf.get_collection('embeddings')
    return loss,accuracy,err,norm,iou,features


loss,accuracy,err,norm,iou,features             = build_graph(next_batch,config,batch_size=config.batch_size)
loss_t,accuracy_t,err_t,norm_t,iou_t,features_t = build_graph(next_batch_test,config,batch_size=config.test_size)

#injected_embeddings   = tf.placeholder(tf.float32,shape=(None,2048),   name='injected_embeddings')  
#function_injected     = injection_wrapper(injected_embeddings,[mode_node,config])
#evals_function_inject = function_wrapper(evals_target['x'],args_=[mode_node,function_injected,config])
#logits_inject         = tf.reshape(evals_function_inject,(config.batch_size,-1,1)) #- levelset
#logits_inject_iou     = tf.concat((logits_inject-level_set,-logits_inject+level_set),axis=-1)
#predictions_inject    = tf.nn.softmax(logits_inject_iou)
#X_inject              = tf.cast(labels,tf.bool)
#Y_inject              = tf.cast(tf.argmax(predictions_inject , 2),tf.bool)
#iou_image_inject      = tf.reduce_sum(tf.cast(tf.logical_and(X_inject,Y_inject),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X_inject,Y_inject),tf.float32),axis=1)





with tf.variable_scope('optimization_cnn',reuse=tf.AUTO_REUSE):
    cnn_vars      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = '2d_cnn_model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node,beta1=0.9,beta2=0.999)
    grads         = optimizer.compute_gradients(loss,var_list=cnn_vars)
    global_step   = tf.train.get_or_create_global_step()
    clip_constant = 10
    g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
    train_op_cnn  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)


all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(var_list=all_vars)
loader = tf.train.Saver(var_list=all_vars)



#%% Train
def evaluate(test_iterator, mode_node, config, accuracy_t, iou_t):
    session.run(run_mode.assign(1)) 
    aa_mov_test    = MOV_AVG(3000000) 
    dd_mov_test    = MOV_AVG(3000000) 
    for epoch_test in range(24):
        session.run(test_iterator.initializer)
        while True:
            try:
                feed_dict = {lr_node            :config.learning_rate,
                             idx_node           :epoch_test%24,
                             level_set          :config.levelset}  
                accuracy_t ,iou_t = session.run([accuracy_t, iou_t],feed_dict=feed_dict) 
                aa_mov_avg_test = aa_mov_test.push(accuracy_t)
                dd_mov_avg_test = dd_mov_test.push(iou_t)
                print(str(dd_mov_avg_test))
            except tf.errors.OutOfRangeError:
                print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(aa_mov_avg_test)+' ,IOU: '+str(dd_mov_avg_test))
                break
    session.run(run_mode.assign(0)) 
    return aa_mov_avg_test, dd_mov_avg_test



    
session = tf.Session()
session.run(tf.initialize_all_variables())
loss_plot      = []
acc_plot       = []
iou_plot       = []
acc_plot_test  = []
iou_plot_test  = []
max_test_acc   = 0.
max_test_iou   = 0.
if config.finetune:
    loader.restore(session, directory+'/latest-0')
#    loader.restore(session, '/private/home/wolf/gidishape/checkpoints/study_dnn_arch30/latest-0')
    loss_plot     = np.load(directory+'/loss_values.npy')
    acc_plot      = np.load(directory+'/accuracy_values.npy')  
    iou_plot      = np.load(directory+'/iou_values.npy')      
    acc_plot_test = np.load(directory+'/accuracy_values_test.npy') 
    iou_plot_test = np.load(directory+'/iou_values_test.npy') 
    loss_plot     = np.split(loss_plot,loss_plot.shape[0])
    acc_plot      = np.split(acc_plot,acc_plot.shape[0])
    iou_plot      = np.split(iou_plot,iou_plot.shape[0])
    acc_plot_test = np.split(acc_plot_test,acc_plot_test.shape[0])
    iou_plot_test = np.split(iou_plot_test,iou_plot_test.shape[0])    
step           = 0
aa_mov         = MOV_AVG(300) 
bb_mov         = MOV_AVG(300) 
cc_mov         = MOV_AVG(300) 
dd_mov         = MOV_AVG(300) 

session.run(run_mode.assign(0)) 
for epoch in range(1000):
    session.run(train_iterator.initializer)
    while True:
        try:
            feed_dict = {lr_node            :config.learning_rate,
                         idx_node           :epoch%24,
                         level_set          :config.levelset}     
#            _, loss_, accuracy_ ,iou_ = session.run([train_op_cnn, loss ,accuracy, iou],feed_dict=feed_dict) 
            acc_test, iou_test = evaluate(test_iterator, mode_node, config, accuracy_t, iou_t)
            
            aa_mov_avg = aa_mov.push(accuracy_)
            cc_mov_avg = cc_mov.push(loss_)
            dd_mov_avg = dd_mov.push(iou_)   
            print(str(aa_mov_avg))
        except tf.errors.OutOfRangeError:
            acc_test, iou_test = evaluate(test_iterator, mode_node, config, accuracy_t, iou_t)
            
            acc_plot.append(np.expand_dims(np.array(aa_mov_avg),axis=-1))
            loss_plot.append(np.expand_dims(np.array(np.log(cc_mov_avg)),axis=-1))
            iou_plot.append(np.expand_dims(np.array(dd_mov_avg),axis=-1))
            np.save(directory+'/loss_values.npy',np.concatenate(loss_plot))
            np.save(directory+'/accuracy_values.npy',np.concatenate(acc_plot))  
            np.save(directory+'/iou_values.npy',np.concatenate(iou_plot))  
            acc_plot_test.append(np.expand_dims(np.array(acc_test),axis=-1))
            iou_plot_test.append(np.expand_dims(np.array(iou_test),axis=-1))
            np.save(directory+'/accuracy_values_test.npy',np.concatenate(acc_plot_test))  
            np.save(directory+'/iou_values_test.npy',np.concatenate(iou_plot_test)) 
            print('Training: epoch: '+str(epoch)+' ,avg_accuracy: '+str(aa_mov_avg)+' ,avg_loss: '+str(cc_mov_avg)+' ,IOU: '+str(dd_mov_avg))
            print('Testing:  max_test_accuracy: '+str(max_test_acc)+' ,max_test_IOU: '+str(max_test_iou))
            if iou_test>max_test_iou:
                saver.save(session, directory+'/'+str(step), global_step=step)
                saver.save(session, directory+'/latest', global_step=0)
            max_test_acc = np.max([np.max(np.concatenate(acc_plot_test)),max_test_acc])
            max_test_iou = np.max([np.max(np.concatenate(iou_plot_test)),max_test_iou])
            break
 


#%% TEST VANILLA
def test(SN_test, mode_node, config, accuracy, iou_image, features, levelset):
    if config.grid_size==36:
        grid_size_lr   = 32*config.eval_grid_scale
        x_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        y_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        z_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
    else:
        grid_size_lr   = config.grid_size*config.eval_grid_scale
        x_lr           = np.linspace(-1, 1, grid_size_lr)
        y_lr           = np.linspace(-1, 1, grid_size_lr)
        z_lr           = np.linspace(-1, 1, grid_size_lr)    
    xx_lr,yy_lr,zz_lr    = np.meshgrid(x_lr, y_lr, z_lr)    
    step_test      = 0
    aa_mov_test    = MOV_AVG(3000000) 
    dd_mov_test    = MOV_AVG(3000000) 
    samples_xyz_np = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
    samples_ijk_np = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
    embeddings = []
    classes    = []
    ids        = []
    ious       = []
    while SN_test.epoch<2:
        batch                = SN_test.get_batch_multi(type_='')
        samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
        feed_dict = {images             :batch['images'][:,:,:,0:3]/255.,
                     samples_xyz        :np.tile(samples_xyz_np,[config.batch_size,1,1]),
                     samples_sdf        :samples_sdf_np,
                     level_set          :levelset,}
#                     injected_embeddings:np.zeros((1,2048),dtype=np.float32)}     
        accuracy_t ,iou_image_t, features_t = session.run([accuracy, iou_image, features],feed_dict=feed_dict) 
        aa_mov_avg_test = aa_mov_test.push(accuracy_t)
        dd_mov_avg_test = dd_mov_test.push(np.mean(iou_image_t))
        step_test+=1
        embeddings.append(features_t)
        classes.append(batch['classes'])
        ids.append(batch['ids'])
        ious.append(iou_image_t)
        if step_test % 100==0:
            print('TEST::  epoch: '+str(SN_test.epoch)+' step: '+str(step_test)+' ,avg_accuracy: '+str(aa_mov_avg_test)+' ,IOU: '+str(dd_mov_avg_test))
    SN_test.epoch = 0
    return aa_mov_avg_test, dd_mov_avg_test, embeddings, classes, ids, ious

session.run(mode_node.assign(False)) 
levelset = 0.2
acc_test, iou_test, features_test, classes_test, ids_test, ious_test = test(SN_val, mode_node, config, accuracy, iou_image, features[0], levelset)

# saving and loading
Features = np.reshape(np.concatenate(features_test,axis=0) ,(-1,1024))
Classes  = np.concatenate(classes_test,axis=0) 
Classes = Classes[:,0]
ids      = np.concatenate(ids_test,axis=0) 
ious     = np.concatenate(ious_test,axis=0) 
ids=ids[0:-24]
ious=ious[0:-24]
Classes=Classes[0:-24]
Features=Features[0:-24,:]
np.savez_compressed( '/media/gidi/SSD/Thesis/Data/Checkpoints/archsweep_exp79/Features_mv_ls=0.2.npz', Features=Features)
np.save( '/media/gidi/SSD/Thesis/Data/Checkpoints/archsweep_exp60/Meta_mv12_ls=0.2', {'classes':Classes,'ids':ids,'ious':ious})


#Features = np.load( '/media/gidi/SSD/Thesis/Data/Checkpoints/archsweep_exp60/Features_ls=0.1.npz')
#Features = Features['Features']
#meta     = np.load( '/media/gidi/SSD/Thesis/Data/Checkpoints/archsweep_exp60/Meta_ls=0.1.npy')
#ious_loaded      = meta.item().get('ious')






   
    
    

#%% VISUALIZE & INTERPOLATE
#cats=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"]
session.run(mode_node.assign(False))
import matplotlib as mpl 
import time
import matplotlib.pyplot as plt
mpl.style.use('default')
from scipy import misc

grid_size_lr = config.grid_size*3
x            = np.linspace(-1, 1, grid_size_lr)
y            = np.linspace(-1, 1, grid_size_lr)
z            = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)


SN_vis1        = ShapeNet(config.path,config.mesh_path,
                 files=config.test_file,
                 rand=True,
                 batch_size=config.batch_size,
                 grid_size=config.grid_size,
                 levelset=[0.00],
                 num_samples=config.num_samples,
                 list_=["04090263"],
                 rec_mode=False,
                 shuffle_rgb=False)   


SN_vis2        = ShapeNet(config.path,config.mesh_path,
                 files=config.test_file,
                 rand=True,
                 batch_size=config.batch_size,
                 grid_size=config.grid_size,
                 levelset=[0.00],
                 num_samples=config.num_samples,
                 list_=["03001627"],
                 rec_mode=False,
                 shuffle_rgb=False)   





batch1                = SN_vis1.get_batch_multi(type_='')
samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
samples_sdf_np       = np.expand_dims(batch1['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
feed_dict1 = {images           :batch1['images'][:,:,:,0:3]/255.,
             samples_xyz      :np.tile(samples_xyz_np,[config.batch_size,1,1]),
             samples_sdf      :samples_sdf_np,
             level_set        :config.levelset,}
evals_function_d1,accuracy_1 ,iou_image_1,features_1  = session.run([evals_function['y'],accuracy,iou_image,features],feed_dict=feed_dict1) # <= returns jpeg data you can write to disk    
 #Visualize
idx1 = np.argmax(iou_image_1) 
for example in range(1):
    field              = np.reshape(evals_function_d1[idx1,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  

        
        
batch2                = SN_vis2.get_batch_multi(type_='')
batch2_feed           = SN_vis2.process_batch(batch2,config)
samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
samples_sdf_np       = np.expand_dims(batch2['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
feed_dict2 = {images           :batch2['images'][:,:,:,0:3]/255.,
             samples_xyz      :np.tile(samples_xyz_np,[config.batch_size,1,1]),
             samples_sdf      :samples_sdf_np,
             level_set        :config.levelset,}
evals_function_d2,accuracy_2 ,iou_image_2,features_2  = session.run([evals_function['y'],accuracy,iou_image,features],feed_dict=feed_dict2) # <= returns jpeg data you can write to disk    
 #Visualize
idx2 = np.argmax(iou_image_2) 
for example in range(1):
    field              = np.reshape(evals_function_d2[idx2,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], -1.1)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  
#        mesh_plot([cubed_plot],idx=0,type_='cloud')  
#        cubed_plot = {'vertices':batch2_feed['samples_xyz_np'][0,:,:],'faces':faces,'vertices_up':batch2_feed['samples_xyz_np'][0,:,:]}
#        mesh_plot([cubed_plot],idx=0,type_='cloud') 
        
        
        
#samples_xyz_np       = np.expand_dims(verts,0)/(grid_size_lr-1)*2-1
#samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
#samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
#feed_dict = {images           :batch['images'][:,:,:,0:3]/255.,
#             samples_xyz      :np.tile(samples_xyz_np,[config.batch_size,1,1]),
#             samples_sdf      :samples_sdf_np,
#             level_set        :config.levelset,
#             injected_embeddings:np.zeros((1,2048),dtype=np.float32)}     
#evals_function_d,accuracy_ ,iou_image_,features_,jacobian_,norm_  = session.run([evals_function['y'],accuracy,iou_image,features,jacobian,norm],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#jacobian_ =jacobian_/norm_
#for example in range(1):
#    point_cloud = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1,'jacobian':jacobian_[0,:,:],'norm':norm_[0,:,:]}
##    MESHPLOT.mesh_plot([point_cloud],idx=0,type_='mesh') 
#    mesh_plot([point_cloud],idx=0,type_='mesh')  




# Interpolate:
features_plane =    features_1[0][idx1:idx1+1,:].copy()     
features_car   =    features_2[0][idx2:idx2+1,:].copy()    
#features_interp = []
for alpha in np.linspace(0,1,5):
#    features_interp.append(alpha*features_car + (1-alpha)*features_plane)
    features_interp = alpha*features_car + (1-alpha)*features_plane
#    features_interp = np.concatenate(features_interp,0)  
    feed_dict = {samples_xyz      :np.tile(samples_xyz_np,[1,1,1]),
                 level_set        :config.levelset,
                 injected_embeddings:features_interp}     
    evals_function_injected_ = session.run([evals_function_inject],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#    for example in range(0,5):
    field              = np.reshape(evals_function_injected_[0][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')   
        time.sleep(2.0)
       



 
#%% MULTIIMAGE
def test_multi(SN_multi, mode_node, config, iou_image, iou_image_inject, features, levelset):
    classes    = []
    ids        = []
    ious       = []
    ious_multi = []
    SN_multi.epoch = 0
    SN_multi.reset()
    if config.grid_size==36:
        grid_size_lr   = 32*config.eval_grid_scale
        x_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        y_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        z_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
    else:
        grid_size_lr   = config.grid_size*config.eval_grid_scale
        x_lr           = np.linspace(-1, 1, grid_size_lr)
        y_lr           = np.linspace(-1, 1, grid_size_lr)
        z_lr           = np.linspace(-1, 1, grid_size_lr)      
    while SN_multi.epoch<2:
        batch_multi          = SN_multi.get_batch_multi(type_='')
        samples_xyz_np       = np.tile(np.reshape(np.stack((x_lr,y_lr,z_lr),axis=-1),(1,-1,3)),(1,1,1))
        samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
        samples_sdf_np       = np.expand_dims(batch_multi['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
        feed_dict = {images           :batch_multi['images'][:,:,:,0:3]/255.,
                     samples_xyz      :np.tile(samples_xyz_np,[config.batch_size,1,1]),
                     samples_sdf      :samples_sdf_np,
                     level_set        :config.levelset,
                     injected_embeddings:np.zeros((24,2048),dtype=np.float32)}     
        evals_function_d ,iou_image_,features_  = session.run([evals_function['y'],iou_image,features[0]],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#        print('mean batch iou:'+str(np.mean(iou_image_)))
        
        mean_features_ =    np.tile(np.mean(features_,axis=0,keepdims=True) ,(24,1)) 
        feed_dict = {samples_xyz      :np.tile(samples_xyz_np,[24,1,1]),
                     level_set        :config.levelset,
                     samples_sdf      :samples_sdf_np,
                     injected_embeddings:mean_features_}     
        evals_function_injected_m,iou_image_inject_m = session.run([evals_function_inject,iou_image_inject],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#        print('mean batch iou:'+str(np.mean(iou_image_inject_m)))
        print('TEST::  epoch: '+str(SN_multi.epoch)+' step: '+str(SN_multi.train_step))
        classes.append(batch_multi['classes'])
        ids.append(batch_multi['ids'])
        ious.append(iou_image_)
        ious_multi.append(iou_image_inject_m)
    return classes,ids,ious,ious_multi

session.run(mode_node.assign(False)) 
levelset = 0.0
classes_m,ids_m,ious_m,ious_multi_m = test_multi(SN_multi, mode_node, config, iou_image, iou_image_inject, features, levelset)
ious_m           = np.concatenate(ious_m,axis=0) 
ious_multi_m     = np.concatenate(ious_multi_m,axis=0) 
ids_m            = np.concatenate(ids_m,axis=0) 

ious_m=ious_m[0:-24]
ious_multi_m=ious_multi_m[0:-24]

for example in range(0,4):
    field              = np.reshape(evals_function_d[example,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')   
        time.sleep(1.0)
     


#%% RAY-TRACE
    
#cam_pos      = tf.placeholder(tf.float32,shape=(1,1,3),name='camera_pos')  
#cam_mat      = tf.placeholder(tf.float32,shape=(1,3,3),name='camera_mat')  
#position     = (samples_xyz+1.)/2. * 223 * 0.57
#pt_trans     = tf.matmul(position-cam_pos, tf.transpose(cam_mat,(0,2,1)))   
#X,Y,Z        = tf.split(pt_trans,[1,1,1],axis=-1)
#F            = 248
#h            = (-Y)/(-Z)*F + 224/2.0
#w            = X/(-Z)*F + 224/2.0
#h            = tf.clip_by_value(h,clip_value_min=0,clip_value_max=223)
#w            = tf.clip_by_value(w,clip_value_min=0,clip_value_max=223)
#h_idx               = tf.cast(tf.round(h),tf.int64)
#w_idx               = tf.cast(tf.round(w),tf.int64)
#point_cloud_lin_idx = h_idx + 223*w_idx
#vox_idx, idx, count = tf.unique_with_counts(tf.squeeze(point_cloud_lin_idx),out_idx=tf.int32)
#values              = -1*tf.unsorted_segment_max(-1*tf.squeeze(evals_function['y'],0),idx,tf.reduce_max(idx)+1)
#max_values          = tf.gather(values,idx)
#point_cloud_idx     = tf.squeeze(tf.concat((h_idx,w_idx),axis=-1),0)
#max_values_idx      = tf.gather(point_cloud_idx,idx)
#voxels              = tf.scatter_nd(max_values_idx, max_values, (224,224,1))
#    
#   
#session = tf.Session()
#session.run(tf.initialize_all_variables())
#if config.finetune:
#    loader.restore(session, config.saved_model_path)
#samples_xyz_np       = np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3))
#samples_xyz_tile_np  = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(config.batch_size,1,1))
#samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
#
#session.run(mode_node.assign(False)) 
#while step < 100000000:
#    batch                = SN_train.get_batch(type_='')
#    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)   
#    
#    
#    feed_dict = {images           :batch['images']/255.,
#                 samples_xyz      :samples_xyz_tile_np,
#                 samples_sdf      :samples_sdf_np,
#                 cam_pos          :np.expand_dims(batch['camera_pose'],0),
#                 cam_mat          :batch['camera_mat'],
#                 }
#    evals_function_d,accuracy_ ,voxels_  = session.run([evals_function['y'],accuracy,voxels],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#    
# 
#import matplotlib.pyplot as plt
#
#pic = batch['images'][0,:,:,:]
#fig = plt.figure()
#plt.imshow(pic/255.)
#
#
#pic = np.squeeze(batch['alpha'][0,:,:,:])
#fig = plt.figure()
#plt.imshow(pic)
#
#projection = np.tanh(voxels_[:,:,0])
#fig = plt.figure()
#plt.imshow(projection)



#%% RAY-TRACE

#ray_steps,reset_opp,evals  = Ray_render.eval_func(model_fn = function_wrapper,args=[mode_node,theta],
#                                                 step_size=None, 
#                                                 epsilon = 0.0001, 
#                                                 temp=1., 
#                                                 ambient_weight=0.3)
#
#
#
#init_new_vars_op = tf.initialize_variables(Ray_render_vars)
#session.run(init_new_vars_op)
#Ray_render.trace(session,ray_steps,reset_opp,num_steps=50)
#
#
#
#evals_             = session.run(evals) # <= returns jpeg data you can write to disk
#target_render      = evals_['raw_images'][0]
#target_normals     = (np.transpose(evals_['normals'][0],(1,2,0))+1.)/2
#target_incident    = evals_['incidence_angles'][0]
#target_dist        = evals_['distances'][0]
#
#
#fig = plt.figure()
#title('target_image')
#ax2 = fig.add_subplot(1, 1, 1)
#ax2.imshow(target_render.astype(np.uint8))
#
#fig = plt.figure()
#title('target_image_normals')
#ax2 = fig.add_subplot(1, 1, 1)
#ax2.imshow(target_normals)
#
#fig = plt.figure()
#title('target_dist')
#ax2 = fig.add_subplot(1, 1, 1)
#ax2.imshow(target_dist)

