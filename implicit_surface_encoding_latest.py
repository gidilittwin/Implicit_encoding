
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
    parser.add_argument('--experiment_name', type=str, default= 'archsweep_exp60')
    parser.add_argument('--model_params_path', type=str, default= './archs/resnet_5w2.json')
    parser.add_argument('--padding', type=str, default= 'VALID')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--grid_size', type=int,  default=36)
    parser.add_argument('--img_size', type=int,  default=[137,137])
#    parser.add_argument('--grid_size', type=int,  default=256)
#    parser.add_argument('--img_size', type=int,  default=[224,224])  
    parser.add_argument('--eval_grid_scale', type=int,  default=1)
    parser.add_argument('--batch_size', type=int,  default=32)
    parser.add_argument('--batch_norm', type=int,  default=0)
    parser.add_argument('--bn_l0', type=int,  default=0)
    parser.add_argument('--shuffle_rgb', type=int,  default=1)
    parser.add_argument('--rgba', type=int,  default=0)
    parser.add_argument('--symetric', type=int,  default=0)
    parser.add_argument('--radius', type=float,  default=0.1)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--global_points', type=int,  default=1000)    
    parser.add_argument('--noise_scale', type=float,  default=0.1)
    parser.add_argument('--checkpoint_every', type=int,  default=10000)
    parser.add_argument('--categories', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
#    parser.add_argument('--categories', type=int,  default=["02691156"], help='number of point samples')
    parser.add_argument('--plot_every', type=int,  default=1000)
    parser.add_argument('--test_every', type=int,  default=10000)
    parser.add_argument('--learning_rate', type=float,  default=0.00005)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=True)
    if socket.gethostname() == 'gidi-To-be-filled-by-O-E-M':
        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNetRendering/")
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
        parser.add_argument("--saved_model_path", type=str, default="/private/home/wolf/gidishape/checkpoints/")
    return parser.parse_args()
config = parse_args()
print('#############################################################################################')
print('###############################  '+config.experiment_name+'   ################################################')
print('#############################################################################################')







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

with open(MODEL_PARAMS, 'r') as f:
    model_params = json.load(f)
config.model_params = model_params    
    
    
directory = config.checkpoint_path + config.experiment_name 
if not os.path.exists(directory):
    os.makedirs(directory)


if config.rgba:
    color_channels = 4
else:
    color_channels = 3


#%%
SN_train       = ShapeNet(config.path,config.mesh_path,
                 files=config.train_file,
                 rand=True,
                 batch_size=config.batch_size,
                 grid_size=config.grid_size,
                 levelset=[0.00],
                 num_samples=config.num_samples,
                 list_=config.categories,
                 rec_mode=False,
                 shuffle_rgb=config.shuffle_rgb)

SN_test        = ShapeNet(config.path,config.mesh_path,
                 files=config.test_file,
                 rand=False,
                 batch_size=config.batch_size,
                 grid_size=config.grid_size,
                 levelset=[0.00],
                 num_samples=config.num_samples,
                 list_=config.categories,
                 rec_mode=False,
                 shuffle_rgb=config.shuffle_rgb)


#SN_train     = ShapeNet(config.iccv_path+'train',config.mesh_path,
#                 files=[],
#                 rand=True,
#                 batch_size=config.batch_size,
#                 grid_size=config.grid_size,
#                 levelset=[0.00],
#                 num_samples=config.num_samples,
#                 list_=config.categories,
#                 rec_mode=False)
#
#
#SN_test     = ShapeNet(config.iccv_path+'test',config.mesh_path,
#                 files=[],
#                 rand=False,
#                 batch_size=config.batch_size,
#                 grid_size=config.grid_size,
#                 levelset=[0.00],
#                 num_samples=config.num_samples,
#                 list_=config.categories,
#                 rec_mode=False)    
 



#%% Function wrappers   
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')
    
   
def function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_sdf2(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function


def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)


def mpx_function_wrapper(encoding,args_):
    with tf.variable_scope('multiplexer_model',reuse=tf.AUTO_REUSE):
        return CNN.multiplexer(encoding,args_[0])



with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
    Ray_render = RAY.Raycast(config.batch_size,resolution=(137,137),sky_color=(127.5, 127.5, 127.5))
    Ray_render.add_lighting(position=[0.0, 0.0, 1.0],color=[255., 120., 20.],ambient_color = [255., 255., 255.])
    Ray_render.add_camera(camera_position=[1.0, 1.0, 1.0], lookAt=(0,0,0),focal_length=2,name='camera_1')
Ray_render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'display_tracer')





#%% Sampling in XYZ domain  
images                = tf.placeholder(tf.float32,shape=(None,config.img_size[0],config.img_size[1],color_channels), name='images')  
samples_sdf           = tf.placeholder(tf.float32,shape=(None,None,1), name='samples_sdf')  
samples_xyz           = tf.placeholder(tf.float32,shape=(None,None,3),   name='samples_xyz')  
evals_target          = {}
evals_target['x']     = samples_xyz
evals_target['y']     = samples_sdf
evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
embeddings            = CNN_function_wrapper(images,[mode_node,config])
evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,embeddings,config],shape = [config.batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)


labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(config.batch_size,-1)),config.levelset),tf.int64)
labels_float       = tf.cast(labels,tf.float32)
logits             = tf.reshape(evals_function['y'],(config.batch_size,-1,1)) #- levelset
logits             = tf.concat((logits,-logits),axis=-1)
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_y            = tf.square(evals_function['y']-evals_target['y'])
norm               = tf.reduce_max(tf.abs(evals_function['dydx_norm']))
norm_loss          = tf.reduce_mean((evals_function['dydx_norm'] - 1.0)**2)
sample_w           = tf.squeeze(tf.exp(-(evals_target['y']-config.levelset)**2/config.radius),axis=-1)
loss_class         = tf.reduce_mean(sample_w*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'),axis=-1)
loss_class         = loss_class/tf.reduce_mean(sample_w,axis=-1)
iou_logit          = logits[:,:,1:2]
iou_target         = tf.expand_dims(labels_float,-1)


#vae                = tf.get_collection('VAE_loss')
#log_stddev         = vae[0][1]
#mean               = vae[0][0]
#vae_loss           = 0.01 *tf.reduce_mean( tf.square(mean) + tf.square(tf.exp(log_stddev))  -1. - log_stddev, axis=-1)
#loss               = tf.reduce_mean(loss_class + vae_loss)

loss               = tf.reduce_mean(loss_class )
#loss               = IOU.lovasz_hinge(iou_logit, iou_target, per_image=False)
X                  = tf.cast(labels,tf.bool)
Y                  = tf.cast(tf.argmax(predictions, 2),tf.bool)
iou                = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1))
features           = tf.get_collection('embeddings')





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
loader = tf.train.Saver(var_list=cnn_vars)



#%% Train
def evaluate(SN_test, mode_node, config, accuracy, iou):
    session.run(mode_node.assign(False)) 
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
    while SN_test.epoch<25:
        batch                = SN_test.get_batch(type_='')
        samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
        feed_dict = {images             :batch['images'][:,:,:,0:3]/255.,
                     samples_xyz        :np.tile(samples_xyz_np,[config.batch_size,1,1]),
                     samples_sdf        :samples_sdf_np}     
        accuracy_t ,iou_t = session.run([accuracy, iou],feed_dict=feed_dict) 
        aa_mov_avg_test = aa_mov_test.push(accuracy_t)
        dd_mov_avg_test = dd_mov_test.push(iou_t)
        step_test+=1
        if step_test % 100==0:
            print('TEST::  epoch: '+str(SN_test.epoch)+' step: '+str(step_test)+' ,avg_accuracy: '+str(aa_mov_avg_test)+' ,IOU: '+str(dd_mov_avg_test))
    SN_test.epoch = 0
    session.run(mode_node.assign(True)) 
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



session.run(mode_node.assign(True)) 
while step < 100000000:
    batch                = SN_train.get_batch(type_='')
    batch_feed           = SN_train.process_batch(batch,config)
    feed_dict = {images             :batch_feed['images']/255.,
                 lr_node            :config.learning_rate,
                 samples_xyz        :batch_feed['samples_xyz_np'],
                 samples_sdf        :batch_feed['samples_sdf_np'],
                 }     
    _, loss_,norm_, accuracy_ ,iou_ = session.run([train_op_cnn, loss, norm ,accuracy, iou],feed_dict=feed_dict) 

    aa_mov_avg = aa_mov.push(accuracy_)
    cc_mov_avg = cc_mov.push(loss_)
    dd_mov_avg = dd_mov.push(iou_)
    if step % 100==0:
        print('Training: epoch: '+str(SN_train.epoch)+' step: '+str(step)+' ,avg_accuracy: '+str(aa_mov_avg)+' ,avg_loss: '+str(cc_mov_avg)+' ,IOU: '+str(dd_mov_avg))
        print('Testing:  max_test_accuracy: '+str(max_test_acc)+' ,max_test_IOU: '+str(max_test_iou))
    if step % config.checkpoint_every == 0 and step!=0:
        saver.save(session, directory+'/'+str(step), global_step=step)
        saver.save(session, directory+'/latest', global_step=0)
        last_saved_step = step
    if step % config.plot_every == 0:
        acc_plot.append(np.expand_dims(np.array(aa_mov_avg),axis=-1))
        loss_plot.append(np.expand_dims(np.array(np.log(cc_mov_avg)),axis=-1))
        iou_plot.append(np.expand_dims(np.array(dd_mov_avg),axis=-1))
        np.save(directory+'/loss_values.npy',np.concatenate(loss_plot))
        np.save(directory+'/accuracy_values.npy',np.concatenate(acc_plot))  
        np.save(directory+'/iou_values.npy',np.concatenate(iou_plot))  
    if step % config.test_every == config.test_every -1:
        acc_test, iou_test = evaluate(SN_test, mode_node, config, accuracy, iou)
        acc_plot_test.append(np.expand_dims(np.array(acc_test),axis=-1))
        iou_plot_test.append(np.expand_dims(np.array(iou_test),axis=-1))
        np.save(directory+'/accuracy_values_test.npy',np.concatenate(acc_plot_test))  
        np.save(directory+'/iou_values_test.npy',np.concatenate(iou_plot_test)) 
        max_test_acc = np.max([np.max(np.concatenate(acc_plot_test)),max_test_acc])
        max_test_iou = np.max([np.max(np.concatenate(iou_plot_test)),max_test_iou])
    step+=1




#%% TEST
    


def test(SN_test, mode_node, config, accuracy, iou, features):
    session.run(mode_node.assign(False)) 
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
    ious       = []
    while SN_test.epoch<2:
        batch                = SN_test.get_batch(type_='')
        samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
        feed_dict = {images             :batch['images'][:,:,:,0:3]/255.,
                     samples_xyz        :np.tile(samples_xyz_np,[config.batch_size,1,1]),
                     samples_sdf        :samples_sdf_np}     
        accuracy_t ,iou_t, features_t = session.run([accuracy, iou, features],feed_dict=feed_dict) 
        aa_mov_avg_test = aa_mov_test.push(accuracy_t)
        dd_mov_avg_test = dd_mov_test.push(iou_t)
        step_test+=1
        embeddings.append(features_t)
        classes.append(batch['classes'])
        ious.append(batch['classes'])
        if step_test % 100==0:
            print('TEST::  epoch: '+str(SN_test.epoch)+' step: '+str(step_test)+' ,avg_accuracy: '+str(aa_mov_avg_test)+' ,IOU: '+str(dd_mov_avg_test))
    SN_test.epoch = 0
    session.run(mode_node.assign(True)) 
    return aa_mov_avg_test, dd_mov_avg_test, embeddings, classes


    
acc_test, iou_test, features_test, classes_test = test(SN_test, mode_node, config, accuracy, iou, features)
Features = np.reshape(np.concatenate(features_test,axis=0) ,(-1,2048))
Classes  = np.concatenate(classes_test,axis=0) 
perm = np.random.permutation(Features.shape[0])
X = Features[perm[:500],:]
Y = Classes[perm[:500]]


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(Y, colors, config.categories):
    plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
plt.legend()
plt.show()


#%% VISUALIZE




grid_size_lr =  2*config.grid_size
x           = np.linspace(-1, 1, grid_size_lr)
y           = np.linspace(-1, 1, grid_size_lr)
z           = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)

session.run(mode_node.assign(False)) 
example=0
samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
batch                = SN_train.get_batch(type_='')
samples_sdf_np       = np.expand_dims(batch['sdf'][example:example+1,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
feed_dict = {images           :batch['images'][:,:,:,0:3][example:example+1,:,:,:]/255.,
             samples_xyz      :samples_xyz_np,
             samples_sdf      :samples_sdf_np}
evals_function_d,accuracy_   = session.run([evals_function['y'],accuracy],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
field              = np.reshape(evals_function_d[0,:,:],(-1,))
field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
    verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
    cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
    verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][example,:,:,:], config.levelset)
    cubed = {'vertices':verts/(config.grid_size-1)*2-1,'faces':faces}
    MESHPLOT.double_mesh_plot([cubed_plot,cubed],idx=0,type_='cubed')





#%% RAY-TRACE
    
cam_pos      = tf.placeholder(tf.float32,shape=(1,1,3),name='camera_pos')  
cam_mat      = tf.placeholder(tf.float32,shape=(1,3,3),name='camera_mat')  
position     = (samples_xyz+1.)/2. * 223 * 0.57
pt_trans     = tf.matmul(position-cam_pos, tf.transpose(cam_mat,(0,2,1)))   
X,Y,Z        = tf.split(pt_trans,[1,1,1],axis=-1)
F            = 248
h            = (-Y)/(-Z)*F + 224/2.0
w            = X/(-Z)*F + 224/2.0
h            = tf.clip_by_value(h,clip_value_min=0,clip_value_max=223)
w            = tf.clip_by_value(w,clip_value_min=0,clip_value_max=223)
h_idx               = tf.cast(tf.round(h),tf.int64)
w_idx               = tf.cast(tf.round(w),tf.int64)
point_cloud_lin_idx = h_idx + 223*w_idx
vox_idx, idx, count = tf.unique_with_counts(tf.squeeze(point_cloud_lin_idx),out_idx=tf.int32)
values              = -1*tf.unsorted_segment_max(-1*tf.squeeze(evals_function['y'],0),idx,tf.reduce_max(idx)+1)
max_values          = tf.gather(values,idx)
point_cloud_idx     = tf.squeeze(tf.concat((h_idx,w_idx),axis=-1),0)
max_values_idx      = tf.gather(point_cloud_idx,idx)
voxels              = tf.scatter_nd(max_values_idx, max_values, (224,224,1))
    
   
session = tf.Session()
session.run(tf.initialize_all_variables())
if config.finetune:
    loader.restore(session, config.saved_model_path)
samples_xyz_np       = np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3))
samples_xyz_tile_np  = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(config.batch_size,1,1))
samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)

session.run(mode_node.assign(False)) 
while step < 100000000:
    batch                = SN_train.get_batch(type_='')
    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)   
    
    
    feed_dict = {images           :batch['images']/255.,
                 samples_xyz      :samples_xyz_tile_np,
                 samples_sdf      :samples_sdf_np,
                 cam_pos          :np.expand_dims(batch['camera_pose'],0),
                 cam_mat          :batch['camera_mat'],
                 }
    evals_function_d,accuracy_ ,voxels_  = session.run([evals_function['y'],accuracy,voxels],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
    
 
import matplotlib.pyplot as plt

pic = batch['images'][0,:,:,:]
fig = plt.figure()
plt.imshow(pic/255.)


pic = np.squeeze(batch['alpha'][0,:,:,:])
fig = plt.figure()
plt.imshow(pic)

projection = np.tanh(voxels_[:,:,0])
fig = plt.figure()
plt.imshow(projection)



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

