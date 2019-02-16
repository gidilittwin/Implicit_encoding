
import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
import matplotlib.pyplot as plt
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
from src.utilities import raytrace as RAY
import matplotlib.pyplot as plt
from src.utilities import iou_loss as IOU


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


MODEL_PARAMS = './model_params.json'
with open(MODEL_PARAMS, 'r') as f:
    config = json.load(f)







#%%

rand     = True
rec_mode = False
#BATCH_SIZE = 1
SN_train       = ShapeNet(config['path'],
                 files=config['train_file'],
                 rand=rand,
                 batch_size=config['batch_size'],
                 grid_size=config['grid_size'],
                 levelset=[-0.00],
                 num_samples=config['num_samples'],
                 list_=config['categories'],
                 rec_mode=rec_mode)
#for ii in range(0,SN_train.train_size):
#    batch = SN_train.get_batch(type_=type_)
#    print(str(SN_train.train_step)+' /'+str(SN_train.train_size))

SN_test        = ShapeNet(config['path'],
                 files=config['test_file'],
                 rand=False,
                 batch_size=config['batch_size'],
                 grid_size=config['grid_size'],
                 levelset=[-0.00],
                 num_samples=config['num_samples'],
                 list_=config['categories'],
                 rec_mode=False)


    

#batch = SN_train.get_batch(type_=type_)
#size_ = SN_train.train_size
#psudo_sdf = batch['sdf'][0,:,:,:]
#verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
#cubed0 = {'vertices':verts0/(grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(grid_size-1)*2-1}
#MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')    


#vertices             = np.concatenate((batch['vertices'][:,:,:,0],batch['vertices'][:,:,:,1]),axis=1)/(grid_size-1)*2-1
#gaussian_noise       = np.random.normal(loc=0.0,scale=0.1,size=vertices.shape).astype(np.float32)
#vertices             = np.clip((vertices+gaussian_noise),-1.0,1.0)
#cubed = {'vertices':vertices[0,:,:],'faces':faces0,'vertices_up':vertices[0,:,:]}
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud_up')  
     
#pic = batch['images'][0,:,:,:]
#fig = plt.figure()
#plt.imshow(pic/255.)







x           = np.linspace(-1, 1, config['grid_size'])
y           = np.linspace(-1, 1, config['grid_size'])
z           = np.linspace(-1, 1, config['grid_size'])
xx,yy,zz    = np.meshgrid(x, y, z)

grid_size_lr   = config['grid_size']*2
x_lr           = np.linspace(-1, 1, grid_size_lr)
y_lr           = np.linspace(-1, 1, grid_size_lr)
z_lr           = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x_lr, y_lr, z_lr)




#%% Function wrappers   
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')
    
   
def function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_sdf2(coordinates,args_[0],args_[1])
        return evaluated_function


def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)


def mpx_function_wrapper(encoding,args_):
    with tf.variable_scope('multiplexer_model',reuse=tf.AUTO_REUSE):
        return CNN.multiplexer(encoding,args_[0])



with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
    Ray_render = RAY.Raycast(config['batch_size'],resolution=(137,137),sky_color=(127.5, 127.5, 127.5))
    Ray_render.add_lighting(position=[0.0, 0.0, 1.0],color=[255., 120., 20.],ambient_color = [255., 255., 255.])
    Ray_render.add_camera(camera_position=[1.0, 1.0, 1.0], lookAt=(0,0,0),focal_length=2,name='camera_1')
Ray_render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'display_tracer')





#%% Sampling in XYZ domain  
images                = tf.placeholder(tf.float32,shape=(None,137,137,3), name='images')  
samples_sdf           = tf.placeholder(tf.float32,shape=(None,None,1), name='samples_sdf')  
samples_xyz           = tf.placeholder(tf.float32,shape=(None,None,3),   name='samples_xyz')  
evals_target          = {}
evals_target['x']     = samples_xyz
evals_target['y']     = samples_sdf
evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
embeddings            = CNN_function_wrapper(images,[mode_node,config])
evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,embeddings],shape = [config['batch_size'],config['num_samples']],samples=evals_target['x'] , use_samps=True)


labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(config['batch_size'],-1)),config['levelset']),tf.int64)
labels_float       = tf.cast(labels,tf.float32)
logits             = tf.reshape(evals_function['y'],(config['batch_size'],-1,1)) #- levelset
logits             = tf.concat((logits,-logits),axis=-1)
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_y            = tf.square(evals_function['y']-evals_target['y'])
norm               = tf.reduce_max(tf.abs(evals_function['dydx_norm']))
norm_loss          = tf.reduce_mean((evals_function['dydx_norm'] - 1.0)**2)
radius             = 0.1
sample_w           = tf.squeeze(tf.exp(-(evals_target['y']-config['levelset'])**2/radius),axis=-1)
loss_class         = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'),axis=-1)
#loss_class         = loss_class/tf.reduce_mean(sample_w,axis=-1)
iou_logit          = logits[:,:,1:2]
iou_target         = tf.expand_dims(labels_float,-1)
#loss_y             = tf.reduce_mean(delta_y) 
loss               = tf.reduce_mean(loss_class)
#loss               = IOU.lovasz_hinge(iou_logit, iou_target, per_image=False)
X                  = tf.cast(labels,tf.bool)
Y                  = tf.cast(tf.argmax(predictions, 2),tf.bool)
iou                = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1))





with tf.variable_scope('optimization_cnn',reuse=tf.AUTO_REUSE):
    cnn_vars      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = '2d_cnn_model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node,beta1=0.9,beta2=0.999)
#    optimizer     = tf.train.MomentumOptimizer(lr_node, momentum=0.9)     
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
    grid_size_lr   = config['grid_size']*2
    x_lr           = np.linspace(-1, 1, grid_size_lr)
    y_lr           = np.linspace(-1, 1, grid_size_lr)
    z_lr           = np.linspace(-1, 1, grid_size_lr)
    xx_lr,yy_lr,zz_lr    = np.meshgrid(x_lr, y_lr, z_lr)    
    step_test      = 0
    aa_mov_test    = MOV_AVG(3000000) 
    dd_mov_test    = MOV_AVG(3000000) 
    samples_xyz_np = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
    samples_ijk_np = np.round(((samples_xyz_np+1)/2*(config['grid_size']-1))).astype(np.int32)
    while SN_test.epoch<25:
        batch                = SN_test.get_batch(type_='')
        samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
        feed_dict = {images             :batch['images']/255.,
                     samples_xyz        :np.tile(samples_xyz_np,[config['batch_size'],1,1]),
                     samples_sdf        :samples_sdf_np}     
        accuracy_t ,iou_t = session.run([accuracy, iou],feed_dict=feed_dict) 
        aa_mov_avg_test = aa_mov_test.push(accuracy_t)
        dd_mov_avg_test = dd_mov_test.push(iou_t)
        step_test+=1
        print('epoch: '+str(SN_test.epoch)+' step: '+str(step_test)+' ,avg_accuracy: '+str(aa_mov_avg_test)+' ,IOU: '+str(dd_mov_avg_test))
    return aa_mov_avg_test, dd_mov_avg_test



    
session = tf.Session()
session.run(tf.initialize_all_variables())
if config['finetune']:
    loader.restore(session, config['saved_model_path'])
step   = 0
aa_mov = MOV_AVG(300) 
bb_mov = MOV_AVG(300) 
cc_mov = MOV_AVG(300) 
dd_mov = MOV_AVG(300) 
loss_plot = []
acc_plot  = []
iou_plot  = []
acc_plot_test  = []
iou_plot_test  = []

while step < 100000000:
    session.run(mode_node.assign(True)) 
    batch                = SN_train.get_batch(type_='')
    batch_feed           = SN_train.process_batch(batch)
    feed_dict = {images             :batch['images']/255.,
                 lr_node            :config['learning_rate'],
                 samples_xyz        :batch_feed['samples_xyz_np'],
                 samples_sdf        :batch_feed['samples_sdf_np'],
                 }     
    _, loss_,norm_, accuracy_ ,iou_ = session.run([train_op_cnn, loss, norm ,accuracy, iou],feed_dict=feed_dict) 

    aa_mov_avg = aa_mov.push(accuracy_)
    cc_mov_avg = cc_mov.push(loss_)
    dd_mov_avg = dd_mov.push(iou_)
    print('epoch: '+str(SN_train.epoch)+' step: '+str(step)+' ,avg_accuracy: '+str(aa_mov_avg)+' ,avg_loss: '+str(cc_mov_avg)+' ,IOU: '+str(dd_mov_avg))

    if step % config['checkpoint_every'] == 0 and step!=0:
        saver.save(session, config['checkpoint_path'], global_step=step)
        last_saved_step = step
    if step % config['plot_every'] == 0:
        acc_plot.append(np.expand_dims(np.array(aa_mov_avg),axis=-1))
        loss_plot.append(np.expand_dims(np.array(np.log(cc_mov_avg)),axis=-1))
        iou_plot.append(np.expand_dims(np.array(dd_mov_avg),axis=-1))
        plt.figure(1)
        plt.plot(acc_plot)
        plt.title('accuracy')
        plt.pause(0.05)
        plt.figure(2)
        plt.plot(loss_plot)        
        plt.title('loss')
        plt.pause(0.05)
        np.save(config['checkpoint_path']+'loss_values.npy',np.concatenate(loss_plot))
        np.save(config['checkpoint_path']+'accuracy_values.npy',np.concatenate(acc_plot))  
        np.save(config['checkpoint_path']+'iou_values.npy',np.concatenate(iou_plot))  
    if step % config['test_every'] == 0:
        acc_test, iou_test = evaluate(SN_test, mode_node, config, accuracy, iou)
        acc_plot_test.append(np.expand_dims(np.array(acc_test),axis=-1))
        iou_plot_test.append(np.expand_dims(np.array(iou_test),axis=-1))
        np.save(config['checkpoint_path']+'accuracy_values_test.npy',np.concatenate(acc_plot_test))  
        np.save(config['checkpoint_path']+'iou_values_test.npy',np.concatenate(iou_plot_test))         
        
    step+=1




#%% EVAL
#session.run(mode_node.assign(False)) 
#example=0
#samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(1,1,1))
#samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config['grid_size']-1))).astype(np.int32)
#batch                = SN_train.get_batch(type_='')
#samples_sdf_np       = np.expand_dims(batch['sdf'][example:example+1,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)    
#feed_dict = {images           :batch['images'][example:example+1,:,:,:]/255.,
#             samples_xyz      :samples_xyz_np,
#             samples_sdf      :samples_sdf_np}
#evals_function_d,accuracy_   = session.run([evals_function['y'],accuracy],feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
#field              = np.reshape(evals_function_d[0,:,:],(-1,))
#field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
#if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
#    verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
#    cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
#    verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][example,:,:,:], config['levelset'])
#    cubed = {'vertices':verts/(config['grid_size']-1)*2-1,'faces':faces}
#    MESHPLOT.double_mesh_plot([cubed_plot,cubed],idx=0,type_='cubed')












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

