

import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
#from src.models import signed_dist_functions as SDF
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
#import provider
import matplotlib.pyplot as plt
#from src.utilities import Voxels as VOXELS
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 
from src.utilities import raytrace as RAY



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



grid_size   = 36
canvas_size = grid_size
levelset    = 0.1
BATCH_SIZE  = 1
num_samples = 1000
type_ = ''
list_ = ['02691156']

#%%
path = '/Users/gidilittwin/Dropbox/Thesis/Implicit_Encoding/Data/ShapeNetRendering_test/'
#path = '/media/gidi/SSD/DropboxContainer/Dropbox/Thesis/Implicit_Encoding/Data/ShapeNetVox32/'

SN    = ShapeNet(path,rand=True,batch_size=BATCH_SIZE,grid_size=grid_size,levelset=levelset,list_=list_,type_=type_)
batch = SN.get_batch(type_=type_)
size_ = SN.train_size




verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][0,:,:,:], levelset)
cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts/grid_size*2-1}
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud_up')


x           = np.linspace(-1, 1, grid_size)
y           = np.linspace(-1, 1, grid_size)
z           = np.linspace(-1, 1, grid_size)
xx,yy,zz    = np.meshgrid(x, y, z)

grid_size_lr = grid_size
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

def conditioned_function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        samps = tf.shape(coordinates)[1,]
        embedding = tf.tile(tf.expand_dims(args_[-1],axis=1),(1,samps,1))
        conditioned_input = tf.concat((coordinates,embedding),axis=-1)
        evaluated_function = SF.deep_sdf2(conditioned_input,args_[0],args_[1])
        return evaluated_function

def style_function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        samps = tf.shape(coordinates)[1,]
        embedding = tf.tile(tf.expand_dims(args_[-1],axis=1),(1,samps,1))
        evaluated_function = SF.deep_sdf4(coordinates,embedding,args_[0],args_[1])
        return evaluated_function

def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.resnet_34(image,args_)



def mpx_function_wrapper(encoding,args_):
    with tf.variable_scope('multiplexer_model',reuse=tf.AUTO_REUSE):
        return CNN.multiplexer(encoding,args_[0])



with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
    Ray_render = RAY.Raycast(BATCH_SIZE,resolution=(137,137),sky_color=(127.5, 127.5, 127.5))
    Ray_render.add_lighting(position=[0.0, 0.0, 1.0],color=[255., 120., 20.],ambient_color = [255., 255., 255.])
    Ray_render.add_camera(camera_position=[1.0, 1.0, 1.0], lookAt=(0,0,0),focal_length=2,name='camera_1')
Ray_render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'display_tracer')





#%% Sampling in XYZ domain  
images                = tf.placeholder(tf.float32,shape=(BATCH_SIZE,137,137,3), name='images')  
encoding              = tf.placeholder(tf.float32,shape=(BATCH_SIZE,size_), name='encoding')  
samples_sdf           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,1), name='samples_sdf')  
samples_xyz           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_xyz')  

evals_target          = {}
evals_target['x']     = samples_xyz
reflect               = tf.constant([[[-1.0,1.0,1.0]]])
evals_target['-x']    = evals_target['x']*reflect
evals_target['y']     = samples_sdf
evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)


#embeddings   = CNN_function_wrapper(images,[mode_node,32])
#embeddings   = mpx_function_wrapper(encoding,[mode_node])


in_size      = 3 #+ embeddings.get_shape().as_list()[1]
theta        = []
theta.append({'w':16})
theta.append({'w':16})
theta.append({'w':16})
theta.append({'w':16})
#theta.append({'w':16})
#theta.append({'w':16})
#theta.append({'w':64})
#theta.append({'w':64})
#theta.append({'w':64})
theta.append({'w':1})


evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['x'] , use_samps=True)
evals_function_r      = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['-x'] , use_samps=True)
evals_function['y']   = (evals_function['y']+evals_function_r['y'])/2



labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(BATCH_SIZE,-1)),levelset),tf.int64)
labels_float       = tf.cast(labels,tf.float32)
weights            = labels_float*0.5 + (1.-labels_float)*0.5
logits             = tf.reshape(evals_function['y'],(BATCH_SIZE,-1,1)) #- levelset
logits             = tf.concat((logits,-logits),axis=-1)
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_y            = tf.square(evals_function['y']-evals_target['y'])
norm               = tf.reduce_mean(tf.abs(evals_function['dydx_norm']))

sample_w         = tf.squeeze(tf.exp(-(evals_target['y']-levelset)**2/0.01),axis=-1)
#sample_w         = tf.squeeze(tf.abs((evals_target['y']-levelset)*tf.exp(-(evals_target['y']-levelset)**2/0.01)),axis=-1)
loss_class       = tf.reduce_mean(sample_w*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'))
loss_y           = tf.reduce_mean(delta_y) 
loss             = loss_class







with tf.variable_scope('optimization',reuse=tf.AUTO_REUSE):
    dummy_loss    = tf.reduce_mean(tf.constant(np.zeros([0],dtype=np.float32)))
    loss_check    = tf.is_nan(loss)
    loss          = tf.cond(loss_check, lambda:dummy_loss, lambda:loss)
    model_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'multiplexer_model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node,beta1=0.9,beta2=0.999)
#    optimizer     = tf.train.AdamOptimizer(lr_node,beta1=0.9,beta2=0.999)
#    optimizer     = tf.train.MomentumOptimizer(lr_node, momentum=0.9) 
    grads         = optimizer.compute_gradients(loss,var_list=model_vars)
    global_step   = tf.train.get_or_create_global_step()
    clip_constant = 10
    g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
    train_op_net  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)



#%% Train
#test_image = np.random.randn(BATCH_SIZE,224,224,3).astype(np.float32)
    
session = tf.Session()
session.run(tf.initialize_all_variables())
step = 0
aa_mov = MOV_AVG(300) 
bb_mov = MOV_AVG(10) 
session.run(mode_node.assign(False)) 
for step in range(10000000):
    samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(1,num_samples,3))
    batch                = SN.get_batch(type_=type_)
    samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(grid_size-1))).astype(np.int32)
    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)

    feed_dict = {lr_node            :0.0001,
                 images             :batch['images']/255.,
                 encoding           :batch['code'], 
                 samples_xyz        :np.tile(samples_xyz_np,(BATCH_SIZE,1,1)),
                 samples_sdf        :samples_sdf_np}  
              
    _, loss_class_,norm_, accuracy_  = session.run([train_op_net, loss_class, norm,accuracy ],feed_dict=feed_dict) 
    aa_mov_avg = aa_mov.push(accuracy_)
    bb_mov_avg = bb_mov.push(norm_)
    print('step: '+str(step)+' ,accuracy: '+str(aa_mov_avg)+' ,norm: '+str(bb_mov_avg))




if step % 50==49:
    session.run(mode_node.assign(True)) 
    samples_xyz_np = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(BATCH_SIZE,1,1))
    feed_dict = {encoding         :batch['code'], 
                 images             :batch['images']/255.,
                 samples_xyz      :samples_xyz_np}
    evals_function_d   = session.run(evals_function,feed_dict=feed_dict) # <= returns jpeg data you can write to disk    
    example            = np.random.randint(BATCH_SIZE)
    field              = np.reshape(evals_function_d['y'][example,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.00)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
#        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='cubed')
        
        verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][example,:,:,:], levelset)
        cubed = {'vertices':verts/(grid_size-1)*2-1,'faces':faces}
        MESHPLOT.double_mesh_plot([cubed_plot,cubed],idx=0,type_='cubed')















#%% RAY-TRACE

ray_steps,reset_opp,evals  = Ray_render.eval_func(model_fn = function_wrapper,args=[mode_node,theta],
                                                 step_size=None, 
                                                 epsilon = 0.0001, 
                                                 temp=1., 
                                                 ambient_weight=0.3)



init_new_vars_op = tf.initialize_variables(Ray_render_vars)
session.run(init_new_vars_op)
Ray_render.trace(session,ray_steps,reset_opp,num_steps=50)



evals_             = session.run(evals) # <= returns jpeg data you can write to disk
target_render      = evals_['raw_images'][0]
target_normals     = (np.transpose(evals_['normals'][0],(1,2,0))+1.)/2
target_incident    = evals_['incidence_angles'][0]
target_dist        = evals_['distances'][0]


fig = plt.figure()
title('target_image')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_render.astype(np.uint8))

fig = plt.figure()
title('target_image_normals')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_normals)

fig = plt.figure()
title('target_dist')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_dist)

