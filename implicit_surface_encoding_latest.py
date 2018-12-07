

import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
#from src.models import signed_dist_functions as SDF
from src.models import scalar_functions as SF
from src.models import voxels as CNN
#from src.utilities import Render as RENDER
#import provider
#import matplotlib.pyplot as plt
#from src.utilities import Voxels as VOXELS
from skimage import measure
from provider_binvox import ShapeNet as ShapeNet 



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
levelset    = 0.0
BATCH_SIZE  = 1
num_samples = 1000
type_ = 'debug'
list_ = ['02691156']

#%%
path = '/Users/gidilittwin/Dropbox/Thesis/Implicit_Encoding/Data/ShapeNetVox32/'
#path = '/media/gidi/SSD/DropboxContainer/Dropbox/Thesis/Implicit_Encoding/Data/ShapeNetVox32/'

SN = ShapeNet(path,rand=True,batch_size=BATCH_SIZE,grid_size=grid_size,levelset=levelset,list_=list_)
batch = SN.get_batch(type_=type_)
size_ = SN.debug_size




verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][0,:,:,:], levelset)
cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts/grid_size*2-1}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud_up')
#print(batch['code'])


x           = np.linspace(-1, 1, grid_size)
y           = np.linspace(-1, 1, grid_size)
z           = np.linspace(-1, 1, grid_size)
xx,yy,zz    = np.meshgrid(x, y, z)







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


def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.resnet(image,args_[0],args_[1])



def mpx_function_wrapper(encoding,args_):
    with tf.variable_scope('multiplexer_model',reuse=tf.AUTO_REUSE):
        return CNN.multiplexer(encoding,args_[0])






#%% Sampling in XYZ domain  
images                = tf.placeholder(tf.float32,shape=(BATCH_SIZE,200,200,3), name='images')  
encoding              = tf.placeholder(tf.float32,shape=(BATCH_SIZE,size_), name='encoding')  
samples_sdf           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,1), name='samples_sdf')  
samples_xyz           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_xyz')  



evals_target          = {}
evals_target['x']     = samples_xyz
reflect               = tf.constant([[[-1.0,1.0,1.0]]])
evals_target['-x']    = evals_target['x']*reflect
evals_target['y']     = samples_sdf
evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)


#embeddings   = mpx_function_wrapper(encoding,[mode_node])

in_size      = 3 #+ embeddings.get_shape().as_list()[1]
theta        = []
theta.append({'w':32})
theta.append({'w':32})
theta.append({'w':32})
theta.append({'w':32})
theta.append({'w':32})
#theta.append({'w':128})
theta.append({'w':1})


#theta                 = CNN_function_wrapper(images,[mode_node,layers])
evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['x'] , use_samps=True)
evals_function_r      = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['-x'] , use_samps=True)
evals_function['y']   = (evals_function['y']+evals_function_r['y'])/2


labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(BATCH_SIZE,-1)),levelset),tf.int64)
labels_float       = tf.cast(labels,tf.float32)
#num_labels_pos     = 2000.
#num_labels_neg     = (32.**3-num_labels_pos)
#num_total          = num_labels_pos + num_labels_neg
#weights            = 0.5*(labels_float/num_labels_pos + (1.-labels_float)/num_labels_neg)*num_total
weights            = labels_float*0.5 + (1.-labels_float)*0.5
logits             = tf.reshape(evals_function['y'],(BATCH_SIZE,-1,1)) #- levelset
logits             = tf.concat((-logits,logits),axis=-1)
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_y            = tf.square(evals_function['y']-evals_target['y'])

#sample_w         = tf.squeeze(tf.exp(-evals_target['y']**2/0.2),axis=-1)
loss_class       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'))
loss_y           = tf.reduce_mean(delta_y) 
loss             = loss_class







with tf.variable_scope('optimization',reuse=tf.AUTO_REUSE):
    dummy_loss    = tf.reduce_mean(tf.constant(np.zeros([0],dtype=np.float32)))
    loss_check    = tf.is_nan(loss)
    loss          = tf.cond(loss_check, lambda:dummy_loss, lambda:loss)
    model_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node)
#    optimizer     = tf.train.MomentumOptimizer(lr_node, momentum=0.9) 
    grads         = optimizer.compute_gradients(loss,var_list=model_vars)
    global_step   = tf.train.get_or_create_global_step()
    clip_constant = 10
    g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
    train_op_net  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)



#%% Train
session = tf.Session()
session.run(tf.initialize_all_variables())
step = 0
aa_mov = MOV_AVG(300) 
bb_mov = MOV_AVG(300) 
session.run(mode_node.assign(True)) 
for step in range(10000000):
    samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(1,num_samples,3))
    batch                = SN.get_batch(type_=type_)
    samples_ijk_np       = ((samples_xyz_np+1)/2*(grid_size-1)).astype(np.int32)
    samples_sdf_np       = np.expand_dims(batch['sdf'][:,samples_ijk_np[0,:,1],samples_ijk_np[0,:,0],samples_ijk_np[0,:,2]],-1)
    

    feed_dict = {lr_node            :0.001, 
                 encoding           :batch['code'], 
                 samples_xyz        :np.tile(samples_xyz_np,(BATCH_SIZE,1,1)),
                 samples_sdf        :samples_sdf_np}  
              
    _, loss_class_,loss_y_, accuracy_  = session.run([train_op_net, loss_class, loss_y,accuracy ],feed_dict=feed_dict) 
    aa_mov_avg = aa_mov.push(accuracy_)
    bb_mov_avg = bb_mov.push(loss_y_)
    print('step: '+str(step)+' ,accuracy: '+str(aa_mov_avg)+' ,sdf: '+str(bb_mov_avg))




if step % 50==49:
    session.run(mode_node.assign(True)) 
    samples_xyz_np = np.tile(np.reshape(np.stack((xx,yy,zz),axis=-1),(1,-1,3)),(BATCH_SIZE,1,1))
    feed_dict = {encoding         :batch['code'], 
                 samples_xyz      :samples_xyz_np}
    evals_function_d   = session.run(evals_function,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
    #point_cloud        = np.reshape(evals_function_d['x'],(-1,3))
    
    example            = np.random.randint(BATCH_SIZE)
    field              = np.reshape(evals_function_d['y'][example,:,:],(-1,))
    field              = np.reshape(field,(grid_size,grid_size,grid_size,1))
    
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts/100*2-1}
#            MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='cubed')

        verts, faces, normals, values = measure.marching_cubes_lewiner(batch['sdf'][example,:,:,:], levelset)
        cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts/grid_size*2-1}
        MESHPLOT.double_mesh_plot([cubed_plot,cubed],idx=0,type_='cubed')



#%%


#ray_steps,reset_opp,evals = Ray_render.eval_func(model_fn = target_wrapper,args=args_,step_size=0.004, epsilon = 0.001, temp=1.)
#session.run(reset_opp)
#step = 0
#while step < 50:
#    session.run(ray_steps)
#    step += 1
#    print(step)
#    
#evals_ = session.run(evals) # <= returns jpeg data you can write to disk
#for i in range(len(evals_['renders'])):    
#    with open('/Users/gidilittwin/Desktop/Renders/image_target'+str(step)+'_'+str(i)+'.jpg', 'w') as fd:
#        fd.write(evals_['renders'][i])
#images_ = np.stack(evals_['images'],axis=0).astype(np.float32)/255 



#evals_target_high  = SF.sample_points(model_fn = target_wrapper  ,args=args_            ,shape = [BATCH_SIZE,1000,1000])
#evals_target_high_ = session.run(evals_target_high)
#point_cloud        = np.reshape(evals_target_high_['x'],(-1,3))
#field              = np.reshape(evals_target_high_['y'],(-1,1))
#point_cloud        = point_cloud[field[:,0]<0,:]
#point_cloud_size   = point_cloud.shape[0]
#VOX                = VOXELS.Voxels(canvas_size=128,grid_size = 128, batch_size=1, num_points = point_cloud_size)
#voxels             = VOX.voxelize(tf.expand_dims(point_cloud,axis=0),field)
#voxels_            = session.run(voxels)
#verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_[0,:,:,:,0], 0)
#cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts}
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')





#ray_steps,reset_opp,evals = Ray_render.eval_func(model_fn = function_wrapper, args=[mode_node,theta],step_size=0.004, epsilon = 0.0001, temp=1.)
#session.run(reset_opp)
#step = 0
#while step < 50:
#    session.run(ray_steps,feed_dict=feed_dict)
#    step += 1
#    print(step)
#evals_ = session.run(evals,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
#
#for i in range(len(evals_['renders'])):    
#    with open('/Users/gidilittwin/Desktop/Renders/image'+str(step)+'_'+str(i)+'.jpg', 'w') as fd:
#        fd.write(evals_['renders'][i])
            

#evals_function_high     = SF.sample_points(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,1000,1000],samples=evals_target_high['x'],use_samps=True)
#evals_function_high_ = session.run(evals_function_high,feed_dict=feed_dict)
#point_cloud        = np.reshape(evals_function_high_['x'],(-1,3))
#field              = np.reshape(evals_function_high_['y'],(-1,))
#point_cloud        = point_cloud[field<0,:]
#point_cloud_size = point_cloud.shape[0]
#VOX         = VOXELS.Voxels(canvas_size=128,grid_size = grid_size, batch_size=1, num_points = point_cloud_size)
#voxels      = VOX.voxelize(tf.expand_dims(point_cloud,axis=0))
#voxels_ = session.run(voxels)
#verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_[0,:,:,:,0], 0)
#cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts}
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')




