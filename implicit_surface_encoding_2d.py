

import tensorflow as tf
import numpy as np
from src.models import scalar_functions as SF
#from src.models import voxels as CNN
BATCH_SIZE = 1
from skimage import measure
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator


grid_size   = 128
batch_size  = 16
canvas_size = grid_size
num_points  = 100000



#%% MODELNET ITERATOR

# SIGNED DISTANCE FIELD
x                  = np.linspace(-1, 1, grid_size)
y                  = np.linspace(-1, 1, grid_size)

voxels_     = np.zeros((grid_size,grid_size),dtype=np.float32)
xx,yy       = np.meshgrid(x, y)

voxels_[(xx+0.7)**2+(yy+0.7)**2<0.1]=1
voxels_[np.all(np.stack((np.abs(xx)<0.1,np.abs(yy)<0.1),axis=-1),axis=-1)]=1
voxels_[np.all(np.stack((np.abs(xx-0.6)<0.1,np.abs(yy+0.5)<0.01),axis=-1),axis=-1)]=1


voxels_     = np.expand_dims(np.expand_dims(ndi.distance_transform_edt(voxels_),axis=0),-1)
voxels_     = ndi.morphology.binary_fill_holes(voxels_)


inner_volume       = (voxels_[0,:,:,0]).astype(np.bool)
outer_volume       = (1.-voxels_[0,:,:,0]).astype(np.bool)
sdf_o, closest_point_o = ndi.distance_transform_edt(outer_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
sdf_i, closest_point_i = ndi.distance_transform_edt(inner_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)

closest_point = np.transpose(closest_point_o,(1,2,0)).astype(np.float32)/(grid_size-1)*2-1
#closest_point_i = np.transpose(closest_point_i,(1,2,0)).astype(np.float32)/(grid_size-1)*2-1
#mask            = np.stack((outer_volume,outer_volume),axis=-1)
#closest_point   = np.where(mask,closest_point_o,closest_point_i)



#sdf                = (sdf_o - sdf_i).astype(np.float32)/(grid_size-1)*2
sdf                = (sdf_o/(grid_size-1)*2).astype(np.float32)
grid               = np.stack(np.meshgrid(x,y),axis=-1)
closest_point_vec  = grid - closest_point 
sdf_grads          = closest_point_vec/np.sqrt(np.sum(closest_point_vec**2,axis=-1,keepdims=True))
#sdf_grads          = sdf_grads*np.expand_dims(outer_volume,-1)

interp_sdf         = RegularGridInterpolator((x,y), sdf)
interp_sdf_grads   = RegularGridInterpolator((x,y), sdf_grads)
interp_sdf_cp      = RegularGridInterpolator((x,y), closest_point)
bound_value        = 0.0




import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fig=plt.figure()
ax1 = fig.add_subplot(111)
lvls = np.linspace(-1,1,20)
CF = ax1.contourf(x,y,sdf,
         norm = LogNorm(),
         levels = lvls
        )
CS = ax1.contour(x,y,sdf,
         norm = LogNorm(),
         colors = 'k',
         levels = lvls
        )

Q = ax1.quiver(x, y, sdf_grads[:,:,0], sdf_grads[:,:,1], units='xy',scale_units='xy')
#qk = plt.quiverkey(Q, 5, 5, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
cbar = plt.colorbar(CF, ticks=lvls, format='%.4f')
plt.show()



cp = closest_point - np.min(closest_point)
cp = cp/np.max(cp)
zeros=np.zeros((128,128,1),dtype=np.float32)
cp = np.concatenate((cp,zeros),axis=-1)
fig=plt.figure()
plt.imshow(cp)




#%% Function wrappers   
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')
    
   
def function_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function,tensor = SF.deep_sdf3(coordinates,args_[0],args_[1])
        return evaluated_function,tensor


def CNN_function_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.resnet(image,args_[0],args_[1])
layers = [3,16,1]
#layers = [3,16,1]











#%% Sampling in XYZ domain  
samples_xyz           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,2),   name='samples_xyz')  
samples_sdf           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,1), name='samples_sdf')  
samples_sdf_grads     = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,2),   name='samples_sdf_grads')  
samples_sdf_cp        = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,2),   name='samples_sdf_cp')  


evals_target          = {}
evals_target['x']     = samples_xyz
evals_target['y']     = samples_sdf
evals_target['dydx']  = samples_sdf_grads
evals_target['cp']    = samples_sdf_cp


#theta                 = CNN_function_wrapper(images,[mode_node,layers])
theta = []
theta.append({'w':2,'b':16})
theta.append({'w':16,'b':16})
#theta.append({'w':32,'b':32})
theta.append({'w':16,'b':2})
evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['x'] , use_samps=True)




level_set          = tf.random_uniform([1],
                                        minval=0.0,
                                        maxval=1.0,
                                        dtype=tf.float32)
level_set = 0.0
labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(1,-1)),level_set),tf.int64)
logits             = tf.reshape(evals_function['y'],(1,-1,1))
logits             = tf.concat((logits,-logits),axis=-1) - level_set
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_sdf          = tf.square(evals_function['y']-evals_target['y'])
delta_cp           = tf.square(evals_function['cp']-evals_target['cp'])




delta_dydx         = tf.reduce_sum(tf.square(evals_function['dydx']  -evals_target['dydx']),axis=-1,keep_dims=True)
delta_dydx_n       = tf.reduce_sum(tf.square(evals_function['dydx_norm']-1.),axis=-1,keep_dims=True)

#weights          = tf.exp(-1*evals_target['y']**2/1.0)
loss_class       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'))
loss_sdf         = tf.reduce_mean(delta_sdf) 
loss_cp          = tf.reduce_mean(delta_cp) 

#loss_y_div1      = tf.reduce_mean(delta_dydx) 
loss_y_div1_n    = tf.reduce_mean(delta_dydx_n) 
loss             = 1.0*loss_class   #+0.1*loss_y_div1_n






with tf.variable_scope('optimization',reuse=tf.AUTO_REUSE):
    dummy_loss    = tf.reduce_mean(tf.constant(np.zeros([0],dtype=np.float32)))
    loss_check    = tf.is_nan(loss)
    loss          = tf.cond(loss_check, lambda:dummy_loss, lambda:loss)
    model_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node, beta1=0.5)
    #optimizer = tf.train.MomentumOptimizer(lr_node, momentum=0.9) 
    grads         = optimizer.compute_gradients(loss,var_list=model_vars)
    global_step   = tf.train.get_or_create_global_step()
    clip_constant = 0.1
    g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
    train_op_net  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)



#%% Train

session = tf.Session()
session.run(tf.initialize_all_variables())
step = 0


session.run(mode_node.assign(True)) 
for step in range(1000000):
    samples_xyz_np = np.random.uniform(low=-1.,high=1.,size=(BATCH_SIZE,10000,2))
    samples_ijk_np = samples_xyz_np[:,:,[1,0]]
    samples_sdf_np = np.expand_dims(interp_sdf(samples_ijk_np),axis=-1)
    samples_sdf_grads_np = interp_sdf_grads(samples_ijk_np)
    samples_sdf_cp_np = interp_sdf_cp(samples_xyz_np)

    feed_dict = {lr_node            :0.00001, 
                 samples_xyz        :samples_xyz_np,
                 samples_sdf        :samples_sdf_np,
                 samples_sdf_grads  :samples_sdf_grads_np,
                 samples_sdf_cp     :samples_sdf_cp_np,
                 }                
    _, loss_, loss_sdf_,loss_y_div1_n_, accuracy_  = session.run([train_op_net, loss, loss_sdf, loss_y_div1_n, accuracy ],feed_dict=feed_dict) 
    print('step: '+str(step)+' ,sdf: '+str(loss_sdf_)+' ,cp: '+str(loss_)+' ,accuracy: '+str(accuracy_))










#%% Test



samples_xyz_np = np.reshape(np.stack((xx,yy),axis=-1),(1,-1,2))
samples_ijk_np = samples_xyz_np[:,:,[1,0]]
samples_sdf_np = np.expand_dims(interp_sdf(samples_ijk_np),axis=-1)
samples_sdf_grads_np = interp_sdf_grads(samples_ijk_np)

feed_dict = {samples_xyz        :samples_xyz_np,
             samples_sdf        :samples_sdf_np,
             samples_sdf_grads  :samples_sdf_grads_np}                
sdf_ ,sdf_grad_, cp_, class_= session.run([evals_function['y'],evals_function['dydx'],evals_function['cp'], predictions],feed_dict=feed_dict) 

sdf_      = np.reshape(sdf_,(128,128))
sdf_grad_ = np.reshape(sdf_grad_,(128,128,2))
sdf_grad_n= np.sqrt(np.sum(sdf_grad_**2,axis=-1))
class_  = np.reshape(class_,(128,128,2))[:,:,0]

cp_ = np.reshape(cp_,(128,128,2))


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fig=plt.figure()
ax1 = fig.add_subplot(3,1,1)
lvls = np.linspace(-1,1,20)
CF = ax1.contourf(x,y,sdf_,
         norm = LogNorm(),
         levels = lvls
        )
CS = ax1.contour(x,y,sdf_,
         norm = LogNorm(),
         colors = 'k',
         levels = lvls
        )

ax2 = fig.add_subplot(3,1,2)
CF = ax2.contourf(x,y,sdf,
         norm = LogNorm(),
         levels = lvls
        )
CS = ax2.contour(x,y,sdf,
         norm = LogNorm(),
         colors = 'k',
         levels = lvls
        )

ax2 = fig.add_subplot(3,1,2)
CF = ax2.contourf(x,y,sdf,
         norm = LogNorm(),
         levels = lvls
        )
CS = ax2.contour(x,y,sdf,
         norm = LogNorm(),
         colors = 'k',
         levels = lvls
        )

lvls = np.linspace(0,1,30)

ax3 = fig.add_subplot(3,1,3)
CF = ax3.contourf(x,y,class_,
         norm = LogNorm(),
         levels = lvls
        )
CS = ax3.contour(x,y,class_,
         norm = LogNorm(),
         colors = 'k',
         levels = lvls
        )

cbar = plt.colorbar(CF, ticks=lvls, format='%.4f')
plt.show()







