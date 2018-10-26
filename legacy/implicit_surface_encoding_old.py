

import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
from src.utilities import raytrace as RAY
from src.models import signed_dist_functions as SDF
from src.models import scalar_functions as SF
from src.models import voxels as CNN
from src.utilities import Render as RENDER
import provider
import matplotlib.pyplot as plt
BATCH_SIZE = 1
from src.utilities import Voxels as VOXELS
from skimage import measure
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator


grid_size   = 32
batch_size  = 16
canvas_size = grid_size
num_points  = 100000



#%% MODELNET ITERATOR
ITERATOR = provider.ModelNet10('/Users/gidilittwin/Dropbox/Thesis/ModelNet/Data/ModelNet10/',True,batch_size=16,num_points=10000,multipier=10)
batch    = ITERATOR.get_batch('train')
batch_tf = ITERATOR.convert2tf(batch)

mesh     = batch[0]
mesh_tf  = batch_tf[0]
#DR       = RENDER.Rasterizer(canvas_size=200)
#scale    = tf.Variable([[0.0]], name='scale_var')
#trans    = tf.Variable([[0.0]], name='trans_var')
#rot      = tf.Variable([[0.,1.,0.,0.6]], name='rot_var')   
#mesh_tf  = DR.augment_data_var(mesh_tf,{'scale':scale,'trans':trans ,'rot':rot})
#image_tf = DR.rasterize_tf(mesh_tf)
  

sess_test = tf.Session()
sess_test.run(tf.initialize_all_variables())
#image_    = sess_test.run(image_tf)
#imgplot   = plt.imshow(image_[:,:,0])
#image_ = np.random.uniform(-1.,1.,(200,200,1)).astype(np.float32)


# SIGNED DISTANCE FIELD
x                  = np.linspace(-1, 1, grid_size)
y                  = np.linspace(-1, 1, grid_size)
z                  = np.linspace(-1, 1, grid_size)
point_cloud        = mesh['vertices_up'].astype(np.float32)
VOX                = VOXELS.Voxels(canvas_size=grid_size,grid_size = grid_size, batch_size=1, num_points = num_points)
voxels             = VOX.voxelize(tf.expand_dims(point_cloud,axis=0),None)
voxels_            = sess_test.run(voxels)
voxels_            = ndi.morphology.binary_fill_holes(voxels_)


#voxels_     = np.zeros((grid_size,grid_size,grid_size),dtype=np.float32)
#xx,yy,zz    = np.meshgrid(x, y, z)
##voxels_[xx**2+yy**2+zz**2<0.5]=1
#voxels_[np.all(np.stack((np.abs(xx)<0.5,np.abs(yy)<0.3,np.abs(zz)<0.5),axis=-1),axis=-1)]=1
#voxels_     = np.expand_dims(np.expand_dims(ndi.distance_transform_edt(voxels_),axis=0),-1)
#voxels_     = ndi.morphology.binary_fill_holes(voxels_)


inner_volume       = (voxels_[0,:,:,:,0]).astype(np.bool)
outer_volume       = (1.-voxels_[0,:,:,:,0]).astype(np.bool)
sdf_o, closest_point_o = ndi.distance_transform_edt(outer_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)
sdf_i, closest_point_i = ndi.distance_transform_edt(inner_volume, return_indices=True) #- ndi.distance_transform_edt(inner_volume)


closest_point_o      = np.transpose(closest_point_o,(1,2,3,0)).astype(np.float32)/(grid_size-1)*2-1
closest_point_i      = np.transpose(closest_point_i,(1,2,3,0)).astype(np.float32)/(grid_size-1)*2-1
mask                 = np.stack((outer_volume,outer_volume,outer_volume),axis=-1)
closest_point        = np.where(mask,closest_point_o,closest_point_i)

#sdf                = (sdf_o - sdf_i)/(grid_size-1)*2
sdf                = sdf_o/(grid_size-1)*2
grid               = np.stack(np.meshgrid(x,y,z),axis=-1)
closest_point_vec  = grid - closest_point 
sdf_grads          = closest_point_vec/np.sqrt(np.sum(closest_point_vec**2,axis=-1,keepdims=True))
sdf_grads          = sdf_grads*np.expand_dims(outer_volume,-1)

interp_sdf         = RegularGridInterpolator((x, y, z), sdf)
interp_sdf_grads   = RegularGridInterpolator((x, y, z), sdf_grads)
interp_sdf_cp      = RegularGridInterpolator((x, y, z), closest_point)

bound_value        = 0.00


#MESHPLOT.mesh_plot([mesh],idx=0,type_='cloud_up')
#MESHPLOT.mesh_plot([mesh],idx=0,type_='mesh')

verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, bound_value)
cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts/grid_size*2-1}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')
#MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud_up')
#MESHPLOT.mesh_plot([mesh],idx=0,type_='cloud_up')



#%% Initiate renderers 
#with tf.variable_scope('train_tracer',reuse=tf.AUTO_REUSE):
#    Ray = RAY.Raycast(BATCH_SIZE,resolution=(100,100))
#    Ray.add_lighting(position=[0., 1., 1.],color=[255., 255., 255.],ambient_color = [244., 83., 66.])
#    Ray.add_camera(camera_position='random', lookAt=(0,0,0),focal_length=1,name='camera_1')
#with tf.variable_scope('image_tracer',reuse=tf.AUTO_REUSE):
#    Ray_image = RAY.Raycast(BATCH_SIZE,resolution=(200,200))
#    Ray_image.add_lighting(position=[0., 1., 1.],color=[255., 255., 255.],ambient_color = [244., 83., 66.])
#    Ray_image.add_camera(camera_position=[0.0, 0.9, 0.0], lookAt=(0,0,0),focal_length=1,name='camera_1')
#with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
#    Ray_render = RAY.Raycast(BATCH_SIZE,resolution=(600,800))
#    Ray_render.add_lighting(position=[0., 1., 1.],color=[255., 255., 255.],ambient_color = [244., 83., 66.])
#    Ray_render.add_camera(camera_position=[0.0, 0.9, 0.0], lookAt=(0,0,0),focal_length=1,name='camera_1')
#Ray_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'train_tracer')
#Ray_image_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'image_tracer')
#Ray_render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'display_tracer')




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
layers = [3,8,1]
#layers = [3,16,1]











#Sampling in XYZ domain  
images                = tf.placeholder(tf.float32,shape=(BATCH_SIZE,200,200,3), name='images')  
samples_xyz           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_xyz')  
samples_sdf           = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,1), name='samples_sdf')  
samples_sdf_grads     = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_sdf_grads')  
samples_xyz_bound     = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_xyz_bound')  
samples_xyz_bound_n   = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_xyz_bound_normals')  
samples_sdf_cp        = tf.placeholder(tf.float32,shape=(BATCH_SIZE,None,3),   name='samples_sdf_cp')  


evals_target          = {}
evals_target['x']     = samples_xyz
evals_target['y']     = samples_sdf
evals_target['dydx']  = samples_sdf_grads
evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
evals_target['x_b']   = samples_xyz_bound
evals_target['x_b_n'] = samples_xyz_bound_n
evals_target['cp']    = samples_sdf_cp


#theta                 = CNN_function_wrapper(images,[mode_node,layers])
theta = []
theta.append({'w':3,'b':16})
theta.append({'w':16,'b':16})
theta.append({'w':16,'b':16})
theta.append({'w':16,'b':3})
evals_function        = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['x'] , use_samps=True)
#evals_function_bound  = SF.sample_points_list(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,100000],samples=evals_target['x_b'],use_samps=True)





labels             = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(1,-1)),0.0),tf.int64)
logits             = tf.reshape(evals_function['y'],(1,-1,1))
logits             = tf.concat((logits,-logits),axis=-1)
predictions        = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(predictions, 2), labels)
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err                = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
delta_y            = tf.square(evals_function['y']-evals_target['y'])
delta_dydx         = tf.reduce_sum(tf.square(evals_function['dydx']  -evals_target['dydx']),axis=-1,keep_dims=True)
delta_cp           = tf.square(evals_function['cp']-evals_target['cp'])
#delta_y_bound      = tf.square(evals_function_bound['y']-bound_value)
#delta_y_bound_n    = 1-tf.reduce_sum(evals_function_bound['dydx']*(-1*evals_target['x_b_n']),axis=-1,keep_dims=True)
delta_dydx_n       = tf.reduce_sum(tf.square(evals_function['dydx_norm']-1.),axis=-1,keep_dims=True)


#weights          = tf.exp(-1*evals_target['y']**2/1.0)
loss_class       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy'))
loss_y           = tf.reduce_mean(delta_y) 
loss_y_div1      = tf.reduce_mean(delta_dydx) 
loss_y_div1_n    = tf.reduce_mean(delta_dydx_n) 
loss_cp          = tf.reduce_mean(delta_cp) 
#loss_y_bounds    = tf.reduce_mean(delta_y_bound) 
#loss_y_bounds_n  = tf.reduce_mean(delta_y_bound_n) 
loss             = 1.0*loss_class   #+0.01*loss_y_div1_n
#loss             = 1.0*loss_y_bounds +1.0*loss_y

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
images_np = np.expand_dims(image_,0)
images_np = np.concatenate((images_np,images_np,images_np),axis=-1)
xx,yy,zz = np.meshgrid(x, y, z)
samples_xyz_np = np.reshape(np.stack((xx,yy,zz),axis=-1),(1,-1,3))
session.run(mode_node.assign(True)) 
for step in range(100000):
    samples_xyz_np       = np.random.uniform(low=-1.,high=1.,size=(BATCH_SIZE,10000,3))
    samples_ijk_np       = samples_xyz_np[:,:,[1,0,2]]
    samples_sdf_np       = np.expand_dims(interp_sdf(samples_ijk_np),axis=-1)
    samples_sdf_grads_np = interp_sdf_grads(samples_ijk_np)
    samples_sdf_cp_np    = interp_sdf_cp(samples_xyz_np)
    feed_dict = {lr_node            :0.001, 
                 images             :images_np, 
                 samples_xyz        :samples_xyz_np,
                 samples_sdf        :samples_sdf_np,
                 samples_sdf_grads  :samples_sdf_grads_np,
                 samples_sdf_cp     :samples_sdf_cp_np}  
              
    _, loss_, loss_y_, accuracy_  = session.run([train_op_net, loss, loss_y, accuracy ],feed_dict=feed_dict) 
    print('step: '+str(step)+' ,sdf: '+str(loss_y_)+' ,cp: '+str(loss_)+' ,accuracy: '+str(accuracy_))

#    evals_function_ = session.run(evals_function,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
#    evals_target_   = session.run(evals_target,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
#    evals_function_bound_   = session.run(evals_function_bound,feed_dict=feed_dict) # <= returns jpeg data you can write to disk



    if (step+15)%100==1:

        samples_xyz_np = np.reshape(np.stack((xx,yy,zz),axis=-1),(1,-1,3))
        samples_ijk_np = samples_xyz_np[:,:,[1,0,2]]
        samples_sdf_np = np.expand_dims(interp_sdf(samples_ijk_np),axis=-1)
        samples_sdf_grads_np = interp_sdf_grads(samples_ijk_np)
        samples_xyz_b_np = np.expand_dims(cubed['vertices'],axis=0)
        feed_dict = {images           :images_np, 
                     samples_xyz      :samples_xyz_np,
                     samples_sdf      :samples_sdf_np,
                     samples_sdf_grads:samples_sdf_grads_np,
                     samples_xyz_bound:samples_xyz_b_np}
        evals_function_d = session.run(evals_function,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
        point_cloud        = np.reshape(evals_function_d['x'],(-1,3))
        field              = np.reshape(evals_function_d['y'],(-1,))
        field = np.reshape(field,(grid_size,grid_size,grid_size,1))
        
        
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='cubed')
#        MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')




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



evals_target_high  = SF.sample_points(model_fn = target_wrapper  ,args=args_            ,shape = [BATCH_SIZE,1000,1000])
evals_target_high_ = session.run(evals_target_high)
point_cloud        = np.reshape(evals_target_high_['x'],(-1,3))
field              = np.reshape(evals_target_high_['y'],(-1,1))
point_cloud        = point_cloud[field[:,0]<0,:]
point_cloud_size   = point_cloud.shape[0]
VOX                = VOXELS.Voxels(canvas_size=128,grid_size = 128, batch_size=1, num_points = point_cloud_size)
voxels             = VOX.voxelize(tf.expand_dims(point_cloud,axis=0),field)
voxels_            = session.run(voxels)
verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_[0,:,:,:,0], 0)
cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')





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
            

evals_function_high     = SF.sample_points(model_fn = function_wrapper,args=[mode_node,theta],shape = [BATCH_SIZE,1000,1000],samples=evals_target_high['x'],use_samps=True)
evals_function_high_ = session.run(evals_function_high,feed_dict=feed_dict)
point_cloud        = np.reshape(evals_function_high_['x'],(-1,3))
field              = np.reshape(evals_function_high_['y'],(-1,))
point_cloud        = point_cloud[field<0,:]
point_cloud_size = point_cloud.shape[0]
VOX         = VOXELS.Voxels(canvas_size=128,grid_size = grid_size, batch_size=1, num_points = point_cloud_size)
voxels      = VOX.voxelize(tf.expand_dims(point_cloud,axis=0))
voxels_ = session.run(voxels)
verts, faces, normals, values = measure.marching_cubes_lewiner(voxels_[0,:,:,:,0], 0)
cubed = {'vertices':verts/grid_size*2-1,'faces':faces,'vertices_up':verts}
MESHPLOT.mesh_plot([cubed],idx=0,type_='cubed')








