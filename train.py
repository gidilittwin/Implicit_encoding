
import json
import tensorflow as tf
import numpy as np
from src.utilities import mesh_handler as MESHPLOT
#from src.utilities.depthestimate import tf_nndistance as CHAMFER
#from src.utilities.external import tf_nndistance as CHAMFER
#from src.utilities.nn_distance import tf_nndistance_cpu as CHAMFER
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN
import os
import argparse
import tfrecords_handler as TFH
import socket
from skimage import measure

# tar -czvf meta_functionals.tar.gz metafunctionals/ --exclude-vcs


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'test')
    parser.add_argument('--model_params_path', type=str, default= './archs/resnet_sdf.json')
    parser.add_argument('--padding', type=str, default= 'VALID')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--batch_size', type=int,  default=2)
    parser.add_argument('--beta1', type=float,  default=0.9)
    parser.add_argument('--dropout', type=float,  default=1.0)
    parser.add_argument('--stage', type=int,  default=0)
    parser.add_argument('--multi_image', type=int,  default=0)
    parser.add_argument('--multi_image_views', type=int,  default=24)
    parser.add_argument('--multi_image_pool', type=str,  default='max')
    parser.add_argument('--norm_loss_alpha', type=float,  default=0.0000)

    parser.add_argument('--surfaces', type=int,  default=0)
    parser.add_argument('--alpha', type=float,  default=0.003)
    parser.add_argument('--grid_size', type=int,  default=36)
    parser.add_argument('--grid_size_v', type=int,  default=256)
    parser.add_argument('--compression', type=int,  default=0)
    parser.add_argument('--pretrained', type=int,  default=0)
    
    parser.add_argument('--embedding_size', type=int,  default=256)
    parser.add_argument('--num_blocks', type=int,  default=4)
    parser.add_argument('--block_width', type=int,  default=512)
    parser.add_argument('--bottleneck', type=int,  default=512)
    
    parser.add_argument('--img_size', type=int,  default=[137,137])
    parser.add_argument('--im_per_obj', type=int,  default=24)
    parser.add_argument('--test_size', type=int,  default=24)
    parser.add_argument('--shuffle_size', type=int,  default=1000)  
    parser.add_argument('--test_every', type=int,  default=10000)    
    parser.add_argument('--save_every', type=int,  default=1000) 
    parser.add_argument("--postfix_load"   , type=str, default="")
    parser.add_argument('--fast_eval', type=int,  default=1)    

    parser.add_argument('--eval_grid_scale', type=int,  default=1)
    parser.add_argument('--batch_norm', type=int,  default=0)
    parser.add_argument('--bn_l0', type=int,  default=0)
    parser.add_argument('--augment', type=int,  default=1)
    parser.add_argument('--rgba', type=int,  default=1)
    parser.add_argument('--symetric', type=int,  default=0)
    parser.add_argument('--num_samples', type=int,  default=0)
    parser.add_argument('--global_points', type=int,  default=36**3) 
    parser.add_argument('--global_points_test', type=int,  default=36**3)    
    parser.add_argument('--noise_scale', type=float,  default=[0.1])
#    parser.add_argument('--categories'      , type=str,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
    parser.add_argument('--categories'      , type=int,  default=[0,1,2,3,4,5,6,7,8,9,10,11,12], help='number of point samples')
#    parser.add_argument('--categories'      , type=int,  default=[2], help='number of point samples')
    parser.add_argument('--category_names', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
    parser.add_argument('--learning_rate', type=float,  default=0.000001)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=False)
    parser.add_argument('--plot_every', type=int,  default=1000)
    if socket.gethostname() == 'gidi-To-be-filled-by-O-E-M':
        parser.add_argument("--path"            , type=str, default="/media/gidi/SSD/Thesis/Data/ShapeNet_TF")
        parser.add_argument("--checkpoint_path" , type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/media/gidi/SSD/Thesis/Data/Checkpoints/")
        parser.add_argument("--pretrained_path",  type=str, default="/media/gidi/SSD/Thesis/Data/pretrained/")
    elif socket.gethostname() == 'Gidis-MBP-3':
        parser.add_argument("--path"            , type=str, default="/Users/gidilittwin/meta_data/ShapeNet_TF")
        parser.add_argument("--checkpoint_path" , type=str, default="/Users/gidilittwin/meta_data/checkpoints/")
        parser.add_argument("--saved_model_path", type=str, default="/Users/gidilittwin/meta_data/checkpoints/")   
        parser.add_argument("--pretrained_path",  type=str, default="/Users/gidilittwin/meta_data/pretrained/")
    else:
        parser.add_argument("--path"            , type=str, default="/private/home/wolf/gidishape/data/ShapeNet_TF")
        parser.add_argument("--checkpoint_path" , type=str, default="/private/home/wolf/gidishape/checkpoints2/")
        parser.add_argument("--saved_model_path", type=str, default="/private/home/wolf/gidishape/checkpoints2/")   
        parser.add_argument("--pretrained_path",  type=str, default="/private/home/wolf/gidishape/pretrained/")
    
    return parser.parse_args()
config = parse_args()

if config.grid_size==32:
    config.grid_size_v = 256
    config.img_size    = [224,224]
    config.im_per_obj  = 20
    config.test_size   = 20
    config.shuffle_size= 1000
    config.test_every  = 10000
    config.save_every  = 1000
#    config.postfix     = str(config.stage)+'_'+str(config.grid_size)
    config.postfix_load     = str(config.stage-1)+'_'+str(config.grid_size)
    config.postfix_save     = str(config.stage)+'_'+str(config.grid_size)    
    config.fast_eval   = 0
    config.path        = config.path+str(config.grid_size)+'/'
elif config.grid_size==64:
    config.grid_size_v = 256
    config.img_size    = [224,224]
    config.im_per_obj  = 20
    config.test_size   = 20
    config.shuffle_size= 1000
    config.test_every  = 10000
    config.save_every  = 1000
    config.postfix     = str(config.stage)+'_'+str(config.grid_size)
    config.fast_eval   = 0
    config.path        = config.path+str(config.grid_size)+'/'
elif config.grid_size==128:
    config.grid_size_v = 256
    config.img_size    = [137,137]
    config.im_per_obj  = 20
    config.test_size   = 20
    config.shuffle_size= 100
    config.test_every  = 10000
    config.save_every  = 1000
    config.compression = 0
    config.postfix_load     = str(config.stage-1)+'_'+str(config.grid_size)
    config.postfix_save     = str(config.stage)+'_'+str(config.grid_size)
    config.fast_eval   = 0
    config.path        = config.path+str(config.grid_size)+'_v3/'
elif config.grid_size==256:
    config.grid_size_v = 256
    config.img_size    = [224,224]
    config.im_per_obj  = 20
    config.test_size   = 20
    config.shuffle_size= 100
    config.test_every  = 10000
    config.save_every  = 1000
    config.compression = 0
    config.postfix_load     = str(config.stage-1)+'_'+str(config.grid_size)
    config.postfix_save     = str(config.stage)+'_'+str(config.grid_size)
    config.fast_eval   = 0
    config.path        = config.path+str(config.grid_size)+'_v2/'
elif config.grid_size==36:
    config.grid_size_v = 36
    config.path        = config.path+'/'
    config.postfix_load     = str(config.stage-1)+'_'+str(config.grid_size)
    config.postfix_save     = str(config.stage)+'_'+str(config.grid_size)    
if config.pretrained==1:
    config.rgba = 0
    
if config.multi_image==1:
    train_iterator_batch_size = config.batch_size/config.multi_image_views
else:
    train_iterator_batch_size = config.batch_size


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


    

#%% Data iterators
train_iterator = TFH.iterator(config.path+'train/',
                              train_iterator_batch_size,
                              epochs=10000,
                              shuffle=True,
                              img_size=config.img_size[0],
                              im_per_obj=config.im_per_obj,
                              grid_size=config.grid_size,
                              num_samples=10000,
                              shuffle_size=config.shuffle_size,
                              categories = config.categories,
                              compression = config.compression)


test_iterator  = TFH.iterator(config.path+'test/',
                              1,
                              epochs=10000,
                              shuffle=False,
                              img_size=config.img_size[0],
                              im_per_obj=config.im_per_obj,
                              grid_size=config.grid_size,
                              num_samples=10000,
                              shuffle_size=config.shuffle_size,
                              categories = config.categories,
                              compression = config.compression)
    

idx_node          = tf.placeholder(tf.int32,shape=(), name='idx_node')  
level_set         = tf.placeholder(tf.float32,shape=(),   name='levelset')  
next_element      = train_iterator.get_next()
next_element_test = test_iterator.get_next()

if not config.surfaces:
    
    
    # if config.multi_image==1:
    #     next_batch        = TFH.process_batch_center_train(next_element,config)
    # else:
    #     next_batch        = TFH.process_batch_train(next_element,idx_node,config)
        
    next_batch = TFH.process_batch_render(next_element,idx_node,config)
        
    if config.grid_size==256 or config.grid_size==128 or config.grid_size==32:
        next_batch_test = TFH.process_batch_evaluate(next_element_test,idx_node,config)
    else:
        next_batch_test = TFH.process_batch_test(next_element_test,idx_node,config)
        
else:
    next_batch        = TFH.process_batch_surface(next_element,idx_node,config,config.batch_size)
    next_batch_test   = TFH.process_batch_surface_test(next_element_test,idx_node,config,config.test_size)

grid_size_lr = config.grid_size
x            = np.linspace(-1, 1, grid_size_lr)
y            = np.linspace(-1, 1, grid_size_lr)
z            = np.linspace(-1, 1, grid_size_lr)
xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)


if True==False:

    import matplotlib.pyplot as plt   
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    #session.run(mode_node.assign(False)) 
    session.run(train_iterator.initializer)
    batch,batch_ = session.run([next_element,next_batch],feed_dict={idx_node:0})
    idx=0
    
    vertices             = batch_['samples_xyz'][:,:,:]
    cubed = {'vertices':vertices[0,:,:],'faces':[],'vertices_up':vertices[0,:,:]}
    MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  
    
    
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
    
    
    
    pic = batch_['images'][idx,:,:,3]
    fig = plt.figure()
    plt.imshow(pic)
    
    
    
    
    aa=np.max(psudo_sdf*(zz_lr+1)/2.,axis=-1)
    fig = plt.figure()
    plt.imshow(aa)    
    
    aa=np.max(psudo_sdf*(xx_lr+1)/2.,axis=-2)
    fig = plt.figure()
    plt.imshow(aa)    

    aa=np.max(psudo_sdf*(yy_lr+1)/2.,axis=-3)
    fig = plt.figure()
    plt.imshow(aa)   



#%% Function wrappers     
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')

def g_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_shape(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   

def g_wrapper2(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = CNN.mlp(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   
    
def g_conv_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_sensor(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   
    
def f_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)

def f3_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return current
    
def r_wrapper(encoding,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.render(encoding,args_)
        return current
    
def m_wrapper(ids,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.multiplexer(ids,args_)
        return CNN.regressor(current,args_)    

def f2_wrapper(image,args_):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_50(image,args_)
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_) 

def f_conv_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.conv_regressor(current,args_)
    
    
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

    evals_function        = SF.sample_points_list(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)
    evals_function        = SF.render_sil(evals_function,evals_target,config)
    loss_class            = evals_function['loss_sil']
    
    labels                = tf.cast(tf.less_equal(tf.reshape(evals_target['y'],(batch_size,-1)),0.0),tf.int64)
    logits                = tf.reshape(evals_function['y'],(batch_size,-1,1)) #- levelset
    logits_iou            = tf.concat((logits-level_set,-logits+level_set),axis=-1)
    logits_ce             = tf.concat((logits,-logits),axis=-1)
    predictions           = tf.nn.softmax(logits_iou)
    correct_prediction    = tf.equal(tf.argmax(predictions, 2), labels)
    accuracy              = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    err                   = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # loss_class            = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_ce,name='cross-entropy'),axis=-1)
#    loss_sdf              = tf.reduce_mean((evals_function['dydx_norm']-1.0)**2)
    loss                  = tf.reduce_mean(loss_class) #+ config.norm_loss_alpha*loss_sdf
    X                     = tf.cast(labels,tf.bool)
    Y                     = tf.cast(tf.argmax(predictions, 2),tf.bool)
    iou_image             = tf.reduce_sum(tf.cast(tf.logical_and(X,Y),tf.float32),axis=1)/tf.reduce_sum(tf.cast(tf.logical_or(X,Y),tf.float32),axis=1)
    iou                   = tf.reduce_mean(iou_image)
    return {'loss':loss,'loss_class':tf.reduce_mean(loss_class),'accuracy':accuracy,'err':err,'iou':iou,'iou_image':iou_image,'evals_function':evals_function}

train_dict = build_graph(next_batch,config,batch_size=config.batch_size)
test_dict  = build_graph(next_batch_test,config,batch_size=config.test_size)

    
    
    
#def build_graph(next_batch,config,batch_size):
#    images                = next_batch['images'] 
#    vertices              = next_batch['vertices']  
#    samples_xyz           = next_batch['samples_xyz']
#    evals_target          = {}
#    evals_target['x']     = samples_xyz
#    evals_target['y']     = vertices
#    g_weights             = f_wrapper(images,[mode_node,config])
#    evals_function        = SF.sample_points_list(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,config.num_samples],samples=evals_target['x'] , use_samps=True)
#    evals_function['y']   = samples_xyz + evals_function['y'] 
#    dist                  = CHAMFER.nn_distance_cpu(evals_function['y'],evals_target['y'])
#    loss                  = 0.5*(tf.reduce_mean(dist[0]**2) + tf.reduce_mean(dist[2]**2))
#    err                   = 0.5*(tf.reduce_mean(dist[0]) + tf.reduce_mean(dist[2]))
#    acc                   = err
#    return {'loss':loss,'err':err,'accuracy':acc,'images':images,'iou':acc,'iou_image':acc, 'vertices':evals_target['y'] ,'shape':evals_function['y']}
#
#train_dict = build_graph(next_batch,config,batch_size=config.batch_size)
#test_dict  = build_graph(next_batch_test,config,batch_size=config.test_size)


#def build_image_graph(next_batch,config,batch_size):
#    images                = next_batch['images'] 
#    samples_sdf           = next_batch['samples_sdf']  
#    samples_xyz           = next_batch['samples_xyz']
#    evals_target          = {}
#    evals_target['x']     = samples_xyz
#    evals_target['y']     = samples_sdf
#    evals_target['mask']  = tf.cast(tf.greater(samples_sdf,0),tf.float32)
#    g_weights             = f_wrapper(images,[mode_node,config])
##    g_weights             = f_conv_wrapper(images,[mode_node,config])
#    evals_function        = SF.camera_vector(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,10], use_samps=False)
##    evals_function        = SF.const_vector(model_fn = g_conv_wrapper,args=[mode_node,g_weights,config],shape = [batch_size,4], use_samps=False)
#    eval_rendering        = r_wrapper(evals_function['y'],[mode_node,config])
#    labels                = tf.image.resize_images(images[:,:,:,0:3],eval_rendering.shape[1:3])
#    logits                = eval_rendering
##    logits                = tf.reshape(evals_function['y'],(batch_size,config.img_size[0],config.img_size[1],1)) #- levelset
##    loss                  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
#    loss                  = tf.reduce_mean((labels-tf.sigmoid(logits))**2)
#    err                   = tf.reduce_mean(tf.sqrt((labels-tf.sigmoid(logits))**2))
#    acc                   = 1.-err
#    return {'loss':loss,'err':err,'accuracy':acc,'logits':logits,'images':next_batch['images'],'iou':acc,'iou_image':acc }
#
#train_dict = build_image_graph(next_batch,config,batch_size=config.batch_size)
#test_dict  = build_image_graph(next_batch_test,config,batch_size=config.test_size)



with tf.variable_scope('optimization_cnn',reuse=tf.AUTO_REUSE):
    cnn_vars      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = '2d_cnn_model')
    lr_node       = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    optimizer     = tf.train.AdamOptimizer(lr_node,beta1=config.beta1,beta2=0.999)
    grads         = optimizer.compute_gradients(train_dict['loss'],var_list=cnn_vars)
    global_step   = tf.train.get_or_create_global_step()
    clip_constant = 10
    g_v_rescaled  = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads]
    train_op_cnn  = optimizer.apply_gradients(g_v_rescaled, global_step=global_step)
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



def evaluate(test_iterator, session, mode_node, config, test_dict, next_element_test):
    session.run(mode_node.assign(False)) 
    acc_mov_test  = MOV_AVG(3000000) # exact mean
    iou_mov_test  = MOV_AVG(3000000) # exact mean
    classes       = []
    ids           = []  
    ious          = []
    if config.fast_eval!=0:
        num_epochs = config.im_per_obj/config.test_size
    else:
        num_epochs = 1
    for epoch_test in range(num_epochs):
        session.run(test_iterator.initializer)
        while True:
            try:
                feed_dict = {lr_node            :config.learning_rate,
                             idx_node           :epoch_test%config.im_per_obj,
                             level_set          :config.levelset}  
                accuracy_t_ ,iou_t_, iou_image_t, batch_ = session.run([test_dict['accuracy'], test_dict['iou'], test_dict['iou_image'], next_element_test],feed_dict=feed_dict) 
                acc_mov_avg_test = acc_mov_test.push(accuracy_t_)
                iou_mov_avg_test = iou_mov_test.push(iou_t_)
                classes.append(np.tile(batch_['classes'],(config.test_size,1)))
                ids.append(np.tile(batch_['ids'],(config.test_size,1)))
                ious.append(iou_image_t)   
            except tf.errors.OutOfRangeError:
                print('TEST::  epoch: '+str(epoch_test)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,IOU: '+str(iou_mov_avg_test))
                break
    session.run(mode_node.assign(True)) 
    classes  = np.concatenate(classes,axis=0)[:,0]
    ious     = np.concatenate(ious,axis=0) 
    ids      = np.concatenate(ids,axis=0)[:,0]     
#    mean_average_iou = ma_iou(ious,classes,config)
    return acc_mov_avg_test, iou_mov_avg_test, classes, ids, ious




    
session = tf.Session()
session.run(tf.initialize_all_variables())
loss_plot      = []
acc_plot       = []
iou_plot       = []
acc_plot_test  = []
iou_plot_test  = []
max_test_acc   = 0.
max_test_iou   = 0.

if config.pretrained:
    pretrained.restore(session,config.pretrained_path+'resnet_v2_50.ckpt')
if config.finetune:
    loader.restore(session, directory+'/latest_train'+config.postfix_load+'-0')
    loss_plot     = np.load(directory+'/loss_values'+config.postfix_load+'.npy')
    acc_plot      = np.load(directory+'/accuracy_values'+config.postfix_load+'.npy')  
    iou_plot      = np.load(directory+'/iou_values'+config.postfix_load+'.npy')      
    acc_plot_test = np.load(directory+'/accuracy_values_test'+config.postfix_load+'.npy') 
    iou_plot_test = np.load(directory+'/iou_values_test'+config.postfix_load+'.npy') 
    loss_plot     = np.split(loss_plot,loss_plot.shape[0])
    acc_plot      = np.split(acc_plot,acc_plot.shape[0])
    iou_plot      = np.split(iou_plot,iou_plot.shape[0])
    acc_plot_test = np.split(acc_plot_test,acc_plot_test.shape[0])
    iou_plot_test = np.split(iou_plot_test,iou_plot_test.shape[0])    
step           = 0
acc_mov        = MOV_AVG(300) # moving mean
loss_mov       = MOV_AVG(300) # moving mean
iou_mov        = MOV_AVG(300) # moving mean
#sdf_mov        = MOV_AVG(300) # moving mean

# train_dict_ = session.run(train_dict,feed_dict=feed_dict) 

               
session.run(mode_node.assign(True)) 
for epoch in range(1000000):
    session.run(train_iterator.initializer)
    while True:
        try:
            feed_dict = {lr_node            :config.learning_rate,
                         idx_node           :epoch%config.im_per_obj,
                         level_set          :config.levelset}     
            _, train_dict_ = session.run([train_op_cnn, train_dict],feed_dict=feed_dict) 
            acc_mov_avg  = acc_mov.push(train_dict_['accuracy'])
            loss_mov_avg = loss_mov.push(train_dict_['loss_class'])
            iou_mov_avg  = iou_mov.push(train_dict_['iou'])   
#            sdf_mov_avg  = sdf_mov.push(train_dict_['loss_sdf'])   

            if step % 10 == 0:
                print('Training: epoch: '+str(epoch)+' ,avg_accuracy: '+str(acc_mov_avg)+' ,avg_loss: '+str(loss_mov_avg)+' ,IOU: '+str(iou_mov_avg)+' ,max_test_IOU: '+str(max_test_iou))
            if step % config.plot_every == 0:
                acc_plot.append(np.expand_dims(np.array(acc_mov_avg),axis=-1))
                loss_plot.append(np.expand_dims(np.array(np.log(loss_mov_avg)),axis=-1))
                iou_plot.append(np.expand_dims(np.array(iou_mov_avg),axis=-1)) 
                np.save(directory+'/loss_values'+config.postfix_save+'.npy',np.concatenate(loss_plot))
                np.save(directory+'/accuracy_values'+config.postfix_save+'.npy',np.concatenate(acc_plot))  
                np.save(directory+'/iou_values'+config.postfix_save+'.npy',np.concatenate(iou_plot)) 
            if step % config.test_every == config.test_every -1:            
                acc_test, iou_test, classes_test, ids_test, ious_test = evaluate(test_iterator, session, mode_node, config, test_dict, next_element_test)
                acc_plot_test.append(np.expand_dims(np.array(acc_test),axis=-1))
                iou_plot_test.append(np.expand_dims(np.array(iou_test),axis=-1))            
                np.save(directory+'/accuracy_values_test'+config.postfix_save+'.npy',np.concatenate(acc_plot_test))  
                np.save(directory+'/iou_values_test'+config.postfix_save+'.npy',np.concatenate(iou_plot_test)) 
                if iou_test>max_test_iou:
                    saver.save(session, directory+'/latest'+config.postfix_save, global_step=0)
                    max_test_iou = iou_test
                print('Testing:  max_test_accuracy: '+str(max_test_acc)+' ,max_test_IOU: '+str(max_test_iou))
            if step % config.save_every == config.save_every -1:  
                saver.save(session, directory+'/latest_train'+config.postfix_save, global_step=0)
            step+=1                
        except tf.errors.OutOfRangeError:
            break
 
    
  
#%% RENDERING
#if True==False:
 


x    = tf.placeholder(tf.float32,shape=(4,10,3), name='xx')  
y    = tf.reduce_sum(x**2,axis=-1,keep_dims=True)


dy_dx = tf.gradients(y,x)

sess = tf.Session()

x_val = np.random.randint(0, 10, (4, 10,3))*1.0
y_val, dy_dx_val = sess.run([y, dy_dx], {x:x_val})

    
#%% TESTING
if True==False:
 
    
    
    
    session.run(mode_node.assign(True)) 
    session.run(train_iterator.initializer)
    feed_dict = {lr_node            :config.learning_rate,
                 idx_node           :0%config.im_per_obj,
                 level_set          :config.levelset}     
    
    
    batch_, next_batch_, train_dict_ = session.run([next_element, next_batch, train_dict],feed_dict=feed_dict) 
        
    samples          = next_batch_['samples_xyz'][0,:,:]
    field            = next_batch_['samples_sdf'][0,:,0]
    vertices         = samples[field<0,:]
    
    cubed = {'vertices':vertices,'faces':[],'vertices_up':vertices}
    MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  
    
    psudo_sdf = batch_['voxels'][0,:,:,:]*1.0
    verts0, faces0, normals0, values0 = measure.marching_cubes_lewiner(psudo_sdf, 0.0)
    cubed0 = {'vertices':verts0/(config.grid_size-1)*2-1,'faces':faces0,'vertices_up':verts0/(config.grid_size-1)*2-1}
    MESHPLOT.mesh_plot([cubed0],idx=0,type_='mesh')    
#    
#    import matplotlib.pyplot as plt   
#    pic = next_batch_['images'][0,:,:,0:3]
#    fig = plt.figure()
#    plt.imshow(pic)
    
    vertices_         = train_dict_['vertices'][0,:,:]
    shape_            = train_dict_['shape'][0,:,:]
    sphere_           = next_batch_['samples_xyz'][0,:,:]
    cubed             = {'vertices':shape_,'faces':[],'vertices_up':shape_}
    MESHPLOT.mesh_plot([cubed],idx=0,type_='cloud')  
    
    
    
    
    
    
    
    
    
    
    
    
    config.test_size=1
    config.batch_size=1
    idx_node          = tf.placeholder(tf.int32,shape=(), name='idx_node')  
    level_set         = tf.placeholder(tf.float32,shape=(),   name='levelset')  
    grid_size_lr = config.grid_size
    x            = np.linspace(-1, 1, grid_size_lr)
    y            = np.linspace(-1, 1, grid_size_lr)
    z            = np.linspace(-1, 1, grid_size_lr)
    xx_lr,yy_lr,zz_lr    = np.meshgrid(x, y, z)
    
    dispaly_iterator  = TFH.iterator(config.path+'train/',
                                  1,
                                  epochs=10000,
                                  shuffle=True,
                                  img_size=config.img_size[0],
                                  im_per_obj=config.im_per_obj,
                                  grid_size=config.grid_size,
                                  num_samples=config.num_samples,
                                  shuffle_size=config.shuffle_size,
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
    
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    loader   = tf.train.Saver(var_list=all_vars)
    
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    loader.restore(session, directory+'/latest'+config.postfix_load+'-0')
    session.run(mode_node.assign(False)) 
    session.run(dispaly_iterator.initializer)
    feed_dict = {idx_node           :0,
                 level_set          :0}   
    
    
      
    evals_target_, evals_function_ = session.run([evals_target, evals_function],feed_dict=feed_dict) 
                
    field              = np.reshape(evals_function_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.0)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  
     
    
    
    field              = np.reshape(evals_target_['y'][0,:,:],(-1,))
    field              = np.reshape(field,(grid_size_lr,grid_size_lr,grid_size_lr,1))
    if np.min(field[:,:,:,0])<0.0 and np.max(field[:,:,:,0])>0.0:
        verts, faces, normals, values = measure.marching_cubes_lewiner(field[:,:,:,0], 0.)
        cubed_plot = {'vertices':verts/(grid_size_lr-1)*2-1,'faces':faces,'vertices_up':verts/(grid_size_lr-1)*2-1}
        MESHPLOT.mesh_plot([cubed_plot],idx=0,type_='mesh')  
     
    








               

