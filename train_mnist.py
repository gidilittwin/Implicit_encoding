
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
mnist = tf.keras.datasets.mnist



def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment_name', type=str, default= 'mnist_embeddings')
    parser.add_argument('--model_params_path', type=str, default= './archs/mnist_light.json')
    parser.add_argument('--padding', type=str, default= 'VALID')
    parser.add_argument('--model_params', type=str, default= None)
    parser.add_argument('--batch_size', type=int,  default=128)
    parser.add_argument('--beta1', type=float,  default=0.9)
    parser.add_argument('--dropout', type=float,  default=1.0)
    parser.add_argument('--stage', type=int,  default=0)
    parser.add_argument('--multi_image', type=int,  default=0)
    parser.add_argument('--multi_image_views', type=int,  default=24)
    parser.add_argument('--multi_image_pool', type=str,  default='max')
    
    parser.add_argument('--alpha', type=float,  default=0.003)
    parser.add_argument('--grid_size', type=int,  default=28)
    parser.add_argument('--grid_size_v', type=int,  default=28)
    parser.add_argument('--compression', type=int,  default=1)
    parser.add_argument('--pretrained', type=int,  default=0)
    
    parser.add_argument('--img_size', type=int,  default=[28,28])
    parser.add_argument('--im_per_obj', type=int,  default=128)
    parser.add_argument('--test_size', type=int,  default=128)
    parser.add_argument('--shuffle_size', type=int,  default=1000)  
    parser.add_argument('--test_every', type=int,  default=10000)    
    parser.add_argument('--save_every', type=int,  default=1000) 
    parser.add_argument("--postfix_load"   , type=str, default="")
    parser.add_argument('--fast_eval', type=int,  default=0)    

    parser.add_argument('--eval_grid_scale', type=int,  default=1)
    parser.add_argument('--batch_norm', type=int,  default=0)
    parser.add_argument('--bn_l0', type=int,  default=0)
    parser.add_argument('--augment', type=int,  default=1)
    parser.add_argument('--rgba', type=int,  default=1)
    parser.add_argument('--symetric', type=int,  default=0)
    parser.add_argument('--radius', type=float,  default=0.1)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--global_points', type=int,  default=1000) 
    parser.add_argument('--global_points_test', type=int,  default=2000)    
    parser.add_argument('--noise_scale', type=float,  default=0.05)
    parser.add_argument('--categories'      , type=int,  default=[0,1,2,3,4,5,6,7,8,9], help='number of point samples')
    parser.add_argument('--category_names', type=int,  default=["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"], help='number of point samples')
    parser.add_argument('--learning_rate', type=float,  default=0.00001)
    parser.add_argument('--levelset'  , type=float,  default=0.0)
    parser.add_argument('--finetune'  , type=bool,  default=False)
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



MODEL_PARAMS = config.model_params_path
MODE_TRAIN = 0
MODE_TEST  = 1
with open(MODEL_PARAMS, 'r') as f:
    model_params = json.load(f)
config.model_params = model_params    
directory = config.checkpoint_path + config.experiment_name 
    
    

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

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
number_of_classes = np.unique(y_train).shape[0]
shape = (28, 28, 1)
batch_size = 128


with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.Variable(True, name='mode_node')

def g_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_shape(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function

def f_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.mnist_config(image,args_)
        return CNN.regressor(current,args_)
    

def f_embedding(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.mnist_config(image,args_)
        return current
    

def g_embedding(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = CNN.mlp(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function








#%% Training graph 
next_batch = {}
next_batch['images']  = tf.placeholder(tf.uint8,shape=(None,28,28,1), name='images')  
    
#def build_graph(next_batch,config,batch_size):
#    images                = next_batch['images'] 
#    g_weights             = f_wrapper(images,[mode_node,config])
#    evals_function        = SF.sample_points_list_2D(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = config.grid_size, use_samps=False)
#    labels                = tf.cast(next_batch['images'],tf.float32)/255.
#    logits                = tf.sigmoid(tf.reshape(evals_function['y'],(batch_size,config.grid_size,config.grid_size,1))) #- levelset
#    loss                  = tf.reduce_mean((labels-logits)**2)
#    err                   = tf.reduce_mean(tf.sqrt((labels-logits)**2))
#    acc                   = 1.-tf.reduce_mean(tf.sqrt((labels-logits)**2))
#    return {'loss':loss,'err':err,'accuracy':acc,'logits':logits,'images':next_batch['images'] }


#def build_dispaly_graph(next_batch,config,batch_size):
#    images                = next_batch['images'] 
#    g_weights             = f_wrapper(images,[mode_node,config])
#    evals_function        = SF.sample_points_list_2D(model_fn = g_wrapper,args=[mode_node,g_weights,config],shape = 1024, use_samps=False)
#    logits                = tf.sigmoid(tf.reshape(evals_function['y'],(batch_size,1024,1024,1))) #- levelset
#    return {'logits':logits,'images':next_batch['images'] }


def build_graph(next_batch,config,batch_size):
    images                = next_batch['images'] 
    embedding             = f_embedding(images,[mode_node,config])
    evals_function        = SF.sample_points_list_2D(model_fn = g_embedding,args=[mode_node,embedding,config],shape = [batch_size,config.grid_size], use_samps=False)
    labels                = tf.cast(next_batch['images'],tf.float32)/255.
    logits                = tf.sigmoid(tf.reshape(evals_function['y'],(batch_size,config.grid_size,config.grid_size,1))) #- levelset
    loss                  = tf.reduce_mean((labels-logits)**2)
    err                   = tf.reduce_mean(tf.sqrt((labels-logits)**2))
    acc                   = 1.-tf.reduce_mean(tf.sqrt((labels-logits)**2))
    return {'loss':loss,'err':err,'accuracy':acc,'logits':logits,'images':next_batch['images'] }



train_dict = build_graph(next_batch,config,batch_size=config.batch_size)
#display_dict = build_dispaly_graph(next_batch,config,batch_size=config.batch_size)




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



#%% Train loop
    
session = tf.Session()
session.run(tf.initialize_all_variables())
loss_plot      = []
acc_plot       = []
acc_plot_test  = []
loss_plot_test = []
acc_plot_test_max = 0.
acc_mov        = MOV_AVG(300) # moving mean
loss_mov       = MOV_AVG(300) # moving mean
epoch          = 0

while epoch < 1000000:
    idx  = np.random.permutation(np.arange(0,60000))
    step = 0
    session.run(mode_node.assign(True)) 
    while step<60000/config.batch_size:
        feed_dict = {lr_node             :config.learning_rate,
                     next_batch['images']: X_train[step*config.batch_size:(step+1)*config.batch_size,:,:,:]}     
        _, train_dict_ = session.run([train_op_cnn, train_dict],feed_dict=feed_dict) 
        acc_mov_avg  = acc_mov.push(train_dict_['accuracy'])
        loss_mov_avg = loss_mov.push(train_dict_['loss'])
        if step % 100 == 0:
            print('Training: epoch: '+str(epoch)+' ,avg_accuracy: '+str(acc_mov_avg)+' ,avg_loss: '+str(loss_mov_avg))
        step+=1    
    acc_plot.append(np.expand_dims(np.array(acc_mov_avg),axis=-1))
    loss_plot.append(np.expand_dims(np.array(np.log(loss_mov_avg)),axis=-1))
    np.save(directory+'/loss_values.npy',np.concatenate(loss_plot))
    np.save(directory+'/accuracy_values.npy',np.concatenate(acc_plot))  
    saver.save(session, directory+'/latest_train', global_step=0)

    idx  = np.random.permutation(np.arange(0,10000))
    step = 0
    session.run(mode_node.assign(False)) 
    acc_mov_test        = MOV_AVG(10000000) # moving mean
    loss_mov_test       = MOV_AVG(10000000) # moving mean
    while step<10000/config.batch_size:
        feed_dict = {lr_node             :config.learning_rate,
                     next_batch['images']: X_test[step*config.batch_size:(step+1)*config.batch_size,:,:,:]}     
        train_dict_  = session.run(train_dict,feed_dict=feed_dict) 
        acc_mov_avg_test  = acc_mov_test.push(train_dict_['accuracy'])
        loss_mov_avg_test = loss_mov_test.push(train_dict_['loss'])
        if step % 100 == 0:
            print('Testing: epoch: '+str(epoch)+' ,avg_accuracy: '+str(acc_mov_avg_test)+' ,avg_loss: '+str(loss_mov_avg_test))
        step+=1  
    acc_plot_test.append(np.expand_dims(np.array(acc_mov_avg_test),axis=-1))
    loss_plot_test.append(np.expand_dims(np.array(np.log(loss_mov_avg_test)),axis=-1))
    np.save(directory+'/loss_values_test.npy',np.concatenate(loss_plot_test))
    np.save(directory+'/accuracy_values_test.npy',np.concatenate(acc_plot_test))  
    if np.max(acc_plot_test)>acc_plot_test_max:
        acc_plot_test_max = np.max(acc_plot_test)
        saver.save(session, directory+'/best_test', global_step=0)
    epoch+=1


reconstructed = (train_dict_['logits']*255.).astype(np.uint8)
originals     = train_dict_['images']
feed_dict = {lr_node             :config.learning_rate,
             next_batch['images']: originals}     
display_dict_  = session.run(display_dict,feed_dict=feed_dict) 
reconstructed_hd = (display_dict_['logits']*255.).astype(np.uint8)

index = 96
import matplotlib.pyplot as plt   
pic = reconstructed[index,:,:,0]
fig = plt.figure(1)
plt.imshow(pic)

pic2 = originals[index,:,:,0]
fig = plt.figure(2)
plt.imshow(pic2)

pic3 = reconstructed_hd[index,:,:,0]
fig = plt.figure(3)
plt.imshow(pic3)





