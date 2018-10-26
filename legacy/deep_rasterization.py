import os
import sys
import tensorflow as tf
import numpy as np
from src.models import voxels as VOX
from src.models import model_config as CFG
from src.models import point_net as POINT
import provider
import argparse
import os 
from src.utilities import train_ops_local




grid_size = 600
BATCH_SIZE = 8
NUM_POINT = 1
LEARNING_RATE = 0.00025
MODE = 'DEBUG'
#MODE = 'TRAIN'
LOG_DIR = '/media/gidi/SSD/Dropbox/Thesis/ModelNet/LOGS/'
MAX_EPOCH = 400   
num_batches = 10000
EXPNAME = 'test'


#%%  Graph  
    
    
labels_node = tf.placeholder(tf.float32,   shape=(BATCH_SIZE,grid_size,grid_size), name='labels')
lr_node = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
mode_node = tf.Variable(True, name='mode_node')
point_cloud_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE,1,9), name='point_cloud')  
    
    
with tf.variable_scope('model') as scope:
    with tf.variable_scope('pointnet') as scope:
        POINTS = POINT.POINTNET(num_points=NUM_POINT,grid_size=grid_size,batch_size=BATCH_SIZE,rep_size=128)
        point_net = POINTS.build([point_cloud_node],mode_node)
    with tf.variable_scope('voxelnet') as scope:
#        logits = VOX.model_3(point_net[-2], mode_node, BATCH_SIZE, vox_params)
        logits = point_net[-1]
net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
    
    
with tf.variable_scope('loss') as scope:
    loss = tf.reduce_mean(tf.pow(logits-labels_node,2))
    err = loss


with tf.variable_scope('optim') as scope:
   # NaN check
    dummy_loss = tf.reduce_mean(tf.constant(np.zeros([0],dtype=np.float32)))
    loss_check = tf.is_nan(loss)
    loss = tf.cond(loss_check, lambda:dummy_loss, lambda:loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(lr_node, beta1=0.5)
    grads_and_vars = optimizer.compute_gradients(loss, var_list = net_vars)
    clip_constant = 0.01
    grads_and_vars_rescaled = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads_and_vars]
    train_op_net = optimizer.apply_gradients(grads_and_vars_rescaled, global_step=global_step)
                    

with tf.name_scope("classifier_metrics"):
    tf.summary.scalar('Training_Error', err,collections=['training'])
    tf.summary.scalar('Validation_Error', err,collections=['validation'])
    tf.summary.scalar('Training_Loss', loss,collections=['training'])
    tf.summary.scalar('Validation_Loss', loss,collections=['validation'])
training_summary_op = tf.summary.merge_all('training')
validation_summary_op = tf.summary.merge_all('validation')
saver = tf.train.Saver()
    
# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

# Add summary writers
train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

# Init variables
init = tf.global_variables_initializer()
sess.run(init) 
sess.run(mode_node.assign(True)) 



#%% DEBUG
if MODE=='DEBUG':
    with tf.variable_scope('debug') as scope:
#            from src.utilities import train_ops_local
        data_batch,labels = provider.generate_polys(BATCH_SIZE,grid_size)
        feed_dict = {point_cloud_node: data_batch,labels_node:labels, lr_node:LEARNING_RATE}
        point_net_ = sess.run(point_net, feed_dict=feed_dict)
        train_ops_local.plot_xyz(labels,labels,idx=0)  



#%% TRAIN LOOP
if MODE=='TRAIN':
    
    LOSS = np.zeros((2,MAX_EPOCH),dtype = np.float32)
    ACC = np.zeros((2,MAX_EPOCH),dtype = np.float32)
    best_test_acc = 0
    for epoch in range(MAX_EPOCH):
        sys.stdout.flush()
        if epoch %100==99:
            LEARNING_RATE=LEARNING_RATE/2
        total_correct = 0
        total_seen = 0
        loss_sum = 0            
        sess.run(mode_node.assign(True)) 
        for batch_idx in range(num_batches):
            data_batch,labels = provider.generate_polys(BATCH_SIZE,grid_size)
            feed_dict = {point_cloud_node: data_batch,labels_node:labels, lr_node:LEARNING_RATE}
            summary, _, step, loss_,logits_, error_ ,point_net_ = sess.run([training_summary_op, train_op_net, global_step,loss,logits, err,point_net], feed_dict=feed_dict)
            
            if batch_idx==0:
               train_ops_local.plot_xyz(logits_,labels,idx=0)  
            

            train_writer.add_summary(summary, step)
            total_seen += 1
            loss_sum += loss_
            print('Epoch: '+str(epoch)+':')
            print('      Train: Mean loss: %f' % (loss_sum / float(total_seen)))
            print('      Train: Accuracy: %f' % (total_correct / float(total_seen)))
            print('      Best Test Accuracy: %f' % best_test_acc)
        LOSS[0,epoch] = loss_sum / float(total_seen)
        ACC[0,epoch] = total_correct / float(total_seen)
               
