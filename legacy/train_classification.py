import os
import sys
import tensorflow as tf
import numpy as np
from src.models import voxels as VOX
from src.models import model_config as CFG
from src.models import point_net as POINT
import provider
import argparse
from src.utilities import train_ops_local


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiments')
#    parser.add_argument('-f', '--train-settings', type=str, required=True,  help='JSON settings file for training.')
    parser.add_argument('--grid_size', type=int,  default=32, help='voxel grid size')
    parser.add_argument('--batch_size', type=int,  default=40, help='batch size')
    parser.add_argument('--num_points', type=int,  default=2048, help='number of point samples')
    parser.add_argument('--learning_rate', type=float,  default=0.00025, help='learning rate')
    parser.add_argument('--experiment', type=str,  default='test', help='experiment name')
    parser.add_argument('--sweep_param', type=float,  default=0, help='voxelnet_architecture')
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')



def train_point2voxel(grid_size,BATCH_SIZE,NUM_POINT,LEARNING_RATE,EXPNAME,SWEEP):


#    grid_size = 32
#    BATCH_SIZE = 2
#    NUM_POINT = 1024
#    LEARNING_RATE = 0.00025

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    LOG_DIR = BASE_DIR+'/LOGS/'
    MAX_EPOCH = 1000
    TRAIN_FILES = provider.getDataFiles(BASE_DIR+'/Data/ModelNet40/train_files.txt')
    TEST_FILES  = provider.getDataFiles(BASE_DIR+'/Data/ModelNet40/test_files.txt')
#    REPSIZE = 128
    REPSIZE = 128
    vox_params = CFG.voxelnet(SWEEP)
#    MODE = 'DEBUG'
    MODE = 'TRAIN'
    
    

    #%%  Graph   
    labels_node = tf.placeholder(tf.int64,   shape=(None), name='labels')
    lr_node = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
    mode_node = tf.Variable(True, name='mode_node')
    point_cloud_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_POINT,3), name='point_cloud')  
    
    
    with tf.variable_scope('model') as scope:
        with tf.variable_scope('pointnet') as scope:
            POINTS = POINT.POINTNET(num_points=NUM_POINT,grid_size=grid_size,batch_size=BATCH_SIZE,rep_size=REPSIZE)
            point_net = POINTS.build([point_cloud_node],mode_node,vox_params)
        with tf.variable_scope('voxelnet') as scope:
#            logits = VOX.model_3(point_net[-2], mode_node, BATCH_SIZE, vox_params)
            logits = point_net[-2]
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
    
    
    with tf.name_scope('loss') as scope:
#        gamma = 0.05
        gamma = vox_params['gamma'] 
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_node))
#        l2_loss = tf.get_collection('l2_norm')
        l2_loss = tf.get_collection('rot_inv')
        l2_loss = tf.add_n(l2_loss)
        loss = loss + gamma*l2_loss 
        predictions = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int64), labels_node)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        err = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
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
        tf.summary.scalar('Training_Accuracy', accuracy,collections=['training'])
        tf.summary.scalar('Validation_Accuracy', accuracy,collections=['validation'])                                     
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
            current_data, current_label = provider.loadDataFile(BASE_DIR+TRAIN_FILES[0])
            current_data = current_data[:,0:NUM_POINT,:]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            current_label = np.squeeze(current_label).astype(np.int64)
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            start_idx = 10 * BATCH_SIZE
            end_idx = (10+1) * BATCH_SIZE
            # Augment batched point clouds by rotation and jittering
            data_batch = current_data[start_idx:end_idx, :, :]
            data_batch = provider.rotate_point_cloud(data_batch)
            batch_mean = np.mean(data_batch,axis=1,keepdims=True)
            data_batch = data_batch-batch_mean
            batch_var = np.std(data_batch,axis=(1,2),keepdims=True)
            data_batch = data_batch/batch_var
#            current_data = provider.rotate_point_cloud(current_data)
#            current_data = provider.jitter_point_cloud(current_data)
            labels = current_label[start_idx:end_idx]
            feed_dict = {point_cloud_node: data_batch,labels_node:labels, lr_node:LEARNING_RATE}
            point_net_ = sess.run(point_net, feed_dict=feed_dict)
#            centers = point_net_[-3]
            train_ops_local.plot_cloud(np.squeeze(point_net_[0],-1),'green',0) 
            train_ops_local.plot_cloud(np.squeeze(point_net_[1],-1),'red',0) 
#            train_ops_local.plot_voxel(point_net_[-1],'green',0) 
#            print(np.mean(counts))
    
    
    
    #%% TRAIN LOOP
    if MODE=='TRAIN':
        
        LOSS = np.zeros((2,MAX_EPOCH),dtype = np.float32)
        ACC = np.zeros((2,MAX_EPOCH),dtype = np.float32)
        best_test_acc = 0
        for epoch in range(MAX_EPOCH):
            sys.stdout.flush()
            train_file_idxs = np.arange(0, len(TRAIN_FILES))
            np.random.shuffle(train_file_idxs)
            if epoch %100==99:
                LEARNING_RATE=LEARNING_RATE/2
            total_correct = 0
            total_seen = 0
            loss_sum = 0            
            sess.run(mode_node.assign(True)) 
            for fn in range(len(TRAIN_FILES)):
                current_data, current_label = provider.loadDataFile(BASE_DIR+TRAIN_FILES[train_file_idxs[fn]])
                current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))   
                current_label = np.squeeze(current_label).astype(np.int64)
                file_size = current_data.shape[0]
                num_batches = file_size // BATCH_SIZE
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    data_batch = current_data[start_idx:end_idx, :, :]
                    labels = current_label[start_idx:end_idx]
                    rand_samp = np.random.randint(0,2048,NUM_POINT)
                    data_batch = data_batch[:,rand_samp,:]                    
                    augmented_data = provider.rotate_point_cloud(data_batch)
                    batch_mean = np.mean(augmented_data,axis=1,keepdims=True)
                    augmented_data = augmented_data-batch_mean
                    batch_var = np.std(augmented_data,axis=(1,2),keepdims=True)
                    augmented_data = augmented_data/batch_var
                    augmented_data = provider.jitter_point_cloud(augmented_data)
                    augmented_data = provider.scale_point_cloud(augmented_data)
        #            train_ops_local.plot_cloud(rotated_data,'green',0) 
                    feed_dict = {point_cloud_node: augmented_data,labels_node:labels, lr_node:LEARNING_RATE}
                    summary, _, step, loss_, error_, accuracy_ ,point_net_ ,l2_loss_ = sess.run([training_summary_op, train_op_net, global_step,loss, err,accuracy,point_net,l2_loss], feed_dict=feed_dict)
                    
#                    if batch_idx==0:
#                       train_ops_local.plot_cloud(augmented_data,'green',0,point_net_[-3])   

                    train_writer.add_summary(summary, step)
                    total_correct += accuracy_
                    total_seen += 1
                    loss_sum += loss_
            print('Epoch: '+str(epoch)+':')
            print('   File: '+str(fn)+'/'+str(len(TRAIN_FILES))+'  batch: '+str(batch_idx)+'/'+str(num_batches))
            print('      Train: Mean loss: %f' % (loss_sum / float(total_seen)))
            print('      Train: Accuracy: %f' % (total_correct / float(total_seen)))
            print('      Reg            : %f' % l2_loss_)
            print('      Best Test Accuracy: %f' % best_test_acc)
            LOSS[0,epoch] = loss_sum / float(total_seen)
            ACC[0,epoch] = total_correct / float(total_seen)

            total_correct_test = 0
            total_seen_test = 0
            loss_sum_test = 0
            sess.run(mode_node.assign(False)) 
            for fn in range(len(TEST_FILES)):
                current_data, current_label = provider.loadDataFile(BASE_DIR+TEST_FILES[fn])              
                current_label = np.squeeze(current_label)
                file_size = current_data.shape[0]
                num_batches_test = file_size // BATCH_SIZE
                for batch_idx in range(num_batches_test):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    data_batch = current_data[start_idx:end_idx, :, :]
                    labels = current_label[start_idx:end_idx]                    
                    rand_samp = np.random.randint(0,2048,NUM_POINT)
                    data_batch = data_batch[:,rand_samp,:]  
                    batch_mean = np.mean(data_batch,axis=1,keepdims=True)
                    data_batch = data_batch-batch_mean
                    batch_var = np.std(data_batch,axis=(1,2),keepdims=True)
                    data_batch = data_batch/batch_var                    
                    feed_dict = {point_cloud_node: data_batch,labels_node:labels, lr_node:LEARNING_RATE}
                    summary, step, loss_, error_, accuracy_, predictions_ = sess.run([validation_summary_op, global_step,loss, err,accuracy,predictions], feed_dict=feed_dict)
                    test_writer.add_summary(summary, step)
                    total_correct_test += accuracy_
                    total_seen_test += 1
                    loss_sum_test += loss_
            print('Epoch: '+str(epoch)+':')
            print('   File: '+str(fn)+'/'+str(len(TEST_FILES))+'  batch: '+str(batch_idx)+'/'+str(num_batches_test))
            print('      Test: Mean loss: %f' % (loss_sum_test / float(total_seen_test)))
            print('      Test: Accuracy: %f' % (total_correct_test / float(total_seen_test)))
            print('      Best Test Accuracy: %f' % best_test_acc)
            LOSS[1,epoch] = loss_sum_test / float(total_seen_test)
            ACC[1,epoch] = total_correct_test / float(total_seen_test)                 
            np.save(LOG_DIR+EXPNAME+'_results.npy',[LOSS,ACC])
            if (total_correct_test / float(total_seen_test))>best_test_acc:
                best_test_acc = (total_correct_test / float(total_seen_test))
                       
            
#%%
if __name__ == '__main__':
    args = parse_args()
    train_point2voxel(args.grid_size,args.batch_size,args.num_points,args.learning_rate,args.experiment,args.sweep_param)
    print('Finished training')
    
    
    
    
    
    
    
    