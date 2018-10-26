import tensorflow as tf
import numpy as np
import os
import models
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.io
import os
import scipy.misc
import scipy.io
import argparse
#DATA_PATH = '/home/etai/deep/cifar100/train/'
#VAL_DATA_PATH = '/home/etai/deep/cifar100/test/'
DATA_PATH = '/home/etai/deep/train/'
TEST_DATA_PATH = '/home/etai/deep/test/'
#PERM_PATH = '/home/etai/deep/cifar10/'



def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("--dataset", type=str, default="cifar100", help="which_set")
    parser.add_argument("--zca", type=int, default=0, help="perform zca")
    parser.add_argument("--bn", type=int, default=0, help="batch normalization")
    parser.add_argument("--batchsize", type=int, default=250, help="batchsize")
    parser.add_argument("--epochs", type=int, default=500, help="num_of_epochs")
    parser.add_argument("--sample_size", type=int, default=35, help="num_of_samples_for_var_est")
    parser.add_argument("--eps", type=float, default=0.1, help="epsilon_for_stability")
    parser.add_argument("--use_reg", type=int, default=1, help="use regularizer")
    parser.add_argument("--model", type=int, default=1, help="layer_width")
    parser.add_argument("--gamma", type=float, default=0.02, help="gamma_for_reg")
    parser.add_argument("--gamma_l2", type=float, default=0.0001, help="gamma_for_l2")
    parser.add_argument("--activation", type=str, default='elu', help="activation_type")
    parser.add_argument("--train_path_10", type=str, default='/home/etai/deep/train/', help="train_path")
    parser.add_argument("--test_path_10", type=str, default='/home/etai/deep/test/', help="test_path")
    parser.add_argument("--train_path_100", type=str, default='/home/etai/deep/cifar100/train/', help="train_path")
    parser.add_argument("--test_path_100", type=str, default='/home/etai/deep/cifar100/test/', help="test_path")
    args = parser.parse_args()

    return args


args = parseArguments()

def load_data(path):
    data = np.load(path+'data.npy')
    labels = np.load(path+'labels.npy')
    return data, labels

def random_batch(batch):
    shape = batch.shape
    new_batch = np.zeros((shape[0],shape[1]+8,shape[2]+8,shape[3]))
    new_batch[:,4:36,4:36,:] = batch
    for i in range(shape[0]):
        idx1 = np.random.randint(0,9)
        idx2 = np.random.randint(0,9)
        batch[i,:,:,:] = new_batch[i,idx1:(idx1+32),idx2:(idx2+32),:]
    return batch

def random_flip_batch(batch):
    for i in xrange(batch.shape[0]):
        if np.random.randn(1)>0:
            batch[i,:,:,:] = np.fliplr(batch[i,:,:,:])
#        batch[i,:,:,:] = tf.image.random_brightness(batch[i,:,:,:],max_delta=63).eval()                                               
    return batch

def lrelu(x):
    return tf.nn.relu(x) -0.2*tf.nn.relu(-x)


zca = args.zca
if args.activation=='elu':
    activation = tf.nn.elu
if args.activation=='relu':
    activation = tf.nn.relu
if args.activation=='lrelu':
    activation = lrelu
    
use_reg = args.use_reg
eps = args.eps
epochs = args.epochs
gamma = args.gamma
gamma_l2 = args.gamma_l2
sample_size = args.sample_size
batchsize = args.batchsize
bn = args.bn
model = args.model
dataset = args.dataset
if dataset=='cifar10':
    num_of_labels=10
    train_path = args.train_path_10
    test_path = args.test_path_10
else:
    num_of_labels=100
    train_path = args.train_path_100
    test_path = args.test_path_100


images, lab = load_data(train_path)
me = np.mean(images,0)
images = images - me
std = np.std(images,0)
images = images/std

#val_images, val_lab = load_data(args.val_path)
#val_images = (val_images - me)/std

test_images, test_lab = load_data(test_path)
test_images = (test_images - me)/std

if zca==1:
    v_im = np.reshape(images,(images.shape[0],32*32*3))
    cov = np.matmul(v_im.T,v_im)/images.shape[0]
    U, S, V = np.linalg.svd(cov)
    epsilon = 0.1
    t = np.dot(U, np.diag(1/np.sqrt(S+epsilon)))
    zcaWhiteMat = np.dot(t,U.T)    
    v_im = np.dot(v_im,zcaWhiteMat.T)
    images = np.reshape(v_im,(images.shape[0],32,32,3))
    
#    v_im_t = np.reshape(val_images,(val_images.shape[0],32*32*3))
#    v_im_t = np.dot(v_im_t,zcaWhiteMat.T)
#    val_images = np.reshape(v_im_t,(val_images.shape[0],32,32,3))
    
    v_im_t = np.reshape(test_images,(test_images.shape[0],32*32*3))
    v_im_t = np.dot(v_im_t,zcaWhiteMat.T)
    test_images = np.reshape(v_im_t,(test_images.shape[0],32,32,3))


session = tf.Session()


    

data   = tf.placeholder(tf.float32, shape=(batchsize, 32,32,3), name='data')
labels = tf.placeholder(tf.int32, shape=(batchsize), name='labels')
lr_node = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
isTrainVar = tf.Variable(False, name='isTrainVar', trainable = False)

if model==1:
    logits = models.conv_model1(data, isTrainVar, batchsize, num_of_labels, activation, sample_size, eps, bn)
else:
    logits = models.conv_model2(data, isTrainVar, batchsize, num_of_labels, activation, sample_size, eps, bn)

net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

l2_loss = tf.get_collection('l2_norm')
l2_loss = tf.add_n(l2_loss)

l2_n = tf.get_collection('l2_n')
l2_n_loss = tf.add_n(l2_n)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = labels))
if use_reg==1:
    loss = loss + gamma*l2_loss + gamma_l2*l2_n_loss
else:
    loss = loss + gamma_l2*l2_n_loss
    
opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
opt = tf.train.MomentumOptimizer(lr_node, 0.9)
grad = opt.compute_gradients(loss, var_list = opt_vars)
rescales_grad = [(tf.clip_by_norm(gv[0],1),gv[1]) for gv in grad]
train_op_net = opt.apply_gradients(rescales_grad)

predictions = logits
correct_prediction = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

feed_dict = {}
feed_dict[lr_node] = 0.05
init = tf.global_variables_initializer()
session.run(init)
saver = tf.train.Saver()
error_plot_train = []
error_plot_test = []
error_plot_val = []
min_test_error = 1
min_val_error = 1
test_at_min_val = 1
for i in range(0,epochs):       
    perm = np.random.permutation(images.shape[0])
    error = 0
    avg_loss = 0
    avg_const = 0
    session.run(isTrainVar.assign(True))
    if i>180:
        feed_dict[lr_node] = 0.02
    if i>300:
        feed_dict[lr_node] = 0.002
    if i>400:
        feed_dict[lr_node] = 0.0002
    for j in range(50000/batchsize):
        batch_idx = perm[j*batchsize:j*batchsize + batchsize]
        feed_dict[data] = random_batch(random_flip_batch(images[batch_idx,:,:,:]))
        feed_dict[labels] = lab[batch_idx]
        _, loss_val, error_val = session.run([train_op_net, loss, err], feed_dict = feed_dict)
        error+=error_val
        avg_loss+=loss_val
    error_plot_train.append(error/(j+1))

    session.run(isTrainVar.assign(False))
    error = 0
    for j in range(10000/batchsize):
        feed_dict[data] = test_images[j*batchsize:j*batchsize + batchsize,:,:,:]
        feed_dict[labels] = test_lab[j*batchsize:j*batchsize + batchsize]
        error_test = session.run(err, feed_dict = feed_dict)
        error+=error_test
    if min_test_error>error/(j+1):
        min_test_error = error/(j+1)
    error_plot_test.append(error/(j+1))
    
#    error = 0
#    for j in range(5000/batchsize):
#        feed_dict[data] = val_images[j*batchsize:j*batchsize + batchsize,:,:,:]
#        feed_dict[labels] = val_lab[j*batchsize:j*batchsize + batchsize]
#        error_val = session.run(err, feed_dict = feed_dict)
#        error+=error_val
#    if min_val_error>error/(j+1):
#        min_val_error = error/(j+1)
#        test_at_min_val = min_val_error
#    error_plot_val.append(error/(j+1))
    
    plt.figure(2)
    plt.plot(error_plot_train,'r',error_plot_test,'b')
    plt.pause(0.05)
    print('_______________________________________________________________')
    print("dataset: %s, use_reg: [%5d], eps: %.8f, activation: %s, gamma: %.8f, gamma_l2: %.8f, bn: [%5d]"  % (str(dataset), use_reg, eps, str(activation), args.gamma, args.gamma_l2, bn))
    print("Testing: epoch [%5d], min_test error: %.8f" % (i, min_test_error))
    print("Testing: epoch [%5d], test error: %.8f, train error: %.8f" % (i, error_plot_test[i], error_plot_train[i]))    


