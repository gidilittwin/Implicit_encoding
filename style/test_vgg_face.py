#####################################################################################################
# testing VGG face model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
#####################################################################################################

import vgg_face
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np

# build the graph
graph = tf.Graph()
with graph.as_default():
    image = tf.placeholder(tf.float32, [None, 224, 224, 3])
    network = 'vgg-face.mat'
    vgg_weights, average_image = vgg_face.load_net(network)

# read sample image
img = imread('/Users/gidilittwin/Dropbox/Thesis/neural-style-master/examples/CONTENT/face2_.jpg', mode='RGB')
img = imresize(img, [224, 224])
img = img - average_image

# run the graph
with tf.Session(graph=graph) as sess:
    # testing on the sample image
    net = vgg_face.net_preloaded(vgg_weights, image, [])
    content_features = net[layer].eval(feed_dict={image: img})
    
    
    
    [prob, ind, out] = sess.run([values, indices, output], feed_dict={input_maps: [img]})
    print prob, ind
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    for i in range(k):
        print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (class_names[ind[i]][0][0], prob[i]*100))
    sess.close()

