# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import vgg
#import vgg_face as vgg
import tensorflow as tf
import numpy as np
#import src.facenet as facenet

from sys import stderr
from sklearn.feature_extraction import image as sk_image
from PIL import Image
#CONTENT_LAYERS = ('relu4_2','relu5_2')
#STYLE_LAYERS = ('relu3_1', 'relu4_1')
CONTENT_LAYERS = ('relu5_2',)
STYLE_LAYERS = ('relu3_1',)
kernel_s = 5


try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)



    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})



    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features_bank = sk_image.extract_patches_2d(np.squeeze(features), (kernel_s, kernel_s))
                style_features[i][layer] = [features_bank,features]





    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
#            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
       

        # content loss
        content_layers_weights = {}
#        content_layers_weights['relu4_2'] = content_weight_blend
#        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend
        content_layers_weights['relu4_2'] = 0.5
        content_layers_weights['relu5_2'] = 0.5
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:              
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)




        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                
                # Calculate normalized layer
                layer = tf.expand_dims(net[style_layer],axis=4)
                paddings = [[0, 0], [1,1], [1,1], [0, 0],[0,0]]
                layer_depth = layer.get_shape().as_list()[3]
                layer_pad = tf.pad(layer, paddings, "CONSTANT")
                layer_norm = tf.sqrt(tf.nn.conv3d(tf.pow(layer_pad,2),tf.ones((kernel_s,kernel_s,layer_depth,1,1),dtype=tf.float32),strides=[1, 1, 1, 1, 1],padding='VALID'))
                
                # Calculate normalized filter bank
                style_filters = np.transpose(style_features[i][style_layer][0],(1,2,3,0))
                style_filters = np.expand_dims(style_filters,axis=3)
                style_filters_norm = np.sqrt(np.sum(np.power(style_filters,2),axis=(0,1,2)))
                style_filters_normalized = style_filters/style_filters_norm
                
                # Calculate normalized correlations
                layer_filtered = tf.nn.conv3d(layer_pad,style_filters_normalized,strides=[1, 1, 1, 1, 1],padding='VALID')/layer_norm
                
                # Find maximum response and index into the filters
                max_filter_response_idx = tf.squeeze(tf.argmax(layer_filtered,axis=4))
#                max_filter_response_idx = tf.squeeze(tf.argmax(tf.abs(layer_filtered),axis=4))
                max_filter_response_idx = tf.reshape(max_filter_response_idx,[-1])
                max_filter_response_weight = tf.squeeze(tf.reduce_max(tf.abs(layer_filtered),axis=4))
                max_filter_response_weight = tf.reshape(max_filter_response_weight,[-1])
                max_filter_response_weight = max_filter_response_weight/tf.reduce_max(max_filter_response_weight)
                
                style_filters_tf = tf.transpose(tf.squeeze(tf.convert_to_tensor(style_filters, np.float32)),(3,0,1,2))
                style_filters_tf_gathered = tf.gather(style_filters_tf,max_filter_response_idx)
                style_filters_tf_gathered = tf.reshape(style_filters_tf_gathered,(style_filters_tf_gathered.get_shape().as_list()[0], -1))
                layer_patches = tf.extract_image_patches(tf.squeeze(layer_pad,axis=4),
                                        [1,kernel_s,kernel_s,1],
                                        [1,1,1,1],
                                        [1,1,1,1],
                                        padding="VALID")
                layer_size = tf.shape(layer_patches)
                layer_patches = tf.reshape(layer_patches,(-1, layer_size[3]))
                style_norm = tf.cast(layer_size[1]*layer_size[2]*layer_size[3],dtype=tf.float32)

#                gram1 = tf.matmul(tf.transpose(layer_patches), layer_patches) / style_norm
#                gram2 = tf.matmul(tf.transpose(style_filters_tf_gathered), style_filters_tf_gathered) / style_norm
#                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram1- gram2))
                
                loss_ = tf.reduce_mean(tf.reduce_mean(tf.pow(layer_patches-style_filters_tf_gathered, 2),axis=1)*tf.stop_gradient(max_filter_response_weight))
                style_losses.append(style_layers_weights[style_layer] * 2 * loss_)
                
                
                
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
#                print(str(max_filter_response_weight.eval()))
#                print(' ')
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    yield (
                        (None if last_step else i),
                        img_out
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb
