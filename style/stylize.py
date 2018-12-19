import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
import matplotlib.pyplot as plt
from PIL import Image

CONTENT_LAYERS = ('relu3_1', 'relu4_1')
#CONTENT_LAYERS = ('conv3_1',)
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce


def stylize(Ray_render,ray_steps,reset_opp,session, network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
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
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)]).astype(np.float32)
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    image = initial - tf.cast(tf.reshape(vgg_mean_pixel,(1,1,1,3)),tf.float32)
    net   = vgg.net_preloaded(vgg_weights, image, pooling)

    # content loss
    content_layers_weights = 1/(1.0*len(CONTENT_LAYERS))


    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(content_layers_weights * content_weight * (2 * tf.nn.l2_loss(
                net[content_layer] - content_features[content_layer]) /
                content_features[content_layer].size))
    content_loss += reduce(tf.add, content_losses)

    # style loss
    style_loss = 0
    # overall loss
    loss = content_loss + style_loss #+ tv_loss









    # optimizer setup
    render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'parameters')
    with tf.variable_scope('OPTIMIZATION',reuse=tf.AUTO_REUSE):
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss,var_list=render_vars)
    session.run(tf.initialize_all_variables())
    Ray_render.trace(session,ray_steps,reset_opp,num_steps=50)
    
    
#    evals_ = session.run(tf.squeeze(initial,axis=0)) # <= returns jpeg data you can write to disk


    def print_progress():
        stderr.write('  content loss: %g\n' % content_loss.eval(session=session))
#        stderr.write('    style loss: %g\n' % style_loss.eval(session=session))
        stderr.write('    total loss: %g\n' % loss.eval(session=session))
    print_progress()
        
        
        
#    aa= np.squeeze(net[CONTENT_LAYERS[0]].eval(session=session),0)
#    bb = np.squeeze(content_features[CONTENT_LAYERS[0]],0)
#    pic_aa=np.squeeze(content)
#    pic_bb=np.squeeze(initial.eval(session=session),0)
#    
#    fig = plt.figure(1)
#    ax2 = fig.add_subplot(1, 1, 1)
#    ax2.imshow(pic_aa)
#    
#    fig = plt.figure(2)
#    ax2 = fig.add_subplot(1, 1, 1)
#    ax2.imshow(pic_bb)    
#    aa.aa=1 
    
    # optimization
    stderr.write('Optimization started...\n')
    if (print_iterations and print_iterations != 0):
        print_progress()
    for i in range(iterations):
        stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))

        train_step.run(session=session)
        Ray_render.trace(session,ray_steps,reset_opp,num_steps=50)
        last_step = (i == iterations - 1)
        print_progress()
        
        
        if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
            image_  = image.eval(session=session)
            img_out = vgg.unprocess(image_.reshape(shape[1:]), vgg_mean_pixel)

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
