
import tensorflow as tf
import numpy as np
import scipy
from src.utilities import mesh_handler as MESHPLOT
from src.utilities import raytrace as RAY
from src.models import signed_dist_functions as SDF
from src.models import voxels as CNN
import provider
import matplotlib.pyplot as plt
from src.utilities import Voxels as VOXELS
from style.stylize import stylize as stylize
from PIL import Image
from style import vgg
from argparse import ArgumentParser


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

CONTENT_WEIGHT = 1#5e1
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 0#5e2
TV_WEIGHT = 0#1e2
STYLE_LAYER_WEIGHT_EXP = 1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'Data/imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
BASE = '/Users/gidilittwin/Dropbox/Thesis/Implicit_Encoding/Data/origin/'
OUT_PATh = BASE+'results/'
output = ''
content = BASE+'CONTENT/image0_0.jpg'
styles = []
initial = content
width = 224#224 
checkpoint_iterations = 1 
    
grid_size   = 32
canvas_size = grid_size
num_points  = 100000
BATCH_SIZE  = 1



  

#%% Function wrappers   
with tf.variable_scope('mode_node',reuse=tf.AUTO_REUSE):
    mode_node = tf.get_variable(name='mode_node',initializer=True,dtype=tf.bool)
   
with tf.variable_scope('parameters_target',reuse=tf.AUTO_REUSE):
    rad    = tf.get_variable(initializer=np.array([0.8],dtype=np.float32), name='radius',trainable=False)
    offset = tf.get_variable(initializer=np.array([[[[0.0,0.0,0.0]]]],dtype=np.float32), name='center',dtype=tf.float32,trainable=False)
theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'parameters_target')
 
with tf.variable_scope('parameters_opt',reuse=tf.AUTO_REUSE):
    rad_opt    = tf.get_variable(initializer=np.array([0.8],dtype=np.float32), name='radius',trainable=False)
    offset_opt = tf.get_variable(initializer=np.array([[[[0.1,0.1,-1.]]]],dtype=np.float32), name='center',dtype=tf.float32,trainable=True)
theta_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'parameters_opt')
      
  
def target_wrapper(coordinates,args_):
    with tf.name_scope('model'):
        evaluated_function = SDF.sphere(coordinates,args_)
#        evaluated_function = SDF.box(coordinates,args_)
        return evaluated_function



with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
    Ray_render = RAY.Raycast(BATCH_SIZE,resolution=(400,400),sky_color=(127.5, 127.5, 127.5))
    Ray_render.add_lighting(position=[1.0, 0.0, 1.0],color=[255., 120., 20.],ambient_color = [15., 244., 66.])
    Ray_render.add_camera(camera_position=[0.0, 0.0, 1.0], lookAt=(0,0,0),focal_length=1,name='camera_1')
#Ray_render_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'display_tracer')

with tf.variable_scope('optimized_tracer',reuse=tf.AUTO_REUSE):
    Ray_render_opt = RAY.Raycast(BATCH_SIZE,resolution=(400,400),sky_color=(127.5, 127.5,127.5))
    Ray_render_opt.add_lighting(position=[1.0, 0.0, 1.0],color=[255., 120., 20.],ambient_color = [15., 244., 66.])
    Ray_render_opt.add_camera(camera_position=[0.0, 0.0, 1.0], lookAt=(0,0,0),focal_length=1,name='camera_1')
#Ray_render_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'optimized_tracer')




session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(mode_node.assign(True)) 
ray_steps,reset_opp,evals             = Ray_render.eval_func(model_fn = target_wrapper,     args=theta,
                                                             step_size=0.004, 
                                                             epsilon = 0.0001, 
                                                             temp=1., 
                                                             ambient_weight=0.0)
ray_steps_opt,reset_opp_opt,evals_opt = Ray_render_opt.eval_func(model_fn = target_wrapper, args=theta_opt,
                                                             step_size=0.004, 
                                                             epsilon = 0.0001, 
                                                             temp=1., 
                                                             ambient_weight=0.0)
Ray_render.trace(session,ray_steps,reset_opp,num_steps=50)
Ray_render_opt.trace(session,ray_steps_opt,reset_opp_opt,num_steps=50)

evals_             = session.run(evals) # <= returns jpeg data you can write to disk
target_render      = evals_['raw_images'][0]
target_normals     = (np.transpose(evals_['normals'][0],(1,2,0))+1.)/2
target_incident    = evals_['incidence_angles'][0]

initial_image      = session.run(evals_opt['raw_images'][0]) # <= returns jpeg data you can write to disk



fig = plt.figure()
title('target_image')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_render.astype(np.uint8))

fig = plt.figure()
title('target_image_normals')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_normals)

fig = plt.figure()
title('image_incident')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(target_incident)


fig = plt.figure()
title('initial_image')
ax2 = fig.add_subplot(1, 1, 1)
ax2.imshow(initial_image.astype(np.uint8))




#%% OPTIMIZATION
image_opt     = evals_opt['raw_images'][0]
target_image  = tf.placeholder('float', shape=target_render.shape)
loss  = tf.reduce_mean(tf.square(image_opt - target_image))
error = tf.abs(image_opt - target_image)/2.


# optimizer setup
LEARNING_RATE = 0.001
with tf.variable_scope('OPTIMIZATION',reuse=tf.AUTO_REUSE):
    global_step= tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.1)
#    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grad = tf.gradients(loss, [offset_opt])
#    grad, _ = tf.clip_by_global_norm(grad, 1.0)
    train_step = optimizer.apply_gradients([(grad[0], offset_opt)])
    
session.run(tf.initialize_all_variables())
for i in range(10000):
    print(i)
    Ray_render_opt.trace(session,ray_steps_opt,reset_opp_opt,num_steps=50)
    _ ,loss_,error_,evals_opt_= session.run([train_step,loss,error,evals_opt],feed_dict={target_image:target_render})
    print('loss = '+   str(loss_))
#    with open('/Users/gidilittwin/Dropbox/Thesis/Implicit_Encoding/Data/origin/results/image'+str(0)+'_'+str(i)+'.jpg', 'w') as fd:
#        fd.write(evals_opt_['renders'][0])
    if i==0:
        fig = plt.figure(1)
        image = plt.imshow(error_.astype(np.uint8))
    else:
        image.set_data(error_.astype(np.uint8))
        plt.pause(0.05)







fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
title('opt_image')
ax2.imshow(evals_opt_['raw_images'][0].astype(np.uint8))

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
title('target_image')
ax2.imshow(evals_['raw_images'][0].astype(np.uint8))

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
title('raw_image_filtered')
ax2.imshow(error_.astype(np.uint8))

aa.aa=1




#%%  Imagenet
# default arguments




def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', default=content)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', default=styles)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', default=output)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS',default=checkpoint_iterations)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH',default=width)
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL',default=initial)
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    return parser


parser = build_parser()
options = parser.parse_args()
content_image = imread(options.content)
style_images = [imread(style) for style in options.styles]   
target_shape = content_image.shape
for i in range(len(style_images)):
    style_scale = STYLE_SCALE
    if options.style_scales is not None:
        style_scale = options.style_scales[i]
    style_images[i] = scipy.misc.imresize(style_images[i], style_scale *target_shape[1] / style_images[i].shape[1]).astype(np.float32)
style_blend_weights = options.style_blend_weights
if style_blend_weights is None:
    # default is equal weights
    style_blend_weights = [1.0/len(style_images) for _ in style_images]
else:
    total_blend_weight = sum(style_blend_weights)
    style_blend_weights = [weight/total_blend_weight
                           for weight in style_blend_weights]



image_rendered = tf.expand_dims(evals['raw_images'][0],0)
for iteration, image in stylize(
    Ray_render=Ray_render,
    ray_steps=ray_steps,
    reset_opp=reset_opp,
    session=session,
    network=options.network,
    initial=image_rendered,
    initial_noiseblend=0.0,
    content=content_image,
    styles=style_images,
    preserve_colors=options.preserve_colors,
    iterations=options.iterations,
    content_weight=options.content_weight,
    content_weight_blend=options.content_weight_blend,
    style_weight=options.style_weight,
    style_layer_weight_exp=options.style_layer_weight_exp,
    style_blend_weights=style_blend_weights,
    tv_weight=options.tv_weight,
    learning_rate=options.learning_rate,
    beta1=options.beta1,
    beta2=options.beta2,
    epsilon=options.epsilon,
    pooling=options.pooling,
    print_iterations=options.print_iterations,
    checkpoint_iterations=options.checkpoint_iterations
    ):
    
    output_file = None
    combined_rgb = image
    output_file = OUT_PATh+str(iteration)+'.jpg'
    imsave(output_file, combined_rgb)

