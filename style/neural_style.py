# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc
#import pylab as plt
import matplotlib.pyplot as plt

from stylize_Li_mask import stylize
#from stylize_Li import stylize
#from stylize import stylize

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
CONTENT_WEIGHT = 5e1#5e1
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2#5e2
TV_WEIGHT = 1e2#1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
#VGG_PATH = 'vgg-face.mat'
POOLING = 'max'
BASE = '/Users/gidilittwin/neural-style-master/examples/'
OUT_PATh = BASE+'results/'
output = ''
content = BASE+'CONTENT/female-blank-mask.jpg'
styles = [BASE+'CONTENT/stat2.jpg']
initial = content
width = 224 
checkpoint_iterations = 10 

# mask_
#center_l = [40./128.* width,57./128.* width]
#center_r = [87./128.* width,55./128.* width]
# stat2 mouth
#center_l = [96./256.* width,181./256.* width]
#center_r = [134./256.* width,185./256.* width]
# stat2 eyes
#center_l = [60./224.* width,100./224.* width]
#center_r = [155./224.* width,100./224.* width]
# emale-blank-mask mouth
center_l = [92./224.* width,186./224.* width]
center_r = [118./224.* width,186./224.* width]


sigma_x = width/8
sigma_y = width/12.


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


#def main():
parser = build_parser()
options = parser.parse_args()

#if not os.path.isfile(options.network):
#    parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

content_image = imread(options.content)
style_images = [imread(style) for style in options.styles]

width = options.width
if width is not None:
    new_shape = (int(math.floor(float(content_image.shape[0]) /
            content_image.shape[1] * width)), width)
    content_image = scipy.misc.imresize(content_image, new_shape).astype(np.float32)
    x, y = np.meshgrid(np.linspace(0, new_shape[1]-1, new_shape[1]), np.linspace(0, new_shape[0]-1, new_shape[0]))
    content_mask = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-((x-center_l[0])**2/(2*sigma_x**2)+ (y-center_l[1])**2/(2*sigma_y**2))) 
                  + 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-((x-center_r[0])**2/(2*sigma_x**2)+ (y-center_r[1])**2/(2*sigma_y**2))))
    content_mask = (content_mask/np.max(content_mask)).astype(np.float32)
    content_mask_ = (content_mask>0.95).astype(np.float32)
#    content_mask_ = np.logical_and(content_mask>0.6, content_mask<0.7).astype(np.float32)
    plt.imshow(content_mask_)
    plt.show()
    plt.pause(0.01)

    
    
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

initial = options.initial
if initial is not None:
    initial = scipy.misc.imresize(imread(initial), content_image.shape[:2]).astype(np.float32)
    # Initial guess is specified, but not noiseblend - no noise should be blended
    if options.initial_noiseblend is None:
        options.initial_noiseblend = 0.0
else:
    # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
    if options.initial_noiseblend is None:
        options.initial_noiseblend = 1.0
    if options.initial_noiseblend < 1.0:
        initial = content_image
        


if options.checkpoint_output and "%s" not in options.checkpoint_output:
    parser.error("To save intermediate images, the checkpoint output "
                 "parameter must contain `%s` (e.g. `foo%s.jpg`)")

for iteration, image in stylize(
    network=options.network,
    initial=initial,
    initial_noiseblend=options.initial_noiseblend,
    content=content_image,
    content_mask = content_mask_,
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
#        if iteration is not None:
##            if options.checkpoint_output:
##            output_file = options.checkpoint_output % iteration
#            output_file = 10 % iteration
#        else:
#            output_file = options.output
#        if output_file:
    output_file = OUT_PATh+str(iteration)+'.jpg'
    imsave(output_file, combined_rgb)



#if __name__ == '__main__':
#    main()
