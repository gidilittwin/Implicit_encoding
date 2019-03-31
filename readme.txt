This is the Tensorflow implementation of "Deep Meta Functionals for Shape Representation" which includes the basic graph definition, training and testing on the ShapeNet dataset.

Requirements:
Python2.7+ with Numpy
Tensorflow (tested for version 1.4.0+)
This code was tested on Ubuntu 14.04. with Python 2.7, TensorFlow 1.4.0, CUDA 8.0

Training:
python train.py

Training is controlled by the following (main) arguments:
experiment_name     -experiment name
model_params_path   -model architecture (defined by json file)
grid_size           -size of voxel grid - 32/64/256 etc
img_size            -size of input image in uv
im_per_obj          -number of renders per object in dataset
batch_size          -training batch size
test_size           -testing batch size
multi_image         -train in a multi-view scenario
multi_image_views   -in case of multiview, number of views in training
batch_norm          -use batch norm
shuffle_rgb         -rgb shuffle augmentation of the training data
rgba                -train on rgba data
num_samples         -number of boundary samples per example
global_points       -number of uniformly sampled points  
noise_scale         -boundary sample variance
learning_rate       -learning rate
finetune            -use pretrained model
path                -path to training data




