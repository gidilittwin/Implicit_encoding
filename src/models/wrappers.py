import tensorflow as tf
from src.models import scalar_functions as SF
from src.models import feature_extractor as CNN


def g_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_shape(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   

def g_wrapper2(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = CNN.mlp(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   
    
def g_conv_wrapper(coordinates,args_):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        evaluated_function = SF.deep_sensor(coordinates,args_[0],args_[1],args_[2])
        return evaluated_function   
    
def f_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.regressor(current,args_)

def f3_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return current
    
def r_wrapper(encoding,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.render(encoding,args_)
        return current
    
def m_wrapper(ids,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.multiplexer(ids,args_)
        return CNN.regressor(current,args_)    

def f2_wrapper(image,args_):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_50(image,args_)
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_) 

def f_conv_wrapper(image,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        current = CNN.resnet_config(image,args_)
        return CNN.conv_regressor(current,args_)
    
    
def injection_wrapper(current,args_):
    with tf.variable_scope('2d_cnn_model',reuse=tf.AUTO_REUSE):
        return CNN.regressor(current,args_)
