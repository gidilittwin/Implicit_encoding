
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm




def unravel_index(indices, shape, Type=tf.int32):
#    indices = tf.transpose(indices,(1,0))
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    shape = tf.cast(shape, tf.float32)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    strides = tf.cast(strides, Type)
    strides_shifted = tf.cast(strides_shifted, Type)
    def even():
        rem = indices - (indices // strides) * strides
        return rem // strides_shifted
    def odd():
        div = indices // strides_shifted
        return div - (div // strides) * strides
    rank = tf.rank(shape)
    return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)

def ravel_xy(bxy):
    shape = bxy.get_shape().as_list()
    x = tf.slice(bxy,[0,0],[-1,1])
    y = tf.slice(bxy,[0,1],[-1,1])
    return y + shape[1]*x 

def ravel_index(bxyz, shape):
    b = tf.slice(bxyz,[0,0],[-1,1])
    x = tf.slice(bxyz,[0,1],[-1,1])
    y = tf.slice(bxyz,[0,2],[-1,1])
    z = tf.slice(bxyz,[0,3],[-1,1])
    return z + shape[3]*y + (shape[2] * shape[1])*x + (shape[3] * shape[2] * shape[1])*b

def ravel_index_bxy(bxyz, shape):
    b = tf.slice(bxyz,[0,0],[-1,1])
    x = tf.slice(bxyz,[0,1],[-1,1])
    y = tf.slice(bxyz,[0,2],[-1,1])
    return y + (shape[2])*x + (shape[2] * shape[1])*b


def BatchNorm(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,  
                lambda: batch_norm(inputT, is_training=True,  
                                   center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),  
                lambda: batch_norm(inputT, is_training=False,  
                                   center=True, scale=True, decay=0.9, updates_collections=None, scope=scope, reuse = True))  


def lrelu(x, leak=0.2, name="LRelU"):
   with tf.variable_scope(name):
       return tf.maximum(x, leak*x)        

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, bn = True):
   shape = input_.get_shape().as_list()
   with tf.variable_scope(scope or "Linear"):
       matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
       bias = tf.get_variable("bias", [output_size],
           initializer=tf.constant_initializer(bias_start))
       output = tf.matmul(input_, matrix) + bias
       if with_w:
           return output, matrix, bias
       else:
           return output


def CONV2D(shape):
   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=np.sqrt(2)/np.sqrt(shape[0]*shape[1]*shape[2])))
   conv_biases = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases



def CONV2D_I(shape):
   Ident = np.zeros(shape)
   for i in range(shape[3]):
       Ident[0,0,i % shape[2],i] = 1
   conv_weights = tf.get_variable(name='weights', initializer = Ident.astype('float32'),trainable = False, dtype='float32')
   conv_biases = tf.get_variable(name='biases', shape = [shape[-1]], initializer=tf.constant_initializer(0.0), trainable=False)
   return conv_weights, conv_biases 
    
def CONV3D_I(shape):
   Ident = np.zeros(shape)
   for i in range(shape[4]):
       Ident[0,0,0,i % shape[3],i] = 1
   conv_weights = tf.get_variable(name='weights', initializer = Ident.astype('float32'),trainable = False, dtype='float32')
   conv_biases = tf.get_variable(name='biases', shape = [shape[-1]], initializer=tf.constant_initializer(0.0), trainable=False)
   return conv_weights, conv_biases     

def CONV3D_T_I(shape):
   Ident = np.zeros(shape)
   for i in range(shape[4]):
       Ident[0,0,0,i % shape[3],i] = 1
   conv_weights = tf.get_variable(name='weights', initializer = Ident.astype('float32'),trainable = False, dtype='float32')
   conv_biases = tf.get_variable(name='biases', shape = [shape[-2]], initializer=tf.constant_initializer(0.0), trainable=False)
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases     

def cell(in_node, k, M, N, mode_node, var_list, SCOPE, downsample=False):
    # Conv 1
    with tf.variable_scope(SCOPE+'1') as scope:
        conv1_w, conv1_b = CONV2D([k,k,M,N])
        if downsample==False:
            conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 1, 1, 1],padding='SAME')
        else:
            conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, 2, 2, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b) 
        conv1 = BatchNorm(conv1,mode_node,scope)
        relu1 = tf.nn.relu(conv1)
    # Conv 2
    with tf.variable_scope(SCOPE+'2') as scope:
        conv2_w, conv2_b = CONV2D([k,k,N,N])
        conv2 = tf.nn.conv2d(relu1,conv2_w,strides=[1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b) 
        conv2 = BatchNorm(conv2,mode_node,scope)
    # Identity connection
    with tf.variable_scope(SCOPE+'3') as scope:
        if downsample==False:
            out = conv2 + in_node
        else:
            conv3_w, conv3_b = CONV2D_I([3,3,M,N])
            reshape = tf.nn.conv2d(in_node,conv3_w,strides=[1, 2, 2, 1],padding='SAME')
            out = reshape + conv2 
        out = tf.nn.relu(out)
    return out

    
def cell3D_res(in_node, k, M, N, mode_node, stride, SCOPE):
    with tf.variable_scope(SCOPE+'_1') as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV3D([k,k,k,M,N])
#        conv1 = tf.nn.conv3d(relu1,conv1_w,strides=[1, stride, stride, stride, 1],padding='SAME')
        conv1 = tf.nn.conv3d(relu1,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv1 = tf.nn.max_pool3d(conv1,[1,stride,stride,stride,1],[1,stride,stride,stride,1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        batch2 = BatchNorm(conv1,mode_node,scope)
        relu2 = tf.nn.relu(batch2)
        conv2_w, conv2_b = CONV3D([k,k,k,N,N])
        conv2 = tf.nn.conv3d(relu2,conv2_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        print(conv2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
        if M==N and stride==1:
            out = conv2 + in_node
        else:        
            conv3_w, conv3_b = CONV3D_I([k,k,k,M,N])
#            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, stride, stride, stride, 1],padding='SAME')
            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, 1, 1, 1, 1],padding='SAME')
            reshape = tf.nn.max_pool3d(reshape,[1,stride,stride,stride,1],[1,stride,stride,stride,1],padding='SAME')
            out = reshape + conv2 
    return out

   
def cell3D_res_regular(in_node, k, M, N, mode_node, stride, SCOPE):
    with tf.variable_scope(SCOPE+'_1') as scope:
        conv1_w, conv1_b = CONV3D([k,k,k,M,N])
        conv1 = tf.nn.conv3d(in_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv1 = tf.nn.max_pool3d(conv1,[1,stride,stride,stride,1],[1,stride,stride,stride,1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        batch1 = BatchNorm(conv1,mode_node,scope)
        relu1 = tf.nn.relu(batch1)        
        print(relu1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        conv2_w, conv2_b = CONV3D([k,k,k,N,N])
        conv2 = tf.nn.conv3d(relu1,conv2_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        batch2 = BatchNorm(conv2,mode_node,scope)
        print(batch2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
        if M==N and stride==1:
            out = batch2 + in_node
        else:        
            conv3_w, conv3_b = CONV3D_I([k,k,k,M,N])
#            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, stride, stride, stride, 1],padding='SAME')
            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, 1, 1, 1, 1],padding='SAME')
            reshape = tf.nn.max_pool3d(reshape,[1,stride,stride,stride,1],[1,stride,stride,stride,1],padding='SAME')
            out = reshape + batch2 
    return tf.nn.relu(out)  

    
def CONV3D(shape):
   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=np.sqrt(2)/np.sqrt(shape[0]*shape[1]*shape[2]*shape[3])))
#   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=0.001))
   conv_biases = tf.get_variable('biases',[shape[-1]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases 

def CONV3D_T(shape):
   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=np.sqrt(2)/np.sqrt(shape[0]*shape[1]*shape[2]*shape[3])))
#   conv_weights = tf.get_variable('weights',shape, initializer = tf.random_normal_initializer( stddev=0.001))
   conv_biases = tf.get_variable('biases',[shape[-2]], initializer=tf.constant_initializer(0.0))
   tf.add_to_collection('l2_res',(tf.nn.l2_loss(conv_weights)))
   return conv_weights, conv_biases 

def cell3D(in_node, k, M, N, mode_node, stride, SCOPE, padding='SAME'):
    with tf.variable_scope(SCOPE) as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV3D([k,k,k,M,N])
        conv1 = tf.nn.conv3d(relu1,conv1_w,strides=[1, stride, stride, stride, 1],padding=padding)
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())        
        return conv1

def cell3D_regular(in_node, k, M, N, mode_node, stride, SCOPE, padding='SAME', act=True):
    with tf.variable_scope(SCOPE) as scope:
        conv1_w, conv1_b = CONV3D([k,k,k,M,N])
        conv1 = tf.nn.conv3d(in_node,conv1_w,strides=[1, stride, stride, stride, 1],padding=padding)
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        if act==True:
            conv1 = BatchNorm(conv1,mode_node,scope)
            conv1 = tf.nn.relu(conv1)
        print(conv1.get_shape())        
        return conv1
    
def leaky_cell3D(in_node, k, M, N, mode_node, stride, SCOPE, padding='SAME'):
    with tf.variable_scope(SCOPE) as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = lrelu(batch1)
        conv1_w, conv1_b = CONV3D([k,k,k,M,N])
        conv1 = tf.nn.conv3d(relu1,conv1_w,strides=[1, stride, stride, stride, 1],padding=padding)
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())        
        return conv1
    
    
def cell2D(in_node, k1, k2, M, N, mode_node, stride, SCOPE, padding='SAME', bn=True, act=True):
    with tf.variable_scope(SCOPE) as scope:
        conv1_w, conv1_b = CONV2D([k1,k2,M,N])
        conv1 = tf.nn.conv2d(in_node,conv1_w,strides=[1, stride, stride, 1],padding=padding)
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        if bn==True:
            conv1 = BatchNorm(conv1,mode_node,scope)
        if act==True:
            conv1 = tf.nn.relu(conv1)
        print(conv1.get_shape())        
        return conv1
        
def cell1D(in_node,output_size, mode_node, SCOPE=None, stddev=0.02, bias_start=0.0, with_act=True, with_bn=True,act_type=lrelu):
    with tf.variable_scope(SCOPE) as scope:
        shape = in_node.get_shape().as_list()
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
           initializer=tf.constant_initializer(bias_start))
        tf.add_to_collection('l2_res',(tf.nn.l2_loss(matrix)))
        tf.add_to_collection('l2_res',(tf.nn.l2_loss(bias)))
        output = tf.matmul(in_node, matrix) + bias
        if with_bn:
            output = BatchNorm(output,mode_node,scope)
        if with_act:
            output = lrelu(output)
        print(output.get_shape())        
        return output

def skip3D(in_node, k, M, N, mode_node, stride, SCOPE):
        s1 = cell3D(in_node, k, M, N, mode_node, stride, SCOPE+'_1')
        s1 = cell3D(s1, k, N, N, mode_node, 1, SCOPE+'_2')
        print(s1.get_shape())        
        return s1
        
def cell_deconv_3D(in_node, k, M, N, output_shape, mode_node, stride, SCOPE):
    with tf.variable_scope(SCOPE) as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV3D_T([k,k,k,N,M])
        conv1 = tf.nn.conv3d_transpose(relu1,conv1_w,output_shape=output_shape,strides=[1, stride, stride, stride, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b) 
        print(conv1.get_shape())        
        return conv1

def cell2D_res(in_node, k, M, N, mode_node, stride, SCOPE):
    with tf.variable_scope(SCOPE+'_1') as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV2D([k,k,M,N])
        conv1 = tf.nn.conv2d(relu1,conv1_w,strides=[1, stride, stride, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        batch2 = BatchNorm(conv1,mode_node,scope)
        relu2 = tf.nn.relu(batch2)
        conv2_w, conv2_b = CONV2D([k,k,N,N])
        conv2 = tf.nn.conv2d(relu2,conv2_w,strides=[1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        print(conv2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
        if M==N and stride==1:
            out = conv2 + in_node
        else:        
            conv3_w, conv3_b = CONV2D_I([k,k,M,N])
            reshape = tf.nn.conv2d(in_node,conv3_w,strides=[1, stride, stride, 1],padding='SAME')
            out = reshape + conv2 
    return out
  
def cell3D_res_deconv(in_node, k, M, N, mode_node, stride, output_shape, SCOPE):
    with tf.variable_scope(SCOPE+'_1') as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        conv1_w, conv1_b = CONV3D([k,k,k,M,M])
        conv1 = tf.nn.conv3d(relu1,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        print(conv1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        batch2 = BatchNorm(conv1,mode_node,scope)
        relu2 = tf.nn.relu(batch2)
        conv2_w, conv2_b = CONV3D_T([k,k,k,N,M])
        conv2 = tf.nn.conv3d_transpose(relu2,conv2_w,output_shape=output_shape,strides=[1, stride, stride, stride, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b) 
        print(conv2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
#        if M==N:
#            out = conv2 + in_node
#        else:        
#            conv3_w, conv3_b = CONV3D_T_I([k,k,k,N,M])
##            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, stride, stride, stride, 1],padding='SAME')
#            reshape = tf.nn.conv3d_transpose(in_node,conv3_w,output_shape=output_shape,strides=[1, stride, stride, stride, 1],padding='SAME')
#            out = reshape + conv2 
        out = conv2
    return out   


def cell3D_res_deconv_regular(in_node, k, M, N, mode_node, stride, output_shape, SCOPE):
    with tf.variable_scope(SCOPE+'_1') as scope:

        conv1_w, conv1_b = CONV3D([k,k,k,M,M])
        conv1 = tf.nn.conv3d(in_node,conv1_w,strides=[1, 1, 1, 1, 1],padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        batch1 = BatchNorm(conv1,mode_node,scope)
        relu1 = tf.nn.relu(batch1)        
        print(relu1.get_shape())
    with tf.variable_scope(SCOPE+'_2') as scope:
        conv2_w, conv2_b = CONV3D_T([k,k,k,N,M])
        conv2 = tf.nn.conv3d_transpose(relu1,conv2_w,output_shape=output_shape,strides=[1, stride, stride, stride, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b) 
        batch2 = BatchNorm(conv2,mode_node,scope)
        relu2 = tf.nn.relu(batch2)        
        print(relu2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
#        if M==N:
#            out = conv2 + in_node
#        else:        
#            conv3_w, conv3_b = CONV3D_T_I([k,k,k,N,M])
##            reshape = tf.nn.conv3d(in_node,conv3_w,strides=[1, stride, stride, stride, 1],padding='SAME')
#            reshape = tf.nn.conv3d_transpose(in_node,conv3_w,output_shape=output_shape,strides=[1, stride, stride, stride, 1],padding='SAME')
#            out = reshape + conv2 
        out = relu2
    return out   


    





def cell2D_gated(in_node, k, M, N, mode_node, stride, SCOPE, global_cond=None):
    with tf.variable_scope(SCOPE+'_1') as scope:
        batch1 = BatchNorm(in_node,mode_node,scope)
        relu1 = tf.nn.relu(batch1)
        with tf.variable_scope('filter') as scope:
            conv1_fw, conv1_fb = CONV2D([k,k,M,N])
            conv1f = tf.nn.conv2d(relu1,conv1_fw,strides=[1, stride, stride, 1],padding='SAME')
            conv1f = tf.nn.bias_add(conv1f, conv1_fb)
#            if global_cond!=None:
#                with tf.variable_scope('conditioned') as scope:
#                    conv1_fw_cond, conv1_fb_cond = CONV2D([1,1,20,N])
#                    conv1f_cond = tf.nn.conv2d(global_cond,conv1_fw_cond,strides=[1, 1, 1, 1],padding='SAME')
#                    conv1f_cond = tf.nn.bias_add(conv1f_cond, conv1_fb_cond) 
#                    conv1f = conv1f + conv1f_cond
            conv1f = BatchNorm(conv1f,mode_node,scope)
        with tf.variable_scope('gate') as scope:
            conv1_gw, conv1_gb = CONV2D([1,1,M,N])
            conv1g = tf.nn.conv2d(relu1,conv1_gw,strides=[1, stride, stride, 1],padding='SAME')
            conv1g = tf.nn.bias_add(conv1g, conv1_gb) 
            if global_cond!=None:
                with tf.variable_scope('conditioned') as scope:
                    conv1_gw_cond, conv1_gb_cond = CONV2D([1,1,20,N])
                    conv1g_cond = tf.nn.conv2d(global_cond,conv1_gw_cond,strides=[1, 1, 1, 1],padding='SAME')
                    conv1g_cond = tf.nn.bias_add(conv1g_cond, conv1_gb_cond) 
                    conv1g = conv1g_cond + conv1g           
            conv1g = BatchNorm(conv1g,mode_node,scope)
    with tf.variable_scope(SCOPE+'_2') as scope:
        relu2 = tf.nn.relu(conv1f)*tf.sigmoid(conv1g)
        conv2_w, conv2_b = CONV2D([k,k,N,N])
        conv2 = tf.nn.conv2d(relu2,conv2_w,strides=[1, 1, 1, 1],padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        print(conv2.get_shape())         
    with tf.variable_scope(SCOPE+'_skip') as scope:
        if M==N and stride==1:
            out = conv2 + in_node
        else:        
            conv3_w, conv3_b = CONV2D_I([k,k,M,N])
            reshape = tf.nn.conv2d(in_node,conv3_w,strides=[1, stride, stride, 1],padding='SAME')
            out = reshape + conv2 
    return out