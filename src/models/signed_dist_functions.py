import numpy as np
import tensorflow as tf



def length(v):
    return tf.sqrt(tf.reduce_sum(tf.square(v),-1,keep_dims=True))    
    
    
   
#float sdTorus( vec3 p, vec2 t )
#{
#  vec2 q = vec2(length(p.xz)-t.x,p.y);
#  return length(q)-t.y;
#}



    
def torus(grid,s_):
    
    xz = length(tf.concat((grid[:,:,:,0:1],grid[:,:,:,2:3]),axis=-1))-0.5
    qq = tf.concat((xz,grid[:,:,:,1:2]),axis=-1)
    return tf.squeeze(length(qq) - 0.3,axis=-1)
    


    
def sphere(grid,theta):
    return tf.sqrt(tf.reduce_sum(tf.pow(grid-theta[1],2),axis=-1))- theta[0]
    
def box(grid,theta):
    box = tf.abs(grid-theta[1]) - tf.reshape(theta[0],(1,1,1,1))
    return tf.reduce_max(box,axis=-1)
    

    
def point(grid):
    return tf.sqrt(tf.reduce_sum(tf.pow(grid,2),axis=-1,keep_dims=True))
    

def plane(grid,v,b):
    vv = np.expand_dims(np.expand_dims(np.array(v),-1),-1)
    vv = vv/np.sqrt(np.sum(np.square(vv),0,keepdims=True))
    bb = np.expand_dims(np.expand_dims(np.array(b),-1),-1)
    return tf.reduce_sum(grid*vv+bb,axis=0)
    


#def cylinder(grid,s1,s2,s3):
#    v0 = tf.sqrt(tf.pow(grid[0,:,:],2)+tf.pow(grid[2,:,:],2))
#    v1 = tf.sqrt(tf.pow(s1,2)+tf.pow(s2,2))
#    length(v0-v1)-s3
#    return length(v0-v1)-s3
 









def union(d1,d2):
    return tf.reduce_min(tf.stack((d1,d2),0),0)


def smooth_union(d1,d2,k):
    res  = tf.exp( -k*d1 ) + tf.exp( -k*d2 );
    return -tf.log( res )/k    
#    res1 = tf.pow( d1,k )
#    res2 = tf.pow( d2,k );
#    return tf.pow((res1*res2/(res1+res2)),1/k )


def intersection(d1,d2):
    return tf.reduce_max(tf.stack((d1,d2),0),0)








