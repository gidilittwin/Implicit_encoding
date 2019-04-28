
import tensorflow as tf
import numpy as np
import os
import glob
import random





def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def dataset_builder_fn(path,batch):
    example = tf.train.Example(features=tf.train.Features(feature={
        'voxels': _bytes_feature(batch['voxels'][0,:,:,:].tostring()),
        'images': _bytes_feature(batch['images'].astype(np.uint8).tostring()),
        'classes': _bytes_feature(batch['classes'][0,0].tostring()),
        'ids': _bytes_feature(batch['ids'][0,0].tostring()),
        'vertices': _bytes_feature((batch['vertices'][0,:,:,:]*10).astype(np.int32).tostring()),
        }))
    dir_name = path+str(batch['classes'][0,0])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)    
    tfrecords_filename = dir_name+'/'+str(batch['ids'][0,0])+'.tfrecords'
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(tfrecords_filename, options=options) as writer:
        writer.write(example.SerializeToString())
        


def dataset_input_fn(filenames,batch_size,epochs,shuffle,img_size,im_per_obj,grid_size,num_samples,shuffle_size):
  dataset = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
  def parser(record):
    keys_to_features = {
        "voxels":   tf.FixedLenFeature((), tf.string, default_value=""),
        "images":   tf.FixedLenFeature((), tf.string, default_value=""),
        "classes":  tf.FixedLenFeature((), tf.string, default_value=""),
        "ids":      tf.FixedLenFeature((), tf.string, default_value=""),
        "vertices": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed   = tf.parse_single_example(record, keys_to_features)
    parsed['images']   = tf.reshape(tf.decode_raw(parsed['images'],out_type=tf.uint8),(im_per_obj,img_size,img_size,4))
    parsed['voxels']   = tf.reshape(tf.decode_raw(parsed['voxels'],out_type=tf.uint8),(grid_size,grid_size,grid_size))
    parsed['classes']  = tf.reshape(tf.decode_raw(parsed['classes'],out_type=tf.int64),(1,))
    parsed['ids']      = tf.reshape(tf.decode_raw(parsed['ids'],out_type=tf.int64),(1,))
    parsed['vertices'] = tf.cast(tf.reshape(tf.decode_raw(parsed['vertices'],out_type=tf.int32),(num_samples,3)),tf.float32)/10.
    return parsed
  if shuffle:
      dataset = dataset.shuffle(buffer_size=shuffle_size)
  dataset = dataset.map(parser, num_parallel_calls=16)
  dataset = dataset.prefetch(buffer_size = 1 * batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  return dataset




def get_files(files_path,categories):
    all_files = []
    if categories==None:
        categories = range(0,13)
    for cat in categories:
        files_path_cat = files_path+str(cat)+'/'
        files = [f for f in glob.glob(files_path_cat + "*.tfrecords")]
        all_files = all_files+files
    return all_files

def iterator(path,batch_size,epochs,shuffle=True,img_size=137,im_per_obj=24,grid_size=36,num_samples=10000,shuffle_size=1000,categories=None):
    files    = get_files(path,categories=categories)
    if shuffle:
        random.seed()
        random.shuffle(files)
    dataset  = dataset_input_fn(files,batch_size,epochs,shuffle,img_size,im_per_obj,grid_size,num_samples,shuffle_size)
    iterator = dataset.make_initializable_iterator()
    return iterator
    
    


def process_batch_train(next_element,idx_node,config):
    samples_xyz_np       = tf.tile(tf.random_uniform(minval=-1.,maxval=1.,shape=(1,config.global_points,3)),(config.batch_size,1,1))
    vertices             = next_element['vertices']/(config.grid_size_v-1)*2-1
    gaussian_noise       = tf.random_normal(mean=0.0,stddev=config.noise_scale,shape=(config.batch_size,config.num_samples,3))
    vertices             = tf.clip_by_value((vertices+gaussian_noise),clip_value_min=-1.0,clip_value_max=1.0)
    samples_xyz_np       = tf.concat((samples_xyz_np,vertices),axis=1)
    samples_ijk_np       = tf.cast(tf.round(((samples_xyz_np+1)/2*(config.grid_size-1))),dtype=tf.int32)
    batch_idx            = tf.constant(np.tile(np.reshape(np.arange(0,config.batch_size,dtype=np.int32),(config.batch_size,1,1)),(1,config.num_samples+config.global_points,1)))
    samples_ijk_np       = tf.reshape(tf.concat((batch_idx,samples_ijk_np),axis=-1),(config.batch_size*(config.num_samples+config.global_points),4))
    
    b,i,j,k              = tf.split(samples_ijk_np,[1,1,1,1],axis=-1)
    samples_ijk_np_flip  = tf.concat((b,j,i,k),axis=-1)
    voxels_gathered      = tf.gather_nd(next_element['voxels'],samples_ijk_np_flip)
    samples_sdf_np       = tf.reshape(-1.*tf.cast(voxels_gathered,tf.float32) + 0.5,(config.batch_size,-1,1))
    images               = next_element['images']
    images               = tf.cast(tf.gather(images,idx_node,axis=1),dtype=tf.float32)/255.
    if config.augment:
        rgb_idx = tf.concat((tf.random_shuffle(tf.constant([0,1,2])),tf.constant([3])),axis=0)
        images  = tf.gather(images,rgb_idx,axis=-1)
        images = tf.concat((images[0:config.batch_size/2,:,:,:],tf.reverse(images[(config.batch_size/2):,:,:,:],axis=[2])),axis=0)
        filp_xyz = tf.concat((tf.tile(tf.constant([[[1.,1.,1.]]]),(config.batch_size/2,1,1)),tf.tile(tf.constant([[[-1.,1.,1.]]]),(config.batch_size/2,1,1))),axis=0)
        samples_xyz_np = samples_xyz_np*filp_xyz
    if config.rgba==0:
        images           = images[:,:,:,0:3]
    return {'samples_xyz':samples_xyz_np,'samples_sdf':samples_sdf_np,'images':images,'ids':next_element['ids']}


def process_batch_test(next_element,idx_node,config):
    if config.grid_size==36:
        grid_size_lr   = 32*config.eval_grid_scale
        x_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        y_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
        z_lr           = np.linspace(-32./36, 32./36, grid_size_lr)
    else:
        grid_size_lr   = config.grid_size*config.eval_grid_scale
        x_lr           = np.linspace(-1, 1, grid_size_lr)
        y_lr           = np.linspace(-1, 1, grid_size_lr)
        z_lr           = np.linspace(-1, 1, grid_size_lr)    
    xx_lr,yy_lr,zz_lr    = np.meshgrid(x_lr, y_lr, z_lr) 
    images               = next_element['images'][0,:,:,:,:]
    if config.test_size==1:
        images    = tf.expand_dims(tf.gather(images,idx_node,axis=0),axis=0)
    samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(config.test_size,1,1))
    samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
    samples_xyz_np       = tf.cast(tf.constant(samples_xyz_np),dtype=tf.float32)
    samples_ijk_np       = tf.constant(samples_ijk_np)
    batch_idx            = tf.constant(np.tile(np.reshape(np.arange(0,config.test_size,dtype=np.int32),(config.test_size,1,1)),(1,grid_size_lr**3,1)))
    samples_ijk_np       = tf.reshape(tf.concat((batch_idx,samples_ijk_np),axis=-1),(config.test_size*grid_size_lr**3,4))
    b,i,j,k                = tf.split(samples_ijk_np,[1,1,1,1],axis=-1)
    samples_ijk_np_flip  = tf.squeeze(tf.concat((b,j,i,k),axis=-1))
    voxels               = tf.tile(next_element['voxels'],(config.test_size,1,1,1))
    voxels_gathered      = tf.gather_nd(voxels,samples_ijk_np_flip)
    samples_sdf_np       = tf.reshape(-1.*tf.cast(voxels_gathered,tf.float32) + 0.5,(config.test_size,-1,1))
    images               = tf.cast(images,dtype=tf.float32)/255.
    if config.rgba==0:
        images           = images[:,:,:,0:3]
    return {'samples_xyz':samples_xyz_np,'samples_sdf':samples_sdf_np,'images':images,'ids':tf.tile(next_element['ids'],(config.test_size,1))}



    


