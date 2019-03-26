
import tensorflow as tf
import numpy as np
import os
import glob






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
        


def dataset_input_fn(filenames,batch_size,epochs,shuffle=True):
#  dataset = tf.data.TFRecordDataset(filenames)
  dataset = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "voxels":   tf.FixedLenFeature((), tf.string, default_value=""),
        "images":   tf.FixedLenFeature((), tf.string, default_value=""),
        "classes":  tf.FixedLenFeature((), tf.string, default_value=""),
        "ids":      tf.FixedLenFeature((), tf.string, default_value=""),
        "vertices": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed   = tf.parse_single_example(record, keys_to_features)
    
    parsed['images']   = tf.reshape(tf.decode_raw(parsed['images'],out_type=tf.uint8),(24,137,137,4))
    parsed['voxels']   = tf.reshape(tf.decode_raw(parsed['voxels'],out_type=tf.uint8),(36,36,36))
    parsed['classes']  = tf.reshape(tf.decode_raw(parsed['classes'],out_type=tf.int64),(1,))
    parsed['ids']      = tf.reshape(tf.decode_raw(parsed['ids'],out_type=tf.int64),(1,))
    parsed['vertices'] = tf.cast(tf.reshape(tf.decode_raw(parsed['vertices'],out_type=tf.int32),(10000,3)),tf.float32)/10.
    return parsed

  dataset = dataset.map(parser)
  if shuffle:
      dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(batch_size)
#  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
#  dataset = dataset.repeat(epochs)
  return dataset




def get_files(files_path):
    all_files = []
    for cat in range(0,13):
        files_path_cat = files_path+str(cat)+'/'
        files = [f for f in glob.glob(files_path_cat + "*.tfrecords")]
        all_files = all_files+files
    return all_files

def iterator(path,batch_size,epochs,shuffle=True):
    files    = get_files(path)
    dataset  = dataset_input_fn(files,batch_size,epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator
    
    


def process_batch_train(next_element,idx_node,config):
    samples_xyz_np       = tf.random_uniform(minval=-1.,maxval=1.,shape=(config.batch_size,config.global_points,3))
    vertices             = next_element['vertices']/(config.grid_size-1)*2-1
    gaussian_noise       = tf.random_normal(mean=0.0,stddev=config.noise_scale,shape=(config.batch_size,10000,3))
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
    if config.rgba==0:
        images           = images[:,:,:,0:3]
    return {'samples_xyz':samples_xyz_np,'samples_sdf':samples_sdf_np,'images':images}


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
    samples_xyz_np       = np.tile(np.reshape(np.stack((xx_lr,yy_lr,zz_lr),axis=-1),(1,-1,3)),(config.test_size,1,1))
    samples_ijk_np       = np.round(((samples_xyz_np+1)/2*(config.grid_size-1))).astype(np.int32)
    samples_xyz_np       = tf.cast(tf.constant(samples_xyz_np),dtype=tf.float32)
    samples_ijk_np       = tf.constant(samples_ijk_np)
    batch_idx            = tf.constant(np.tile(np.reshape(np.arange(0,config.test_size,dtype=np.int32),(config.test_size,1,1)),(1,grid_size_lr**3,1)))
    samples_ijk_np       = tf.reshape(tf.concat((batch_idx,samples_ijk_np),axis=-1),(config.test_size*grid_size_lr**3,4))
    b,i,j,k                = tf.split(samples_ijk_np,[1,1,1,1],axis=-1)
    samples_ijk_np_flip  = tf.squeeze(tf.concat((b,j,i,k),axis=-1))
    voxels_gathered      = tf.gather_nd(next_element['voxels'],samples_ijk_np_flip)
    samples_sdf_np       = tf.reshape(-1.*tf.cast(voxels_gathered,tf.float32) + 0.5,(config.test_size,-1,1))
    images               = next_element['images']
    images               = tf.cast(tf.gather(images,idx_node,axis=1),dtype=tf.float32)/255.
    if config.rgba==0:
        images           = images[:,:,:,0:3]
    return {'samples_xyz':samples_xyz_np,'samples_sdf':samples_sdf_np,'images':images}



    


