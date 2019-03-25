
import tensorflow as tf
import numpy as np
import os







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
        


def dataset_input_fn(filenames,batch_size,epochs):
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
    parsed['classes']  = tf.decode_raw(parsed['classes'],out_type=tf.int64)
    parsed['ids']      = tf.decode_raw(parsed['ids'],out_type=tf.int64)
    parsed['vertices'] = tf.cast(tf.reshape(tf.decode_raw(parsed['vertices'],out_type=tf.int32),(10000,3)),tf.float32)/10.
    return parsed

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=1)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(epochs)
  return dataset


