from sklearn.feature_extraction import image as sk_image
import tensorflow as tf





a = np.expand_dims(np.random.uniform(0,1,300),axis=0)
a = np.reshape(a,(10,10,3), order='F')
a = np.transpose(a,(1,0,2))



b = np.expand_dims(np.random.uniform(0,1,300),axis=0)
b = np.reshape(b,(10,10,3), order='F')
b = np.transpose(b,(1,0,2))
bb = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(b, np.float32),axis=0),axis=4)

features_bank = sk_image.extract_patches_2d(a, (3, 3))
features_bank = np.expand_dims(np.transpose(features_bank,(1,2,3,0)),axis=3)



# Calculate normalized correlations
layer_filtered = tf.nn.conv3d(bb,features_bank,strides=[1, 1, 1, 1, 1],padding='VALID')
max_filter_response_idx = tf.squeeze(tf.argmax(layer_filtered,axis=4))
max_filter_response_idx_ = tf.reshape(max_filter_response_idx,[-1])


max_filter_response_weight = tf.squeeze(tf.reduce_max(tf.abs(layer_filtered),axis=4))
max_filter_response_weight = tf.reshape(max_filter_response_weight,[-1])
max_filter_response_weight = max_filter_response_weight/tf.reduce_max(max_filter_response_weight)

#style_filters_tf = tf.transpose(tf.squeeze(tf.convert_to_tensor(style_filters, np.float32)),(3,0,1,2))
#style_filters_tf_gathered = tf.gather(style_filters_tf,max_filter_response_idx_)
#style_filters_tf_gathered = tf.reshape(style_filters_tf_gathered,(style_filters_tf_gathered.get_shape().as_list()[0], -1))
layer_patches = tf.extract_image_patches(tf.squeeze(layer_pad,axis=4),
                        [1,3,3,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        padding="VALID")
layer_size = tf.shape(layer_patches)
layer_patches = tf.reshape(layer_patches,(-1, layer_size[3]))
style_norm = tf.cast(layer_size[1]*layer_size[2]*layer_size[3],dtype=tf.float32)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    idx = max_filter_response_idx.eval()
    idx_= max_filter_response_idx_.eval()

