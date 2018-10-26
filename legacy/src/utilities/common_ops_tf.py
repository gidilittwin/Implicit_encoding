import numpy as np
import tensorflow as tf
from LinAlg import normalize_tensor, cross_product


def smooth_L1(x,thresh,batch_size,mask_ids,mode_node):
    x = tf.reshape(x,(batch_size,-1))
    cost_far = thresh*(tf.abs(x) - thresh/2)
    cost_near = 0.5*tf.pow(x, 2)
    mask = tf.greater_equal(tf.abs(x),thresh)
    loss = tf.where(mask,cost_far,cost_near)
#    loss = tf.cond(mode_node, lambda: tf.boolean_mask(loss,mask_ids), lambda: loss)
    cost = tf.reduce_mean(loss)
    return cost
    
def L2(x,thresh,batch_size):
    cost = tf.reduce_mean(tf.pow(x, 2)) 
    return cost


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[1] * shape[2]))
    output_list.append(argmax % (shape[1] * shape[2]) // shape[2])
    output_list.append(argmax % (shape[0] * shape[1]) % shape[0])
    return tf.stack(output_list)

def cross_entropy(logits,labels,batch_size):
    logits_ = tf.reshape(tf.transpose(logits,(0,4,1,2,3)),(batch_size*21,-1))
    labels_ = tf.reshape(tf.transpose(labels,(0,4,1,2,3)),(batch_size*21,-1))
    loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_,labels=tf.argmax(labels_,axis=1)))
    return loss_class



def unravel_index(indices, shape):
#    indices = tf.transpose(indices,(1,0))
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    shape = tf.cast(shape, tf.float32)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    strides = tf.cast(strides, tf.int32)
    strides_shifted = tf.cast(strides_shifted, tf.int32)
    def even():
        rem = indices - (indices // strides) * strides
        return rem // strides_shifted
    def odd():
        div = indices // strides_shifted
        return div - (div // strides) * strides
    rank = tf.rank(shape)
    return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)


def ravel_index(bxyz, shape):
    b = tf.slice(bxyz,[0,0],[-1,1])
    x = tf.slice(bxyz,[0,1],[-1,1])
    y = tf.slice(bxyz,[0,2],[-1,1])
    z = tf.slice(bxyz,[0,3],[-1,1])
    return z + shape[3]*y + (shape[2] * shape[1])*x + (shape[3] * shape[2] * shape[1])*b


    
def voxel_argmax_tf(voxels):
    shape = voxels.get_shape().as_list()
    voxels_reshaped = tf.reshape(voxels,(shape[0],pow(shape[1],3),21))
    max_idx = tf.argmax(voxels_reshaped,axis=1)
    max_idx_c = unravel_argmax(max_idx,shape[1:4])
    cloud = tf.stack((max_idx_c[0],max_idx_c[1],max_idx_c[2]), axis=2)
    return cloud 



def Denormalize_UVD(recon_params, norm_uvd, name='normalized_UVD_to_UVD'):
    '''
    Grabs the cropped normalized uvd parameters obtained from Yaron's preprocessing and computes original image UVD
    :param recon_parameters: tensor with shape [batch_size, 7]  representing reconstruction parameters given by Yaron's scripts
    :param norm_uvd: tensor with shape [batch_size, num_points, 3] representing normalized UVD coordinates
    :param name: Name
    :return:
    '''
    u = tf.gather(norm_uvd, indices=0, axis=-1)  # shape [batch_size, num_points]
    v = tf.gather(norm_uvd, indices=1, axis=-1)  # shape [batch_size, num_points]
    d = tf.gather(norm_uvd, indices=2, axis=-1)  # shape [batch_size, num_points]

    d = (d + 1.0) / 2.0 * tf.gather(recon_params, indices=[6], axis=-1)
    u = (u + 1.0) / 2.0 * tf.gather(recon_params, indices=[5], axis=-1)
    v = (v + 1.0) / 2.0 * tf.gather(recon_params, indices=[5], axis=-1)
    u /= tf.gather(recon_params, indices=[3], axis=-1)
    v /= tf.gather(recon_params, indices=[4], axis=-1)
    u += tf.gather(recon_params, indices=[0], axis=-1)
    v += tf.gather(recon_params, indices=[1], axis=-1)
    d += tf.gather(recon_params, indices=[2], axis=-1)

    uvd = tf.stack([u, v, d], axis=-1)

    return uvd


def center_to_Rotation(camera, center):
    '''

    :param camera: tensor of camera matrix of shape [batch_size, 3, 3] for the input image
    :param center: tensor containing the centers of the crops in pixel coordinates of shape [batch_size,2]
    :return:
    '''
    center = tf.convert_to_tensor(center)
    center = tf.expand_dims(center, axis=1)
    center_rays = tf.squeeze(UV_to_Ray(camera, center), axis=1)
    z = normalize_tensor(center_rays, axis=-1)
    x = normalize_tensor(cross_product(tf.constant([0, 1, 0], dtype=tf.float32), z, axis_a=0, axis_b=-1),
                         axis=-1)
    y = cross_product(z, x, axis_a=-1, axis_b=-1)
    rotation = tf.stack([x, y, z], axis=1)
    return rotation



def UV_to_Ray(camera, uv, name='UV_to_Ray'):
    '''
    Project UV to XYZ where Z is set to 1
    :param camera: tensor of shape [batch_size,3,3]
    :param uv: tensor of shape [batch_size, num_points, 2]
    :param name: Name
    :return: tensor of xyz with shape [batch_size, num_points, 3]
    '''
    with tf.name_scope(name):
        camera_inv = tf.matrix_inverse(camera)
        uvd = tf.concat([uv, tf.ones_like(tf.gather(uv, indices=[0], axis=-1))], axis=-1)
        xyz = tf.matmul(uvd, camera_inv, transpose_b=True)
        xyz *= tf.gather(xyz, indices=[2], axis=-1)
        return xyz

def XYZ_to_UVD(camera, xyz, name='xyz_to_uvd'):
    '''
    Projct XYZ to UVD point
    :param camera: tensor of shape [batch_size,3,3]
    :param xyz: tensor of shape [batch_size, num_points, 3]
    :return: tensor of uvd with shape [batch_size, num_points, 3]
    '''
    uvd = tf.matmul(xyz, camera, transpose_b=True)
    d = tf.gather(uvd, indices=[2], axis=-1)
    uv = tf.gather(uvd, indices=[0, 1], axis=-1) / d
    uvd = tf.concat([uv, d], axis=-1)
    return uvd


#def UVD_to_XYZ(camera, uvd, name='uvd_to_xyz'):
#    '''
#    Project UVD to XYZ point
#    :param camera: tensor of shape [batch_size,3,3]
#    :param uvd: tensor of shape [batch_size, num_points, 3]
#    :param name: Name
#    :return:
#    '''
#
#    camera_inv = tf.matrix_inverse(camera)
#    xyz = tf.matmul(uvd, camera_inv, transpose_b=True)
#    z = tf.gather(xyz, indices=[2], axis=-1)
#    xy = tf.gather(xyz, indices=[0, 1], axis=-1) * z
#    return tf.concat([xy, z], axis=-1)


def UVD_to_XYZ(camera, uvd, name='uvd_to_xyz'):
    '''
    Project UVD to XYZ point
    :param camera: tensor of shape [batch_size,3,3]
    :param xyz: tensor of shape [batch_size, num_points, 3]
    :return:
    '''

    xy = tf.gather(uvd, indices=[0, 1], axis=-1)
    z = tf.gather(uvd, indices=[2], axis=-1)

    fx = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=0, axis=1), indices=0, axis=1),axis=1),axis=1)
    fy = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=1, axis=1), indices=1, axis=1),axis=1),axis=1)
    cx = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=0, axis=1), indices=2, axis=1),axis=1),axis=1)
    cy = tf.expand_dims(tf.expand_dims(tf.gather(tf.gather(camera, indices=1, axis=1), indices=2, axis=1),axis=1),axis=1)
    x = (tf.gather(xy, indices=[0], axis=-1) - cx) * z / fx
    y = (tf.gather(xy, indices=[1], axis=-1) - cy) * z / fy
    xyz = tf.concat((x, y, z), axis=-1)
    return xyz


def homographic_image_crop(image, camera_in, camera_out, rows_out, cols_out, center=None, rotation=None,
                           name='homographic_crop'):
    '''
    Does homographic cropping of image
    :param image: tensor of images of shape [batch_size, row, col, channels]
    :param camera_in: tensor of camera matrix of shape [batch_size, 3, 3] for the input image
    :param camera_out: tensor of camera matrix of shape [batch_size, 3, 3] for the output image
    :param rows_out: number of rows out
    :param cols_out: number of collums out
    :param center: tensor containing the centers of the crops in pixel coordinates of shape [batch_size,2]
    :param rotation: tensor containing the frame of the camera out of shape [batch_size, 3, 3]. If provided value of
            center is ignorred.
    :param name:
    :return: xyz of point cloud after camera rotation
    '''
    with tf.name_scope(name):
        image = tf.convert_to_tensor(image)
        camera_out = tf.convert_to_tensor(camera_out)
        camera_in = tf.convert_to_tensor(camera_in)
        if center is None and rotation is None:
            RuntimeError('You must provide a center or a rotation matrix')
        if rotation is None:
            center = tf.convert_to_tensor(center)
            center = tf.expand_dims(center, axis=1)
            center_rays = tf.squeeze(UV_to_Ray(camera_in, center), axis=1)
            z = normalize_tensor(center_rays, axis=-1)
            x = normalize_tensor(cross_product(tf.constant([0, 1, 0], dtype=tf.float32), z, axis_a=0, axis_b=-1),
                                 axis=-1)
            y = cross_product(z, x, axis_a=-1, axis_b=-1)
            rotation = tf.stack([x, y, z], axis=1)
        vv, uu = tf.meshgrid(tf.range(cols_out, dtype=tf.float32), tf.range(rows_out, dtype=tf.float32))
        uv_flatten = tf.stack([tf.reshape(uu, shape=[-1]), tf.reshape(vv, shape=[-1])], axis=-1)
        xyz_flatten = tf.map_fn(
            lambda cam: tf.squeeze(UV_to_Ray(tf.expand_dims(cam, axis=0), tf.expand_dims(uv_flatten, axis=0))),
            camera_out)
        xyz_flatten = tf.matmul(xyz_flatten, rotation)
        return xyz_flatten


def crop_camera(camera, width, row_out, col_out, rotation=None, center=None, height=None, name='crop_camera'):
    '''
    Given the center and width of a circle to crop, returns the camera parameters of the homographic crop
    :param center: tensor containing the centers of the crops in pixel coordinates of shape [batch_size,2]
    :param rotation: tensor containing the frame of the camera out of shape [batch_size, 3, 3]. If provided value of
            center is ignorred.
    :param width: tensor containing width of shape [batch_size]
    :return:
    '''
    with tf.name_scope(name):
        if height is None:
            height = width
        if center is None and rotation is None:
            RuntimeError('You must provide a center or a rotation matrix')
        if rotation is None:
            center = tf.convert_to_tensor(center)
            center = tf.expand_dims(center, axis=1)
            center_rays = tf.squeeze(UV_to_Ray(camera, center), axis=1)
            z = normalize_tensor(center_rays, axis=-1)
            x = normalize_tensor(cross_product(tf.constant([0, 1, 0], dtype=tf.float32), z, axis_a=0, axis_b=-1),
                                 axis=-1)
            y = cross_product(z, x, axis_a=-1, axis_b=-1)
            rotation = tf.stack([x, y, z], axis=1)
            center = tf.concat([center, tf.ones_like(tf.gather(center, indices=[0], axis=-1))], axis=-1)
        if center is None:
            center = XYZ_to_UVD(camera, tf.gather(rotation, indicies=[2], axis=1))
        r_vec = tf.concat([center + tf.stack(
            [((-1.0) ** i) * height, ((-1.0) ** np.floor(i / 2)) * width, tf.zeros_like(width)], axis=-1) for i in
                           range(4)], axis=1)
        r_vec = UVD_to_XYZ(camera, r_vec)
        tf.matmul(r_vec, rotation, transpose_b=True)
        r_vec = tf.gather(XYZ_to_UVD(camera, tf.matmul(r_vec, rotation, transpose_b=True)), indices=[0, 1], axis=-1)
        radius_out = tf.reduce_max(
            tf.reduce_max(r_vec, axis=1, keep_dims=True) - tf.reduce_min(r_vec, axis=1, keep_dims=True), axis=-1,
            keep_dims=True)
        scale = tf.concat([row_out / radius_out, col_out / radius_out, tf.ones_like(radius_out)], axis=-2)
        cam_out = camera * scale
        center_out = tf.expand_dims(tf.expand_dims(
            tf.stack([tf.cast(row_out, dtype=tf.float32) / 2.0, tf.cast(row_out, dtype=tf.float32) / 2.0, 1.0],
                     axis=0), axis=0), axis=-1)
        center_out = tf.ones_like(tf.gather(cam_out, [2], axis=-1)) * center_out
        cam_out = tf.concat([tf.gather(cam_out, [0, 1], axis=-1), center_out], axis=-1)
        return cam_out, rotation

def depth_image_to_XYZ(camera, depth, name='depht_pointcloud'):
    '''
    Projects all points in a depth image to XYZ
    :param camera: tensor of shape [batch_size, 3, 3]
    :param depth: tensor of shape [batch_size, row, col]
    :return:
    '''

    rows = tf.shape(depth)[1]
    cols = tf.shape(depth)[2]
    uu, vv = image_meshgrid(rows, cols, 1, dtype=tf.float32)  # shape [1,row, col,1]
    uu = tf.squeeze(uu, axis=-1)
    vv = tf.squeeze(vv, axis=-1)

    fx = tf.tile(tf.gather(tf.gather(camera, indices=[0], axis=1), indices=[0], axis=2), multiples=[1, rows, cols])
    fy = tf.tile(tf.gather(tf.gather(camera, indices=[1], axis=1), indices=[1], axis=2), multiples=[1, rows, cols])
    cx = tf.tile(tf.gather(tf.gather(camera, indices=[0], axis=1), indices=[2], axis=2), multiples=[1, rows, cols])
    cy = tf.tile(tf.gather(tf.gather(camera, indices=[1], axis=1), indices=[2], axis=2), multiples=[1, rows, cols])

    xx = (uu - cx) / fx * depth
    yy = (vv - cy) / fy * depth
    return tf.stack([xx, yy, depth], axis=-1)


def image_meshgrid(rows, cols, batch_size=1, channels=1, dtype=tf.float32):
    '''

    Returns a meshgrid of the image with shape [batch_size, rows, cols, channels]
    :param rows: Scalar number of rows
    :param cols: Scalar number of cols
    :param batch_size: Scalar batch size (default 1)
    :param channels: Scalar channels (default 1)
    :return: uu: Tensor image normalized i index meshgrid with shape [batch_size, rows, cols, channels]
    :return: vv: Tensor image normalized j index meshgrid with shape [batch_size, rows, cols, channels]
    '''
    # create meshgrid
    uu = tf.reshape(tf.range(0, tf.cast(rows, dtype=dtype), dtype=dtype), shape=[rows, 1])  # uu has shape [rows, 1]
    uu = tf.tile(uu, multiples=[1, cols])  # uu has shape [rows, cols]
    vv = tf.reshape(tf.range(0, tf.cast(cols, dtype=dtype), dtype=dtype), shape=[1, cols])  # vv has shape [1, cols]
    vv = tf.tile(vv, multiples=[rows, 1])  # vv has shape [rows, cols]

    # tile batch
    uu = tf.reshape(uu, shape=[1, rows, cols])
    uu = tf.tile(uu, multiples=[batch_size, 1, 1])  # uu shape = [batch_size, rows, cols]
    vv = tf.reshape(vv, shape=[1, rows, cols])
    vv = tf.tile(vv, multiples=[batch_size, 1, 1])  # vv shape = [batch_size, rows, cols]

    # tile chanels
    uu = tf.expand_dims(uu, axis=-1)
    uu = tf.tile(uu, multiples=[1, 1, 1, channels])
    vv = tf.expand_dims(vv, axis=-1)
    vv = tf.tile(vv, multiples=[1, 1, 1, channels])

    return uu, vv


def normalized_meshgrid(rows, cols, batch_size=1, channels=1, name='normalized_meshgrid'):
    '''
    Returns a meshgrid on a normalized image with shape [batch_size, rows, cols,
    :param rows: Scalar number of rows
    :param cols: Scalar number of cols
    :param batch_size: Scalar batch size (default 1)
    :param channels: Scalar channels (default 1)
    :return: uu: Tensor image normalized i index meshgrid with shape [batch_size, rows, cols, channels]
    :return: vv: Tensor image normalized j index meshgrid with shape [batch_size, rows, cols, channels]
    '''
    # create meshgrid
    uu = tf.reshape(tf.linspace(0.0, 1.0, rows), shape=[rows, 1])  # uu has shape [rows, 1]
    uu = tf.tile(uu, multiples=[1, cols])  # uu has shape [rows, cols]
    vv = tf.reshape(tf.linspace(0.0, 1.0, cols), shape=[1, cols])  # vv has shape [1, cols]
    vv = tf.tile(vv, multiples=[rows, 1])  # vv has shape [rows, cols]

    # tile batch
    uu = tf.reshape(uu, shape=[1, rows, cols])
    uu = tf.tile(uu, multiples=[batch_size, 1, 1])  # uu shape = [batch_size, rows, cols]
    vv = tf.reshape(vv, shape=[1, rows, cols])
    vv = tf.tile(vv, multiples=[batch_size, 1, 1])  # vv shape = [batch_size, rows, cols]

    # tile chanels
    uu = tf.expand_dims(uu, axis=-1)
    uu = tf.tile(uu, multiples=[1, 1, 1, channels])
    vv = tf.expand_dims(vv, axis=-1)
    vv = tf.tile(vv, multiples=[1, 1, 1, channels])

    return uu, vv






def group_points(point_cloud_idx,point_cloud):
    indices_ = np.ravel_multi_index((point_cloud_idx[:,0],point_cloud_idx[:,1],point_cloud_idx[:,2],point_cloud_idx[:,3]),
                                             (self.batch_size,self.grid_size,self.grid_size,self.grid_size))
    keys = np.unique(indices_)
    voxel_dict = {key:np.zeros((1,25,3),np.float32) for key in keys}
    counter_dict = {key:0 for key in keys}
    for ii in np.arange(0,indices_.shape[0]):
        key_ = indices_[ii]
        if counter_dict[key_]<25:
            voxel_dict[key_][0,counter_dict[key_],:] = point_cloud[ii,1:]
            counter_dict[key_]+=1
    values = np.concatenate(voxel_dict.values(),0)
    return values


