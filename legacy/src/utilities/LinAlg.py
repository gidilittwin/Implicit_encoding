import numpy as np
import tensorflow as tf


############################# Tensorflow Functions ######################################
def sym_row_col_to_flat_idx(row, col, dim):
    if row <= col:
        l = row
        k = col - row
    else:
        l = col
        k = row - col
    return ((2 * dim - l + 1) * l) / 2


def Levi_Cevita_symbol(perm):
    inv_count = 0
    for idx, i in enumerate(perm[:-1]):
        for j in perm[(idx + 1):]:
            if i > j:
                inv_count += 1
    if inv_count % 2 == 1:
        return -1
    else:
        return 0


def cross_product(a, b, axis_a=0, axis_b=0):
    '''
    Cross product that supports brodacasting
    :param a:
    :param b:
    :param axis_a:
    :param axis_b:
    :return:
    '''
    if axis_a != 0 and axis_b != 0 and axis_a != axis_b:
        TypeError('axis_a and axis_b must equal')
    if axis_a == 0:
        concat_axis = axis_b
    else:
        concat_axis = axis_a
    return tf.concat(
        [tf.gather(a, indices=[(i + 1) % 3], axis=axis_a) * tf.gather(b, indices=[(i + 2) % 3], axis=axis_b)
         - tf.gather(a, indices=[(i - 1) % 3], axis=axis_a) * tf.gather(b, indices=[(i - 2) % 3], axis=axis_b)
         for i in range(3)], concat_axis)


def normalize_tensor(vectors, axis=None, ord='euclidean'):
    '''

    :param vectors tensor of shape [a0, a1, ... axis, ...,an] that will have an axis of vectors normalized
    :param axis: axis on which the tensor is defined. rest of axis treated as batch.
    :return:
    '''
    norm_vec = tf.norm(vectors, ord=ord, axis=axis, keep_dims=True)
    return vectors / norm_vec


def sym_matrix_to_matrix(sym_mat, axis, dim, name='symetric_to_matrix'):
    '''

    :param sym_mat: tensor with shape [:axis, dim * (dim+ 1)/2 , (axis+1):]
    :param dim: static constant that describes the original dimension of the matrix
    :param name:
    :return: tensor with shape [:axis, dim, dim , (axis+1):]
    '''
    with tf.name_scope(name):
        return tf.stack([tf.concat(
            [tf.gather(sym_mat, indices=[sym_row_col_to_flat_idx(r_idx, c_idx, dim)], axis=axis) for c_idx in
             range(dim)], axis=-1)  # shape is [:axis, dim , (axis+1):]
            for r_idx in range(dim)], axis=axis)  # shape is [:axis, dim , dim, (axis+1):]


def outer_prod_sq(points, axis, dim, name='outer_prod'):
    '''

    :param points: tensor with shape [:(axis), axis, (axis+1): ]
    :param axis: tensor or scalar representing the axis that contains the vectors that will be outer squared
    :param name:
    :return: tensor of shape [:(axis), dim * (dim+1)/2, (axis+1):] where dim is points.shape[dim]
    '''
    with tf.name_scope(name):
        max_dim = (dim * (dim + 1) / 2)
        prod = [0] * max_dim
        for r_idx in range(dim):
            for c_idx in range(r_idx, dim):
                flat_idx = sym_row_col_to_flat_idx(r_idx, c_idx, dim)
                prod[flat_idx] = tf.gather(points, indices=[r_idx], axis=axis) * tf.gather(points, indices=[c_idx],
                                                                                           axis=axis)
        return tf.concat(prod, axis=axis)


def reduce_outer_sq_sum(points, index, axis, keep_dims=False):
    points_outer_sq = outer_prod_sq(points, axis=index)
    return tf.reduce_sum(points_outer_sq, axis=axis, keep_dims=keep_dims)


def reduce_outer_sq_mean(points, index, axis, keep_dims=False):
    points_outer_sq = outer_prod_sq(points, axis=index)
    return tf.reduce_mean(points_outer_sq, axis=axis, keep_dims=keep_dims)


def compute_covariance(points, dim, weights=None, name='covariance'):
    '''

    :param points:
    :param dim:
    :param name:
    :return:
    '''
    with tf.name_scope(name):
        if weights is None:
            mean = tf.reduce_mean(points, axis=1, keep_dims=False)
            sec_moment = tf.reduce_sum(outer_prod_sq(points, dim), axis=1, keep_dims=False)


############################# Numpy Functions ######################################
def skewMat2Vec(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


def vec2SkewMat(dir):
    return np.array([[0, -dir[2], dir[1]], [dir[2], 0, -dir[0]], [-dir[1], dir[0], 0]])


def so3ToSO3(vec):
    angle = np.mod(np.linalg.norm(vec), 2 * np.pi)
    if angle == 0:
        return np.eye(3)
    dir = vec / angle
    K = vec2SkewMat(dir)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def SO3Toso3(rot):
    angle = np.arccos((np.trace(rot) - 1) / 2)
    if angle == 0:
        return np.zeros((3))
    dir = skewMat2Vec(rot - np.transpose(rot))
    dir = dir / np.linalg.norm(dir)
    return dir * angle


def R3Rotate(rot, point):
    return np.dot(point, rot)


def vector3ArePerpendicular(a, b, eps=1e-10):
    axb = np.cross(a, b)
    return np.dot(axb, axb) < eps


def ComputeCovariance(listPoints, weights=None):
    if not listPoints:
        return [], []
    shape = listPoints[0].shape
    N = len(listPoints)
    firstMoment = np.zeros(shape)
    secMoment = np.outer(firstMoment, firstMoment);
    if weights is None:
        for x in listPoints:
            firstMoment += x
            secMoment += np.outer(x, x)
        mean = firstMoment / N
        cov = (secMoment - np.outer(firstMoment, firstMoment) / N) / (N - 1)
    else:
        sumWeights = 0
        for idx, x in enumerate(listPoints):
            firstMoment += x * weights[idx]
            secMoment += np.outer(x, x) * weights[idx]
            sumWeights += weights[idx]
        mean = firstMoment / sumWeights
        cov = (secMoment - np.outer(mean, mean) * sumWeights) / (sumWeights - (sumWeights / N))
    return cov, mean


def MahalanobisDistance(cov, mean, point):
    cov_inv = np.linalg.pinv(cov)
    return np.dot((point - mean), np.dot(cov_inv, (point - mean)))


def ComputeOutlierRobustCovariance(listPoints, numIterations=5):
    if not listPoints:
        return np.zeros((3, 3)), np.zeros(3), []
    cov, mean = ComputeCovariance(listPoints)
    weights = [MahalanobisDistance(cov, mean, point) for point in listPoints]
    iter = 1
    constant = 2 * np.sqrt(2)
    while (iter < numIterations):
        cov, mean = ComputeCovariance(listPoints, weights=weights)
        cov = cov * constant
        weights = [MahalanobisDistance(cov, mean, point) for point in listPoints]
        iter += 1
    return cov, mean, weights


def getCovarianceRot(cov):
    U, S, V = np.linalg.svd(cov)
    return U


def projectToPlane(points, normal):
    normalNorm = normal / np.linalg.norm(normal)
    proj = lambda x: x - np.dot(x, normalNorm) * normalNorm
    return [proj(point) for point in points]


def FindRayPlaneIntersection(ray, plane_P, plane_N):
    PdotN = np.dot(plane_P, plane_N)
    RaydotN = np.dot(ray, plane_N)
    if abs(RaydotN) < 1e-14:
        return plane_P
    return ray * PdotN / RaydotN
