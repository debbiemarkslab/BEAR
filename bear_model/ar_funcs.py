import tensorflow.compat.v2 as tf
import numpy as np


def _normalize_layer(layer, reduce_dims=[-1]):
    """Normalize a neural network layer.

    Parameters
    ----------
    layer : tensor
    reduce_dims : list of ints, default = [-1]
        Axes to normalize.

    Returns
    -------
    noramlized_layer : tensor
    """
    mean, var = tf.nn.moments(layer, reduce_dims, keepdims=True)
    layer = (layer - mean) / tf.sqrt(var + 1E-5)
    return layer


def make_ar_func_linear(lag, alphabet_size, dtype=tf.float64):
    """Make a linear autoregressive function.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    dtype : dtype, default = tf.float64

    Returns
    -------
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer.
    params : list
        List of parameters as tensorflow variables.
    """
    mat = tf.random.normal([lag, alphabet_size+1, alphabet_size+1], dtype=dtype)
    mat = tf.Variable(0.05*tf.nn.l2_normalize(mat, axis=[1]))

    def ar_func(kmers):
        return tf.nn.softmax(tf.einsum('...jk, jkl-> ...l', kmers, mat))
    return ar_func, [mat]


def make_ar_func_cnn(lag, alphabet_size,
                     filter_width=8, num_filters=30, dtype=tf.float64):
    """Make a convolutional neural network autoregressive function.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    filter_width : int or float, default = 8
    num_filters : int or float, default = 30
    dtype : dtype, default = tf.float64

    Returns
    -------
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer.
    params : list
        List of parameters as tensorflow variables.
    """
    filter_width = int(filter_width)
    num_filters = int(num_filters)
    small_start = 0.05
    kmer_layer1_width = 16
    final_shape = [alphabet_size+1]
    data_dim = [lag, alphabet_size+1]
    filters = tf.random.normal([filter_width, data_dim[1], num_filters], dtype=dtype)
    filters = tf.Variable(tf.nn.l2_normalize(filters, axis=[0, 1]), name='filters')
    kmer_intercept0 = tf.Variable(
        tf.ones([data_dim[0]-filter_width+1, num_filters], dtype=dtype), name='kmer_intercept0')
    kmer_scale0 = tf.Variable(tf.ones(tf.shape(kmer_intercept0), dtype=dtype), name='kmer_scale0')
    kmer_weights1 = tf.random.normal([data_dim[0]-filter_width+1, num_filters, kmer_layer1_width], dtype=dtype)
    kmer_weights1 = tf.Variable(tf.nn.l2_normalize(kmer_weights1, axis=[0]), name='kmer_layer1')
    kmer_intercept1 = tf.Variable(tf.ones([kmer_layer1_width], dtype=dtype), name='kmer_intercept1')
    kmer_scale1 = tf.Variable(tf.ones(tf.shape(kmer_intercept1), dtype=dtype), name='kmer_scale1')
    kmer_weights2 = tf.random.normal([kmer_layer1_width]+final_shape, dtype=dtype)
    kmer_weights2 = tf.Variable(small_start*tf.nn.l2_normalize(kmer_weights2, axis=[0]), name='kmer_layer2')
    kmer_intercept2 = tf.Variable(tf.zeros(final_shape, dtype=dtype), dtype=dtype, name='kmer_intercept2')

    def ar_func(data):
        kmer_nn_0 = kmer_scale0*_normalize_layer(tf.nn.conv1d(data, filters, 1, 'VALID')) + kmer_intercept0
        kmer_nn_1 = (kmer_scale1*_normalize_layer(
            tf.tensordot(tf.nn.elu(kmer_nn_0), kmer_weights1, axes=[[-2, -1], [0, 1]]))
                     + kmer_intercept1)
        kmer_nn_2 = tf.tensordot(tf.nn.elu(kmer_nn_1), kmer_weights2, axes=[[-1], [0]]) + kmer_intercept2
        return tf.nn.softmax(kmer_nn_2)
    return ar_func, ([filters] + [kmer_intercept0] + [kmer_weights1] + [kmer_intercept1]
                     + [kmer_weights2] + [kmer_intercept2] + [kmer_scale0] + [kmer_scale1])


def make_ar_func_stop(lag, alphabet_size, dtype=tf.float64):
    """Make an autoregressive function that always predicts a stop.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    dtype : dtype, default = tf.float64

    Returns
    -------
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. In this case, it always predicts
        a stop. For use with reference AR model.
    params : list
        Empty list of parameters as tensorflow variables.
    """
    stop = np.zeros(alphabet_size+1)
    stop[-1] = 1
    stop = tf.convert_to_tensor(stop, dtype=dtype)

    def ar_func(y):
        return stop
    return ar_func, []
