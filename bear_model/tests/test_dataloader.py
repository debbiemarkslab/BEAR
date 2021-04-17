from bear_model import dataloader
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from pkg_resources import resource_filename


def test_tf_io_version():
    # Check that json multidimensional testor decoder is working
    tensor = tf.convert_to_tensor([[[3, 4, 2, 8], [0, 9, 12, 3], [1, 6, 7, 6]],
                                   [[2, 4, 4, 8], [5, 9, 123, 3], [1, 5, 75, 6]]],
                                  dtype=tf.float64)
    string = '{"x":[[[3,4,2,8],[0,9,12,3],[1,6,7,6]],[[2,4,4,8],[5,9,123,3],[1,5,75,6]]]}'
    dec = tfio.experimental.serialization.decode_json(
        string, {'x': tf.TensorSpec([None, 3, 4], tf.float64)})['x']
    assert tf.math.reduce_all(dec == tensor)


def test_dataloader():
    # Check that the dataloader is otherwise behaving expectedly
    f_name = resource_filename('bear_model', 'models/data/shuffled_virus_kmers_lag_5.tsv')
    data = dataloader.dataloader(f_name, 'dna', 3, 3)
    kmers, counts = next(iter(data))
    kmers_real = np.array([b'TAATC', b'CGGTC', b'ACGCT'])
    assert np.all(kmers.numpy() == kmers_real)
    counts_real = [[[14837, 15127, 22260, 16279, 446], [5029, 5095, 7408, 5487, 134], [16, 16, 23, 17, 0]],
                   [[61890, 729, 39733, 35956, 1017], [20524, 239, 13199, 12046, 309], [69, 0, 45, 39, 0]],
                   [[13965, 23135, 73870, 37045, 1035], [4705, 7591, 24532, 12305, 385], [14, 25, 81, 39, 0]]]
    assert np.all(counts.numpy() == np.array(counts_real))
    assert counts.dtype == tf.float64
    assert len(list(data)) == 1365/3
