from . import core
import tensorflow as tf
import tensorflow_io as tfio


def dataloader(filename, alphabet, batch_size, num_ds,
               cache=True, header=False, n_par=1, dtype=tf.float64):
    """Load counts data into tensorflow data object:

    Parameters
    ----------
    filename : str
        Location of counts data, which should be tsv file of the following format:
        kmer_sequence counts_matrix, delimited by a tab.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    batch_size : int
        By minibatching early, the counts matrices for multiple kmers may be decoded at once.
    num_ds : int
        Number of columns in the count data. Ex: 3 for train, test and reference.
    cache : bool, default = True
        Whether or not to cache the loaded data. Increases loading speed at the cost of memory.
    header : bool, default = False
        Whether or not there is a header in the counts data.
    n_par : int, default = 1
        number of parallel calls to turn counts matrix strings to tensors.
    dtype : dtype, default = tf.float64

    Returns
    -------
    data : tensorflow data object
        One element of the data is a list of batch_size kmers and a counts tensor of shape
        [batch_size, num_ds, alphabet_size+1].
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1
    data = tf.data.experimental.CsvDataset(filename, [tf.string, tf.string], header=header, field_delim='\t')
    data = data.batch(batch_size)

    def map_(kmer_sequences, counts_matrices):
        # All the counts matrix strings are combined into a string of a higher dimensional tensor.
        string_count = tf.strings.reduce_join(
            ['{"x":[', tf.strings.reduce_join(counts_matrices, separator=','), ']}'])
        # This string is decoded.
        tensor_count = tfio.experimental.serialization.decode_json(
            string_count, {'x': tf.TensorSpec([None, num_ds, alphabet_size+1], dtype)})['x']
        return kmer_sequences, tensor_count
    if cache:
        return data.map(map_, num_parallel_calls=n_par).cache()
    else:
        return data.map(map_, num_parallel_calls=n_par)
