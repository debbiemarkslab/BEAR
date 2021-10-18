from . import core
import tensorflow as tf
import tensorflow_io as tfio


def dataloader(file, alphabet, batch_size, num_ds,
               cache=True, header=False, n_par=1, dtype=tf.float64):
    """Load counts data into tensorflow data object.

    Parameters
    ----------
    file : str
        Location of counts data, which should be a tsv file with rows in the format:
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
        Number of parallel calls to turn counts matrix strings to tensors.
    dtype : dtype, default = tf.float64

    Returns
    -------
    data : tensorflow data object
        One element of the data is a list of batch_size kmers and a counts tensor of shape
        [batch_size, num_ds, alphabet_size+1].
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1
    data = tf.data.experimental.CsvDataset(file, [tf.string, tf.string], header=header, field_delim='\t')
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

def sparse_dataloader(file, alphabet, batch_size, num_ds, 
                      cache=False, header=True, n_par=1, dtype=tf.float64):
    """Loads counts that are in sparse format into tensorflow data object.
    
    Parameters
    ----------
    file : str
        Location of counts data, which should be a tsv file with rows in the format:
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
        Number of parallel calls to turn counts matrix strings to tensors.
    dtype : dtype, default = tf.float64
    
    Returns
    -------
    data : tensorflow data object
        One element of the data is a list of batch_size kmers and a counts tensor of shape
        [batch_size, num_ds, alphabet_size+1].
    """
    alphabet_size = len(core.alphabets_en[alphabet]) - 1
    data = tf.data.experimental.CsvDataset(file, [tf.string, tf.string, tf.string], header=header, field_delim=';')
    data = data.batch(batch_size)
    def map_(kmer, pre_string_pos, pre_string_count):
        num_tran = tf.strings.length(tf.strings.regex_replace(
            pre_string_pos, "[^]]", ""))-1
        kmer_num = tf.repeat(tf.range(len(num_tran), dtype=tf.int64), num_tran)
        string_pos = tf.strings.regex_replace(pre_string_pos, '\[\[', '[')
        string_pos = tf.strings.regex_replace(string_pos, ']]', ']')
        string_pos = tf.strings.reduce_join(
            ['{"x":[',tf.strings.reduce_join(string_pos, separator=','), ']}'])
        tensor_pos = tfio.experimental.serialization.decode_json(
            string_pos, {'x': tf.TensorSpec(tf.TensorShape([None, 2]), tf.int64)})['x']
        tensor_pos = tf.concat([kmer_num[:, None], tensor_pos], axis=-1)
        
        string_count = tf.strings.regex_replace(pre_string_count, '\[', '')
        string_count = tf.strings.regex_replace(string_count, ']', '')
        string_count = tf.strings.reduce_join(
            ['{"x":[',tf.strings.reduce_join(string_count, separator=','), ']}'])
        tensor_count = tfio.experimental.serialization.decode_json(
            string_count, {'x': tf.TensorSpec([None], dtype)})['x']
        counts = tf.sparse.SparseTensor(tensor_pos, tensor_count,
                                        [len(kmer), num_ds, alphabet_size + 1])
        counts = tf.sparse.reorder(counts)
        return kmer, tf.sparse.to_dense(counts)
    if cache:
        return data.map(map_, num_parallel_calls=n_par).cache()
    else:
        return data.map(map_, num_parallel_calls=n_par)
    
def _marginal_step(batch, alpha, dtype=tf.float64):
    return (tf.math.reduce_sum(tf.math.lbeta(batch[..., None, :]+alpha[:, None]), axis=0)
            -tf.math.reduce_sum(tf.math.lbeta(0*batch[..., None, :]+alpha[:, None]), axis=0))
    
@tf.function
def _distributed_marginal_step(batch, alpha, strategy):
    liks = strategy.run(_marginal_step, args=(batch, alpha))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, liks, axis=None)

def bmm_likelihood(data, alpha, dtype=tf.float64):
    """Gets BMM likelihoods for data. Parallelizes GPU usage.
    Example usage:
    data = dataloader.sparse_dataloader(file, alphabet, batch_size, 3)
    data = data.map(lambda kmers, counts: counts)
    log_likelihoods = dataloader.dist_marg(data, alpha)
    
    Parameters
    ----------
    data : tensorflow data object
        Must return batches of just counts of size [batch_size, num_ds, alphabet_size+1].
    alpha : 1D numpy or tensorflow array
        Prior values of BMM.
    dtype : dtype, default = tf.float64
    
    Returns
    -------
    log_likelihood : tensor
        BMM likelihoods of size [num_ds, len(alpha)].
    """
    strategy = tf.distribute.MirroredStrategy()
    data_iter = iter(strategy.experimental_distribute_dataset(data))
    
    batch = next(data_iter)
    log_likelihood = _distributed_marginal_step(batch, alpha, strategy)
    for batch in data_iter:
        log_likelihood += _distributed_marginal_step(batch, alpha, strategy)
    return log_likelihood
