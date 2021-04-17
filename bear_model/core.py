import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability import distributions as tfpd
from tensorflow_probability import math as tfpmath


epsilon = tf.keras.backend.epsilon()


class tfpDirichletMultinomialPerm(tfpd.distribution.Distribution):
    """Tensorflow Dirichlet Multinomial distribution counting permutations.

    Attributes:
    total_count : tensor of shape [A1, ..., An]
    concentration : tensor of shape [Am, ..., An, alphabet_size + 1] for m>=1
        Must be of same dtype as total_count, self.dtype.

    Methods:
    _sample_n(n)
        takes an int n and returns a tensor of zeros of the size of
        [n, A1, ..., An, alphabet_size + 1] in self.dtype. A dummy sampler
        that allows this function to be used to create an edward2 random variable.
    ml_output()
        returns a tensor of self.dtype of shape [A1, ..., An] of the
        positions of the highest concentration values along the last axis.
        Resolves ties randomly.
    counts_log_prob(value)
        takes a tensor of self.dtype of shape [A1, ..., An, alphabet_size + 1]
        and returns its probability under a Dirichlet Multinomial assuming the
        transitions were observed in a particular order, i.e. dividing the
        probability by the multinomial coefficient total_counts! divided by the
        product of the facorials of the transition counts. This is the marginal
        likelihood of counts under a BEAR model.
    """

    def __init__(self,
                 total_count,
                 concentration,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='DirichletMultinomialPerm'):

        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.total_count = total_count
            self.concentration = concentration
            self.alphabet_size = tf.cast(tf.shape(self.concentration)[-1] - 1, tf.int32)

            super(tfpDirichletMultinomialPerm, self).__init__(
              dtype=self.concentration.dtype,
              reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

        self.counts_dist = tfpd.DirichletMultinomial(
            self.total_count, self.concentration, self.validate_args,
            self.allow_nan_stats, self.name)

    def _sample_n(self, n, seed=None, dummy=True):
        return tf.zeros(tf.concat([tf.convert_to_tensor([n]), tf.shape(self.total_count),
                                   tf.convert_to_tensor([self.alphabet_size+1])], axis=0),
                        dtype=self.dtype)

    def ml_output(self):
        noise = 100*epsilon*tf.random.normal(tf.shape(self.concentration), dtype=self.dtype)
        return tf.cast(tf.argmax(self.concentration + noise, axis=-1), self.dtype)

    def counts_log_prob(self, value):
        return self.counts_dist.log_prob(value) - tfpmath.log_combinations(self.total_count, value)


class tfpMultinomialPerm(tfpd.distribution.Distribution):
    """Tensorflow Multinomial distribution counting permutations.

    Attributes:
    total_count : tensor of shape [A1, ..., An]
    probs : tensor of shape [A1, ..., An, alphabet_size + 1]
        Must be of same dtype as total_count, self.dtype.

    Methods:
    _sample_n(n)
        takes an int n and returns a tensor of zeros of the size of
        [n, A1, ..., An, alphabet_size + 1] in self.dtype. A dummy sampler
        that allows this function to be used to create an edward2 random variable.
    ml_output()
        returns a tensor of self.dtype of shape [A1, ..., An] of the
        positions of the highest concentration values along the last axis.
        Resolves ties randomly.
    counts_log_prob(value)
        takes a tensor of self.dtype of shape [A1, ..., An, alphabet_size + 1]
        and returns its probability under a Multinomial assuming the
        transitions were observed in a particular order, i.e. dividing the
        probability by the multinomial coefficient total_counts! divided by the
        product of the facorials of the transition counts. This is the marginal
        likelihood of counts under an AR model.
    """
    def __init__(self,
                 total_count,
                 probs,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='DirichletMultinomialPerm'):

        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.total_count = total_count
            self.probs = probs
            self.alphabet_size = tf.cast(tf.shape(self.probs)[-1] - 1, tf.int32)

            super(tfpMultinomialPerm, self).__init__(
              dtype=self.probs.dtype,
              reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

        self.counts_dist = tfpd.Multinomial(
            self.total_count, probs=self.probs, validate_args=self.validate_args,
            allow_nan_stats=self.allow_nan_stats, name=self.name)

    def _sample_n(self, n, seed=None):
        return tf.zeros(tf.concat([tf.convert_to_tensor([n]), tf.shape(self.total_count),
                                   tf.convert_to_tensor([self.alphabet_size+1])], axis=0),
                        dtype=self.dtype)

    def ml_output(self):
        return tf.cast(tf.argmax(self.probs+epsilon*tf.random.normal(tf.shape(self.probs), dtype=self.dtype),
                                 axis=-1), self.dtype)

    def counts_log_prob(self, value):
        return self.counts_dist.log_prob(value) - tfpmath.log_combinations(self.total_count, value)


alphabets_tf = {
    'prot': tf.convert_to_tensor(
        [b'A', b'R', b'N', b'D', b'C', b'E', b'Q', b'G', b'H', b'I', b'L',
         b'K', b'M', b'F', b'P', b'S', b'T', b'W', b'Y', b'V', b'[']),
    'dna': tf.convert_to_tensor(['A', b'C', b'G', b'T', b'[']),
    'rna': tf.convert_to_tensor(['A', b'C', b'G', b'U', b'['])}


@tf.function
def tf_one_hot(seq, alphabet, dtype=tf.float64):
    """One hot encode tensorflow strings into tensors

    Args:
    seq : tensorflow strings of shape [A1, ..., An]
        Must all be the same length and only include letters in alphabet or '['.
    alphabet : one of 'dna', 'rna', 'prot'
        Selects the encoding to use. '[' is alpways in the last column.
    dtype : dtype, default = tf.float64
        output datatype.

    Returns:
        tensor of datatype dtype of shape [A1, ..., An, alphabet_size+1]
        of one hot encoding.
    """
    alph = alphabets_tf[alphabet]
    ohe = tf.math.equal(tf.strings.bytes_split(seq).to_tensor()[..., None], alph)
    return tf.cast(ohe, dtype=dtype)
