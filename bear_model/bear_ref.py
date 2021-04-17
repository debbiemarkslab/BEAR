import tensorflow.compat.v2 as tf
import numpy as np
from . import core


epsilon = tf.keras.backend.epsilon()


def _counts_to_probs(ref_counts, tau, alphabet_size, dtype=tf.float64):
    """Get transition probabilities from reference.

    Parameters
    ----------
    ref_counts : dtype
        A tensor of shape [A1, ..., An, alphabet_size+1] of transition counts from the reference.
        All stops in the reference should be set to 0 so that they may instead be modelled
        by the net function.
    tau : dtype
        A positive constant describing the error rate 1-exp(-tau). Error is added by a Jukes-Cantor model.
    alphabet_size : int
    dtype : dtype, default = tf.float64

    Returns
    -------
    transition_probabilities : dtype
        A tensor of shape [A1, ..., An, alphabet_size+1] normalized in the last dimension and with
        zero probability of stopping.
    """
    # ref_counts have epsilon added to them except at stop, so normalization sums to 1.
    norm = tf.linalg.normalize(ref_counts, ord=1, axis=-1)
    # norm is only 0 at stop (by addition of epsilon when loading ref_counts).
    shape = tf.convert_to_tensor(np.r_[np.ones(alphabet_size), 0], dtype=dtype)
    return ((1/alphabet_size)*shape + tf.exp(-tau)*(norm[0]-((1/alphabet_size)*shape)))


def _make_ref_ar_func(lag, alphabet_size, make_net_func, af_kwargs, dtype=tf.float64):
    """Make an autoregressive function that uses the reference.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    make_net_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns an ar_func.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    dtype : dtype

    Returns
    -------
    ar_func : function, default = None
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
        Set to 0 if None.
    params : list
        List of parameters as tensorflow variables.
    """
    net_weight_signed = tf.Variable(-np.log(100), dtype=dtype, name='net_weight_signed')
    tau_signed = tf.Variable(np.log(1/30), dtype=dtype, name='tau_signed')
    net_func, ar_func_params = make_net_func(lag, alphabet_size, **af_kwargs, dtype=dtype)

    def ar_func(kmer_seqs, ref_counts):
        nw = tf.math.exp(net_weight_signed)
        tau = tf.math.exp(tau_signed)
        return ((nw*net_func(kmer_seqs)
                 + _counts_to_probs(ref_counts, tau, alphabet_size, dtype=dtype))
                / (nw + 1))
    return ar_func, ([tau_signed, net_weight_signed] + ar_func_params)


def _bear_kmer_counts(kmer_seqs, kmer_total_counts, ref_counts,
                      condition_trans_counts=None, h=None, ar_func=None):
    """Get random variable of kmer transition counts conditioned on observed counts.

    Parameters
    ----------
    kmer_seqs : dtype
        A tensor of shape [A1, ..., An, lag, alphabet_size+1] of one-hot encoded kmers.
    kmer_total_counts : int
        A tensor of shape [A1, ..., An] of counts of each kmer.
    condition_trans_counts : int, default = None
        A tensor of shape [A1, ..., An, alphabet_size+1] of transition counts of each
        kmer to condition on. set to 0 if None.
    h : dtype, default = None
        A positive constant of the h parameter from the BEAR model. Set to 1 if None.
    ar_func : function, default = None
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
        Set to 0 if None.

    Returns
    -------
    x : tensorflow probability distribution
        Distribution of kmer transition counts for a BEAR model.
    """
    dtype = kmer_seqs.dtype
    if condition_trans_counts is None:
        condition_trans_counts = tf.constant(0., dtype)
    if h is None or ar_func is None:
        h = tf.constant(1., dtype)

        def ar_func(x, y):
            return tf.constant(0., dtype)
    concentrations = ar_func(kmer_seqs, ref_counts) / h + condition_trans_counts + epsilon
    x = core.tfpDirichletMultinomialPerm(kmer_total_counts, concentrations, name='x')
    return x


def _ar_kmer_counts(kmer_seqs, kmer_total_counts, ref_counts, ar_func):
    """Get Random variable of kmer transition counts conditioned on observed counts.

    Parameters
    ----------
    kmer_seqs : dtype
        A tensor of shape [A1, ..., An, lag, alphabet_size+1] of one-hot encoded kmers.
    kmer_total_counts : int
        A tensor of shape [A1, ..., An] of counts of each kmer.
    ar_func : function, default = None
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and another of shape [A1, ..., An, alphabet_size+1] and returns a
        tensor of shape [A1, ..., An, alphabet_size+1] of dtype of transition probabilities
        for each kmer. The autoregressive function.

    Returns
    -------
    x : tensorflow probability distribution
        Distribution of kmer transition counts for an AR model.
    """
    probs = ar_func(kmer_seqs, ref_counts) + epsilon
    x = core.tfpMultinomialPerm(kmer_total_counts, probs, name='x')
    return x


def _create_params(lag, alphabet_size, make_ar_func,
                   af_kwargs, dtype=tf.float64):
    """Define and get parameters of BEAR or AR distribution.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns a ar_func.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    dtype : tensorflow dtype, default = tf.float64
        dtype for the ar_func and h.

    Returns
    -------
    params: list
        List of parameters as tensorflow variables.
    h_signed : dtype
        log(h) where h is the BEAR parameter.
    ar_func : function
        The autoregressive function.
    """
    ar_func, ar_func_params = _make_ref_ar_func(lag, alphabet_size, make_ar_func, af_kwargs, dtype)
    h_signed = tf.Variable(0, dtype=dtype, name='h_signed')
    params = ([h_signed] + ar_func_params)
    return params, h_signed, ar_func


def change_scope_params(lag, alphabet_size, make_ar_func,
                        af_kwargs, params, dtype=tf.float64):
    """Redefine and get parameters of BEAR or AR distribution in given scope.

    Used to to get unmirrored variables after training on multiple GPUs in parallel
    or to unpack a list of params into h and the autoregressive function.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns a ar_func.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    params: list
        List of parameters as tensorflow variables.
    dtype : dtype, default = tf.float64
        dtype for the ar_func and h.

    Returns
    -------
    params : list
        List of parameters as tensorflow variables.
    h_signed : dtype
        log(h) where h is the BEAR parameter.
    ar_func : function
        The autoregressive function.
    """
    pos_in_params = 0
    h_nmir = tf.Variable(0, dtype=dtype, name='h_signed')
    h_nmir.assign(params[pos_in_params])
    pos_in_params += 1
    ar_func_nmir, ar_func_params_nmir = _make_ref_ar_func(lag, alphabet_size, make_ar_func, af_kwargs, dtype)
    for param in ar_func_params_nmir:
        param.assign(params[pos_in_params])
        pos_in_params += 1
    params_nmir = ([h_nmir] + ar_func_params_nmir)
    return params_nmir, h_nmir, ar_func_nmir


def _train_step(batch, num_kmers, h_signed, ar_func,
                params, acc_grads, train_ar):
    """Add gradient of unbiased estimate of loss to accumulated gradients.

    Parameters
    ----------
    batch : list of two tensors of the same dtype.
        The first element is a one hot encoding of kmers of shape
        [kmer_batch_size, lag, alphabet_size+1] and the second is the transition
        counts of size [kmer_batch_size, alphabet_size+1].
    num_kmers : int
        Total number of kmers seen in data. Unsed to normalize estimate of loss.
    h_signed : dtype
        log(h) where h is the BEAR parameter.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
    params : list
        List of parameters as tensorflow variables.
    acc_grads : list
        List of accumulated gradients for parameters.
    train_ar : bool
        Whether to evaluate the likelihood using an AR (True) or BEAR (False) model.

    Returns
    -------
    loss : dtype
        An unbiased estimate of the log likelihood of the data.
    """
    with tf.GradientTape() as grad_tape:
        kmer_batch_size = tf.shape(batch[0])[0]
        kmer_seqs = batch[0]
        transition_counts = batch[1]
        kmer_total_counts = tf.math.reduce_sum(transition_counts, axis=-1)
        ref_counts = batch[2]

        if train_ar:
            post = _ar_kmer_counts(kmer_seqs, kmer_total_counts, ref_counts, ar_func)
        else:
            post = _bear_kmer_counts(kmer_seqs, kmer_total_counts, ref_counts,
                                     h=tf.math.exp(h_signed), ar_func=ar_func)
        log_likelihood = tf.reduce_sum(
            post.counts_log_prob(transition_counts))

        elbo = (num_kmers / kmer_batch_size) * log_likelihood
        loss = -elbo

    gradients = grad_tape.gradient(loss, params)
    for tv, grad in zip(acc_grads, gradients):
        if grad is not None:
            tv.assign_add(grad)
    return loss


def train(data, num_kmers, epochs, ds_loc, ds_loc_ref, alphabet, lag, make_ar_func, af_kwargs,
          learning_rate, optimizer_name, train_ar, acc_steps=1,
          params_restart=None, writer=None, dtype=tf.float64):
    """Train a BEAR or AR model with reference transition counts using all available GPUs in parallel.

    Parameters
    ----------
    data : tensorflow data object
        Load sequence data using tools in dataloader.py. Minibatch before passing.
    num_kmers : int
        Total number of kmers seen in data. Unsed to normalize estimate of loss.
    epochs : int
    ds_loc : int
        Column in count data that coresponds with the training data.
    ds_loc_ref : int
        Column in count data that coresponds with the reference data.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    lag : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns an ar_func.  See ar_funcs submodule.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    learning_rate : float
    optimizer_name : str
        For example 'Adam'.
    train_ar : bool
        Whether to train an AR (True) or BEAR (False) model.
    writer : tensorboard writer object, default = None
    acc_steps : int, default = 1
        Number of steps to accumulate gradients over.
    params_restart : list of tensorflow variables, default = None
        Pass the parameter list from a previous run to restart training.
    dtype : dtype, default=tf.float64

    Returns
    -------
    params: list
        List of parameters as tensorflow variables.
    h_signed : dtype
        log(h) where h is the BEAR parameter.
    ar_func : function
        The autoregressive function.
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Define parameters in parallel GPU usage scope.
        if params_restart is None:
            params, h_signed, ar_func = _create_params(
                lag, alphabet_size, make_ar_func,
                af_kwargs, dtype=dtype)
        else:
            params, h_signed, ar_func = change_scope_params(
                lag, alphabet_size, make_ar_func,
                af_kwargs, params_restart, dtype=dtype)

        # Set up accumulated gradient variables.p;';
        acc_grads = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in params]
        for tv in acc_grads:
            tv.assign(tf.zeros_like(tv))

        # Define optimizer in parallel GPU usage scope.
        optimizer = getattr(tf.keras.optimizers, optimizer_name)(
                                    learning_rate=learning_rate)

    # One hot encode kmers and get appropriate column from training data.
    not_stop = (1-tf.eye(alphabet_size+1, dtype=dtype)[-1])

    def map_(kmers, counts):
        return (core.tf_one_hot(kmers, alphabet),
                tf.gather(counts, ds_loc, axis=1),
                (tf.gather(counts, ds_loc_ref, axis=1) + epsilon) * not_stop)
    data = data.map(
        map_, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(10)
    # Get data iterable for use with GPU parallelization.
    data_iter = iter(strategy.experimental_distribute_dataset(data))

    # Define functions for updating parameters and accumulating gradients
    # with GPU parallelization.
    def add_grads(params, acc_grads, optimizer):
        optimizer.apply_gradients(zip(acc_grads, params))

    @tf.function
    def dist_add_grads(params, acc_grads, optimizer):
        strategy.run(add_grads, args=(params, acc_grads, optimizer))

    @tf.function
    def dist_train_step(batch, num_kmers, h_signed, ar_func,
                        params, acc_grads, train_ar):
        losses = strategy.run(_train_step, args=(
            batch, num_kmers, h_signed, ar_func,
            params, acc_grads, train_ar))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses,
                               axis=None)

    # Training loop
    loss = 0.
    step = 1
    for batch in data_iter:
        # Accumulate gradients and update loss.
        loss += dist_train_step(batch, num_kmers, h_signed, ar_func,
                                params, acc_grads, train_ar)

        if step % acc_steps == 0:
            # Record loss
            if writer is not None:
                with writer.as_default():
                    tf.summary.scalar('elbo', - loss / acc_steps, step=step)
            # Update gradients
            dist_add_grads(params, acc_grads, optimizer)
            # Reset accumulated gradients and cumulative loss to zero.
            for tv in acc_grads:
                tv.assign(tf.zeros_like(tv))
            loss = 0

        step += 1

    # Remove the distributed scope from the parameters.
    params, h_signed, ar_func = change_scope_params(
        lag, alphabet_size, make_ar_func,
        af_kwargs, params, dtype=dtype)
    return params, h_signed, ar_func


@tf.function
def evaluation(data, ds_loc_train, ds_loc_test, ds_loc_ref,
               alphabet, h, ar_func, van_reg, dtype=tf.float64):
    """Evaluate a trained BEAR, BMM, or AR model.

    Parameters
    ----------
    data : tensorflow data object
        Load sequence data using tools in dataloader.py. Minibatch before passing.
    ds_loc_train : int
        Column in count data that coresponds with the training data.
    ds_loc_test : int
        Column in count data that coresponds with the testing data.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    h : dtype
        A positive constant of the h parameter from the BEAR model.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and another of shape [A1, ..., An, alphabet_size+1] and returns a
        tensor of shape [A1, ..., An, alphabet_size+1] of dtype of transition probabilities
        for each kmer. The autoregressive function.
    van_reg : float
        Prior on BMM model.
    dtype : dtype, default=tf.float64

    Returns
    -------
    log_likelihood_ear, log_likelihood_arm, log_likelihood_van : float
        Total log likelihood of the data with the model evaluated as a BEAR,
        AR or BMM model.
    perplexity_ear, perplexity_arm, perplexity_van : float
        Perplexity of the data with the model evaluated as a BEAR,
        AR or BMM model.
    accuracy_ear, accuracy_arm, accuracy_van : float
        Accuracy of the data with the model evaluated as a BEAR,
        AR or BMM model. Ties are resolved randomly.
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1

    # Stops are removed from the reference counts as they are unlikely to be representative of
    # stops in read data.
    not_stop = (1-tf.eye(alphabet_size+1, dtype=dtype)[-1])

    def map_(kmers, counts):
        return (core.tf_one_hot(kmers, alphabet),
                tf.gather(counts, ds_loc_train, axis=1),
                tf.gather(counts, ds_loc_test, axis=1),
                (tf.gather(counts, ds_loc_ref, axis=1) + epsilon) * not_stop)
    data_iter = iter(data.map(map_))

    log_likelihood_ear = tf.constant(0., dtype=dtype)
    log_likelihood_arm = tf.constant(0., dtype=dtype)
    log_likelihood_van = tf.constant(0., dtype=dtype)
    correct_ear = tf.constant(0., dtype=dtype)
    correct_arm = tf.constant(0., dtype=dtype)
    correct_van = tf.constant(0., dtype=dtype)
    total_len = tf.constant(0., dtype=dtype)

    for batch in data_iter:
        kmer_seqs = batch[0]
        transition_counts_train = batch[1]
        transition_counts_test = batch[2]
        kmer_total_counts_test = tf.math.reduce_sum(transition_counts_test, axis=-1)
        ref_counts = batch[3]
        # Get posteriors.
        post_ear = _bear_kmer_counts(kmer_seqs, kmer_total_counts_test, ref_counts,
                                     condition_trans_counts=transition_counts_train,
                                     h=h, ar_func=ar_func)
        post_arm = _ar_kmer_counts(kmer_seqs, kmer_total_counts_test, ref_counts, ar_func)
        post_van = _bear_kmer_counts(kmer_seqs, kmer_total_counts_test, ref_counts,
                                     condition_trans_counts=transition_counts_train + van_reg)
        # Get likelihoods.
        log_likelihood_ear += tf.reduce_sum(
            post_ear.counts_log_prob(transition_counts_test))
        log_likelihood_arm += tf.reduce_sum(
            post_arm.counts_log_prob(transition_counts_test))
        log_likelihood_van += tf.reduce_sum(
            post_van.counts_log_prob(transition_counts_test))
        # Get most likely transition and accuracy.
        ml_ear = post_ear.ml_output()
        ml_arm = post_arm.ml_output()
        ml_van = post_van.ml_output()
        oh_ml_ear = tf.cast(tf.math.equal(ml_ear[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        oh_ml_arm = tf.cast(tf.math.equal(ml_arm[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        oh_ml_van = tf.cast(tf.math.equal(ml_van[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        correct_ear += tf.math.reduce_sum(transition_counts_test*oh_ml_ear)
        correct_arm += tf.math.reduce_sum(transition_counts_test*oh_ml_arm)
        correct_van += tf.math.reduce_sum(transition_counts_test*oh_ml_van)
        # Sum total number of transitions.
        total_len += tf.math.reduce_sum(transition_counts_test)
    return (log_likelihood_ear, log_likelihood_arm, log_likelihood_van,
            tf.exp(-log_likelihood_ear/total_len),
            tf.exp(-log_likelihood_arm/total_len),
            tf.exp(-log_likelihood_van/total_len),
            correct_ear/total_len, correct_arm/total_len, correct_van/total_len)
