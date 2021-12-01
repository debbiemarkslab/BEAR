import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import dill
from . import bear_net
from . import core

epsilon = tf.keras.backend.epsilon()

def _cross_str_arrays(array1, array2, exch='X'):
    """Get cross product of two string arrays.
    
    Parameters
    ----------
    exch : char
        Must not appear in either set of strings
    """
    x_array2 = np.array([exch+b for b in array2])
    cross = np.char.join(array1[:, None], x_array2[None, :]).flatten()
    return np.char.lstrip(cross, exch)

def _load_bear(config, path):
    config = configparser.ConfigParser()
    config.read(path + '/config.cfg')

    dtype = getattr(tf, config['general']['precision'])
    lag = int(config['hyperp']['lag'])
    config['data']['alphabet']
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1
    make_ar_func = getattr(ar_funcs, 'make_ar_func_'+config['model']['ar_func_name'])
    af_kwargs = json.loads(config['model']['af_kwargs'])

    results = dill.load(open(os.path.join(path, 'results.pickle'), 'rb'))
    params_restart = results['params']
    (params, h_signed, ar_func) = bear_net.change_scope_params(
            lag, alphabet_size, make_ar_func, af_kwargs, params_restart, dtype=dtype)
    h = np.exp(h_signed.numpy())
    return lag, alphabet, h, decoder

def _get_all_kmers(vars_, wt_seq, lag):
    """Get all kmers relevant to the calculation of variant probabilities wihtout repeats.
    
    Paramters
    ---------
    vars_ : numpy array
        Of the format (wt_aa) (position in wt_seq without start symbols) (var_aa) without spaces or barckets.
    wt_seq : str
        sequence with start and end symbols.
    lag : int
    
    Returns
    -------
    all_kmers : numpy array
    """
    all_kmers = []
    for (wt_aa, mt_aa, pos) in vars_:
        pos = pos + lag
        assert wt_aa == wt_seq[pos:pos+1]
        wt_win = wt_seq[pos-lag:pos+lag+1]
        wt_kmers = [wt_win[i:i+lag] for i in range(len(wt_win)-lag)]
        mt_win = (wt_seq[pos-lag:pos] + mt_aa + wt_seq[pos+1:pos+lag+1])
        mt_kmers = [mt_win[i:i+lag] for i in range(len(mt_win)-lag)]
        all_kmers = all_kmers + list(wt_kmers) + list(mt_kmers)
    return np.array(list(set(all_kmers))).astype(str)


def _get_pdf(kmers, counts, h, decoder, mc_samples, vans, train_col, alphabet, get_map):
    """Get probabilities of all k+1-mer transitions.
    
    Parameters
    ----------
    kmers : tf tensor of str
    counts : tf tensor
    h : float
    decoder : function
    mc_samples : int
    vans : numpy array of floats
    train_col : int
    alphabet : str
    
    Returns
    -------
    df : pandas dataframe
        A dataframe indexed by k+1-mers and with sampled transition probabilities as each of the columns.
        One can pass a k+1-mer to df to get the transition probability samples from each model.
    """
    alphabet_size = len(core.alphabets_en['prot'])
    
    # take only the counts from the training column
    counts = counts[:, train_col, :]
    # get all k+1-mers
    kp1mers = _cross_str_arrays(kmers, core.alphabets_en[alphabet], exch='X')
    
    # get the decoder values on all kmers
    num_models = len(vans)
    alpha = np.array(vans)[:, None, None]*tf.ones([len(kmers), alphabet_size], dtype=tf.float64)[None, ...]
    if h != None:
        num_models += 1
        dec_vals = (tf.nn.softmax(decoder(core.tf_one_hot(kmers, 'prot')))+epsilon)[None, :, 0, :]/h[:, None, None]
        alpha = tf.concat([dec_vals, alpha], axis=0)
    concs = alpha + counts[None, :, :]
    if h != None and get_map:
        num_models += 1
        alpha = tf.concat([dec_vals[None, ...], concs], axis=0)
    
    # sample probabilities for each k+1-mer
    if get_map:
        log_probs = np.log(concs/(np.sum(concs, axis=-1)[..., None])).reshape([1, num_models, -1])
    else:
        log_probs = np.log(tfp.distributions.Dirichlet(concs).sample(mc_samples).numpy().reshape([mc_samples, num_models, -1]))
    
    # Build df
    pd_dict = {'kmer': np.array(kp1mers).astype(str)}
    for i in range(num_models):
        pd_dict['prob_{}'.format(i)] = log_probs[:, i, :].T
    df = pd.concat([pd.DataFrame(v) for k, v in pd_dict.items()], axis=1, keys=pd_dict.keys()).set_index(('kmer', 0))
    return df


def _add_kmer_probs(vars_, scores, wt_seq, pdf, lag, seen_kmers, mc_samples):
    num_models = len(pdf.iloc[0, :])//mc_samples
    for i, (wt_aa, mt_aa, pos) in enumerate(vars_):
        pos = pos + lag
        assert wt_aa == wt_seq[pos:pos+1]
        # get wt and mutant k+1mers that have their first k letters in the batch (seen_kmers)
        wt_win = wt_seq[pos-lag:pos+lag+1]
        wt_kmers = [wt_win[i:i+lag+1] for i in range(len(wt_win)-lag) if wt_win[i:i+lag] in seen_kmers]
        mt_win = (wt_seq[pos-lag:pos] + mt_aa + wt_seq[pos+1:pos+lag+1])
        mt_kmers = [mt_win[i:i+lag+1] for i in range(len(mt_win)-lag) if mt_win[i:i+lag] in seen_kmers]
        # if the wt or mt kmers are in the df, the log probability will be added to the score
        scores[i, :, :] += (np.sum(pdf.loc[mt_kmers].to_numpy().reshape([-1, num_models, mc_samples]), axis=0)
                            - np.sum(pdf.loc[wt_kmers].to_numpy().reshape([-1, num_models, mc_samples]), axis=0))
    return scores


def get_bear_probs(bear_path, data, wt_seq, vars_, train_col,
                   mc_samples=41, vans=[0.1, 1, 10], lag=None, alphabet=None, get_map=False, h=None):
    """Sample posterior predictive probabilities of variants under BEAR by looping through batches of kmers.
    
    Parameters
    ----------
    bear_path : str
        None if using BMM and lag is specified.
    data : tf dataset
        generator of kmers and counts.
    wt_seq : str
        Must not be bytes!
    vars_ : numpy array of str
        Of the format (wt_aa) (position in wt_seq without start symbols) (var_aa) without spaces or barckets.
        Must not be bytes!
    train_col : int
    mc_samples : int, default = 41
        Number of samples to take from the posterior predictive.
    vans : numpy array, default = [0.1, 1, 10]
        vanilla regularization to use for BMM models.
    lag : int, default = None
        Specify if not using BEAR.
    alphabet :str, default = None
        Specify if not using BEAR.
    get_map : bool
        Gets the MAP
    h : numpy array
        For h scans if bear_folder is specified. None if using fit bear h.
        
    Returns
    -------
    scores : numpy array of floats
        [num variants, num models (whather or not to use BEAR + len(vans)), mc_samples].
    """        
    # load bear from bear_path
    if bear_path != None:
        lag, alphabet, h_bear, decoder = load_bear(bear_path)
        if h == None:
            h = np.array([h_bear])
        len_h = len(h)
    else:
        assert lag != None and alphabet != None
        len_h = 0
        decoder = None

    # pad wt_seq from pname
    wt_seq = lag*'['+wt_seq+']'
    vars_ = [(var[0], var[-1], int(var[1:-1])) for var in vars_]

    # get list of all possible kmers
    all_kmers = _get_all_kmers(vars_, wt_seq, lag)
    
    # no sampling if just using the MAP
    if get_map:
        mc_samples = 1
        h = np.array([1.])
        len_h = 1
    num_models = (len(h)!=0)*(len_h + get_map) + len(vans)

    scores = np.zeros([len(vars_), num_models, mc_samples])
    seen_all_kmers = np.zeros(len(all_kmers))  
    
    for kmers, counts in iter(data):
        kmers = kmers.numpy().astype(str)
        print(kmers)
        # first throw out kmers in the batch that can't contribute to the variant scores
        in_kmers = np.isin(kmers, all_kmers)
        print("num seen kmers in this batch:", np.sum(in_kmers))
        seen_kmers = kmers[in_kmers]
        seen_counts = counts[in_kmers]
        # make a record of having seem the kmer
        seen_all_kmers += np.isin(all_kmers, seen_kmers)
        if np.sum(in_kmers)>0:
            # get probabilities of all transitions out of each kmer
            pdf = _get_pdf(seen_kmers, seen_counts, h, decoder, mc_samples, vans, train_col, alphabet, get_map)
            # goes through all mutants and add the probabilities contributed by this batch of kmers
            scores = _add_kmer_probs(vars_, scores, wt_seq, pdf, lag, seen_kmers, mc_samples)

    # some kmers haven't been seen but still affect the probabilities through prior values
    unseen_kmers = (seen_all_kmers == 0)
    print("num unseen kmers:", sum(unseen_kmers))
    if sum(unseen_kmers)>0:
        pdf = _get_pdf(all_kmers[unseen_kmers],
                       tf.zeros([sum(unseen_kmers), train_col+1, 20+1], dtype=tf.float64),
                       h, decoder, mc_samples, vans, train_col, alphabet, get_map)
        scores = _add_kmer_probs(vars_, scores, wt_seq, pdf, lag, all_kmers[unseen_kmers].astype(str),
                                 mc_samples)
    if get_map:
        scores = scores[..., 0]
    return scores
