import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import dill
import json
import os
import configparser
from Bio import Seq
from . import bear_net
from . import ar_funcs
from . import core
from . import dataloader

epsilon = tf.keras.backend.epsilon()

def cross_str_arrays(array1, array2, exch='X'):
    """Get cross product of two string arrays.
    
    Parameters
    ----------
    exch : char
        Must not appear in either set of strings
    """
    x_array2 = np.array([exch+b for b in array2])
    cross = np.char.join(array1[:, None], x_array2[None, :]).flatten()
    return np.char.lstrip(cross, exch)

def load_ds(files_path, start_token, kmer_batch_size, sparse,
            alphabet, num_ds, dtype=tf.float64):
    files = [os.path.join(files_path, file) for file in os.listdir(files_path)
                     if file.startswith(start_token)]
    if sparse:
        dataload_func = dataloader.sparse_dataloader
    else:
        dataload_func = dataloader.dataloader
    if len(files) == 1:
        data = dataload_func(files[0], alphabet,
                             kmer_batch_size, num_ds,
                             cache=False,
                             dtype=dtype)
    else:
        # Interleave loaded data from multiple files.
        data = tf.data.Dataset.from_tensor_slices(files)
        data = data.interleave(
            lambda x: dataload_func(x, alphabet,
                                    kmer_batch_size, num_ds,
                                    cache=False,
                                    dtype=dtype),
            cycle_length=4, num_parallel_calls=4, deterministic=False)
    return data

def load_bear(path):
    config = configparser.ConfigParser()
    config.read(path + '/config.cfg')

    dtype = getattr(tf, config['general']['precision'])
    lag = int(config['hyperp']['lag'])
    alphabet = config['data']['alphabet']
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1
    make_ar_func = getattr(ar_funcs, 'make_ar_func_'+config['model']['ar_func_name'])
    af_kwargs = json.loads(config['model']['af_kwargs'])

    results = dill.load(open(os.path.join(path, 'results.pickle'), 'rb'))
    params_restart = results['params']
    (params, h_signed, ar_func) = bear_net.change_scope_params(
            lag, alphabet_size, make_ar_func, af_kwargs, params_restart, dtype=dtype)
    h = np.exp(h_signed.numpy())
    
    data = load_ds(config['data']['files_path'], config['data']['start_token'], int(config['train']['batch_size']),
                   config['data']['sparse'] == 'True', alphabet, int(config['data']['num_ds']))
    
    @tf.function
    def ar_func_tf(kmers):
        return tf.nn.softmax(ar_func(kmers)) + epsilon
    return lag, alphabet, h, ar_func_tf, data

def df_to_func(df, num_models, mc_samples, summed=True):
    if summed:
        prob_func = lambda kp1mers_ex: np.sum(df.loc[kp1mers_ex].to_numpy().reshape([-1, num_models, mc_samples]), axis=0)
    else:
        prob_func = lambda kp1mers_ex: df.loc[kp1mers_ex].to_numpy().reshape([-1, num_models, mc_samples])
    return prob_func
    
def get_pdf(kmers, counts, h, ar_func, mc_samples, vans, train_col, alphabet, get_map, summed=True, output='func'):
    """Get probabilities of all k+1-mer transitions. Uses a dataframe indexed by k+1-mers
    and with sampled transition probabilities as each of the columns. One can pass a k+1-mer
    to df to get the transition probability samples from each model.
    
    Parameters
    ----------
    kmers : tf tensor of str
    counts : tf tensor
    h : float
    ar_func : function
    mc_samples : int
    vans : numpy array of floats
    train_col : int
    alphabet : str
    summed : bool, default=True
        Whether prob func should sum the probabilities of all kp1-mer transitions (True)
        or return the matrix of probabilities with first dimension kmp1-mers (False)
    output : str, default='func'
        Whether to get a function that returns probabilities from a bunch of kp1-mers ("func"), a df ("df"),
        or a numpy array (shape [kmer, letter, model, mc_samples]) ("numpy").
    
    Returns
    -------
    prob_func : function
        Given a list of k+1-mers, returns [num_models, mc_samples] probability matrix
        contributed by those transitions. Only takes as input k+1-mers starting with an element
        of the kmers parameter.
    """
    alphabet_size = len(core.alphabets_en[alphabet]) - 1
    if get_map:
        mc_samples = 1
    
    # take only the counts from the training column
    counts = counts[:, train_col, :]
    
    # get the ar_func values on all kmers
    num_models = len(vans)
    if len(vans) > 0:
        # laod van concs
        alpha = np.array(vans)[:, None, None]*np.ones([len(kmers), alphabet_size + 1])[None, ...]
    if ar_func != None:
        # load BEAR concs
        num_models += len(h)
        ar_vals = ar_func(core.tf_one_hot(kmers, alphabet)).numpy()
        dec_vals = ar_vals[None, :, :]/h[:, None, None]
        #concatenate the van and BEAR concs
        if len(vans) > 0:
            alpha = np.concatenate([dec_vals, alpha], axis=0)
        else:
            alpha = dec_vals
    concs = alpha + counts[None, :, :]
    if ar_func != None and get_map:
        # also add the AR
        num_models += 1
        concs = np.concatenate([ar_vals[None, ...], concs], axis=0)
    
    # sample probabilities for each k+1-mer
    if get_map:
        log_probs = np.log(concs/(np.sum(concs, axis=-1)[..., None])).reshape([1, num_models, -1])
    else:
        log_probs = np.random.standard_gamma(concs, size=np.r_[[mc_samples], np.shape(concs)])
        log_probs = np.log(log_probs / np.sum(log_probs, axis=-1)[..., None])
        # log_probs = np.log(tfp.distributions.Dirichlet(concs).sample(mc_samples).numpy().reshape([mc_samples, num_models, -1]))
    
    # Build df
    if output == 'numpy':
        return np.transpose(log_probs.reshape([mc_samples, num_models, len(kmers), alphabet_size+1]), [2, 3, 1, 0])
    else:
        # get all k+1-mers
        kp1mers = cross_str_arrays(kmers, core.alphabets_en[alphabet], exch='X')
        
        pd_dict = {'kmer': np.array(kp1mers).astype(str)}
        for i in range(num_models):
            pd_dict['prob_{}'.format(i)] = log_probs[:, i, :].T
        df = pd.concat([pd.DataFrame(v) for k, v in pd_dict.items()], axis=1, keys=pd_dict.keys()).set_index(('kmer', 0))
        if output == 'df':
            df.columns = np.arange(len(df.columns))
            return df
        elif output == 'func':
            prob_func = df_to_func(df, num_models, mc_samples, summed=summed)
            return prob_func

def get_kmc_count(kmer, kmc_file, kmer_token, c):
    kmer_token.from_string(kmer)
    count = int(kmc_file.CheckKmer(kmer_token, c))
    if count == 1:
        count = c.value
    return count

def load_kmc(path):
    file = kmc.KMCFile()
    if not file.OpenForRA(path):
        raise FileNotFoundError('KMC file not found.')
    print("loaded", path)
    return file

def make_kmc_genome_counter(path, lag, reverse=True, no_end=False):
    """ Get a function that takes a batch of kmers and returns transition counts.
    End symbol is 0 because ends in assemblies aren't reliable.
    
    Parameters
    ----------
    kmc_path : str
        Path to kmc file with counts.
    lag : int
    reverse : bool, default=True
        Whether to include counts of the reverse complement of kmers as well.
    no_end : bool, default=False
        Don't load kmc files for starts and ends and assume kmers don't end.
        In this case you can enter the exact res file.
    
    Returns
    -------
    counter : function
        Takes kmer strings and returns transition counts.
    """
    global kmc
    import py_kmc_api as kmc
    
    alphabet = core.alphabets_en['dna'][:-1]
    alphabet_size = len(alphabet)
    
    # create tokens for calling kmc
    kmer_token = kmc.KmerAPI(lag+1)
    c = kmc.Count()
    
    if no_end:
        # Load kmc file into memory
        print("loading", path)
        if '.res' not in path:
            path = path + '_kmc_inter_0_full_{}.res'.format(lag+1)
        file = load_kmc(path)
        def counter(kmers):
            final_shape = np.r_[np.shape(kmers), [alphabet_size+1]]
            counts = np.zeros([np.size(kmers), alphabet_size+1])
            for i, k in enumerate(kmers.flatten()):
                for j, b in enumerate(alphabet):
                    # Get kp1mer count
                    counts[i, j] = get_kmc_count(k + b, file, kmer_token, c)
                    # Get reverse count (assemblies only look at one strand).
                    if reverse:
                        counts[i, j] += get_kmc_count(Seq.reverse_complement(k + b),
                                                      file, kmer_token, c)
            return counts.reshape(final_shape)
    else:
        # Load kmc file into memory
        print("loading", path)
        files = []
        files_suf = []
        for l in np.arange(lag) + 1:
            files.append(load_kmc(path + '_kmc_inter_0_pre_{}.res'.format(l)))
            files_suf.append(load_kmc(path + '_kmc_inter_0_suf_{}.res'.format(l)))
        files.append(load_kmc(path + '_kmc_inter_0_full_{}.res'.format(lag+1)))
        def counter(kmers):
            final_shape = np.r_[np.shape(kmers), [alphabet_size+1]]
            counts = np.zeros([np.size(kmers), alphabet_size+1])
            for i, k in enumerate(kmers.flatten()):
                k = k.replace('[', '')
                for j, b in enumerate(alphabet):
                    # Get kp1mer count
                    counts[i, j] = get_kmc_count(k + b, files[len(k)], kmer_token, c)
                    # Get reverse count (assemblies only look at one strand).
                    if reverse:
                        if len(k) == lag:
                            counts[i, j] += get_kmc_count(Seq.reverse_complement(k + b),
                                                          files[len(k)], kmer_token, c)
                        if len(k) < lag:
                            counts[i, j] += get_kmc_count(Seq.reverse_complement(k + b),
                                                          files_suf[len(k)], kmer_token, c)
                if len(k) == lag:
                    counts[i, -1] = get_kmc_count(k, files_suf[len(k)-1], kmer_token, c)
                    if reverse:
                        counts[i, -1] += get_kmc_count(Seq.reverse_complement(k),
                                                       files[len(k)-1], kmer_token, c)
            return counts.reshape(final_shape)
    return counter


##############################################VARIANTS##########################################
def _add_kmer_probs_vars(vars_, scores, wt_seq, pdf, lag, seen_kmers):
    for i, (wt_aa, mt_aa, pos) in enumerate(vars_):
        pos = pos + lag
        assert wt_aa == wt_seq[pos:pos+1]
        # get wt and mutant k+1mers that have their first k letters in the batch (seen_kmers)
        wt_win = wt_seq[pos-lag:pos+lag+1]
        wt_kmers = [wt_win[i:i+lag+1] for i in range(len(wt_win)-lag) if wt_win[i:i+lag] in seen_kmers]
        mt_win = (wt_seq[pos-lag:pos] + mt_aa + wt_seq[pos+1:pos+lag+1])
        mt_kmers = [mt_win[i:i+lag+1] for i in range(len(mt_win)-lag) if mt_win[i:i+lag] in seen_kmers]
        # if the wt or mt kmers are in the df, the log probability will be added to the score
        scores[i, :, :] += (pdf(mt_kmers) - pdf(wt_kmers))
    return scores


def _get_all_kmers_vars(vars_, wt_seq, lag):
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


def get_bear_probs(bear_path, wt_seq, vars_, train_col,
                   mc_samples=41, vans=[0.1, 1, 10], get_map=False,
                   lag=None, alphabet=None, h=None, data=None,
                   kmc_path=None, kmc_reverse=False):
    """Sample posterior predictive probabilities of variants under BEAR by looping through batches of kmers.
    
    Parameters
    ----------
    bear_path : str
        Path to folder of trained BEAR model. None if using BMM.
    wt_seq : str
        Wild type sequence. Must not be bytes!
    vars_ : numpy array of str
        List of SNPs.
        Of the format (wt_aa) (position in wt_seq without start symbols) (var_aa) without spaces or barckets.
        For example ['A0T','C45G'].
        Must not be bytes!
    train_col : int
        Row of counts data that includes the training data
    mc_samples : int, default = 41
        Number of samples to take from the posterior predictive of the mutation probabilities.
    vans : numpy array, default = [0.1, 1, 10]
        vanilla regularization to use for BMM models.
    get_map : bool
        Gets the probability of the mutations under the MAP model under BEAR instead of sampling models from BEAR.
    lag : int, default = None
        Specify if not using BEAR.
    alphabet :str, default = None
        Specify if not using BEAR.
    h : numpy array
        For h scans if bear_folder is specified. None if using fit BEAR h.
    data : tf dataset
        Generator of kmers and counts. Specify if not using BEAR.
    kmc_path : str
        Specify the path kmc files if one wishes to use kmc to count kmers instead of cycling through whole dataset.
    kmc_reverse : bool, default=False
        Whether to include counts of the reverse complement of kmers when counting using kmc.
        
    Returns
    -------
    scores : numpy array of floats
        [num variants, num models ((1 for the AR model if using BEAR and get_MAP=True) + len(hs) + len(vans)), mc_samples].
    """            
    # load bear from bear_path
    if bear_path != None:
        lag, alphabet, h_bear, ar_func, data = load_bear(bear_path)
        if h is None:
            h = np.array([h_bear])
        len_h = len(h)
    else:
        assert ((lag is not None and alphabet is not None)
                and ((data is not None or kmc_path is not None) and len(vans) > 0))
        if kmc_path is not None:
            assert alphabet == 'dna' and train_col == 0
            #TODO inflate kmc counts to include case where train col of ar_func is not 0
        len_h = 0
        ar_func = None
    alphabet_size = len(core.alphabets_en[alphabet])-1

    # pad wt_seq from pname
    wt_seq = lag*'['+wt_seq+']'
    vars_ = [(var[0], var[-1], int(var[1:-1])) for var in vars_]

    # get list of all possible kmers
    all_kmers = _get_all_kmers_vars(vars_, wt_seq, lag)
    
    # no sampling if just using the MAP
    if get_map:
        mc_samples = 1
    num_models = (ar_func is not None)*(len_h + get_map) + len(vans)

    scores = np.zeros([len(vars_), num_models, mc_samples]) 
    
    if kmc_path is not None:
        counter = make_kmc_genome_counter(kmc_path, lag, reverse=kmc_reverse)
        all_counts = counter(all_kmers)[:, None, :]
        print(all_kmers, all_counts)
        if np.all(all_counts == 0):
            print("no kmers found, are you sure you have the correct kmc file and lag?")
        pdf = get_pdf(all_kmers, all_counts, h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
        scores = _add_kmer_probs_vars(vars_, scores, wt_seq, pdf, lag, all_kmers)
    else:
        seen_all_kmers = np.zeros(len(all_kmers)) 
        for kmers, counts in iter(data):
            kmers = kmers.numpy().astype(str)
            # first throw out kmers in the batch that can't contribute to the variant scores
            in_kmers = np.isin(kmers, all_kmers)
            print("num seen kmers in this batch:", np.sum(in_kmers))
            seen_kmers = kmers[in_kmers]
            seen_counts = counts[in_kmers]
            # make a record of having seem the kmer
            seen_all_kmers += np.isin(all_kmers, seen_kmers)
            if np.sum(in_kmers)>0:
                # get probabilities of all transitions out of each kmer
                pdf = get_pdf(seen_kmers, seen_counts, h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
                # goes through all mutants and add the probabilities contributed by this batch of kmers
                scores = _add_kmer_probs_vars(vars_, scores, wt_seq, pdf, lag, seen_kmers)
        # some kmers haven't been seen but still affect the probabilities through prior values
        unseen_kmers = (seen_all_kmers == 0)
        print("num unseen kmers:", sum(unseen_kmers))
        if sum(unseen_kmers)>0:
            pdf = get_pdf(all_kmers[unseen_kmers],
                           tf.zeros([sum(unseen_kmers), train_col+1, alphabet_size+1], dtype=tf.float64),
                           h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
            scores = _add_kmer_probs_vars(vars_, scores, wt_seq, pdf, lag, all_kmers[unseen_kmers].astype(str))
    if get_map:
        scores = scores[..., 0]
    return scores

##############################################whole_seqs##########################################
def _add_kmer_probs_seqs(seqs, scores, pdf, lag, seen_kmers):
    for i, seq in enumerate(seqs):
        kmers = [seq[i:i+lag+1] for i in range(len(seq)-lag) if seq[i:i+lag] in seen_kmers]
        # if the wt or mt kmers are in the df, the log probability will be added to the score
        scores[i, :, :] += pdf(kmers)
    return scores

def _get_all_kmers_seqs(seqs, lag):
    """Get all kmers relevant to the calculation of seq probabilities (i.e. not including ]) wihtout repeats.
    
    Paramters
    ---------
    seqs : list
    lag : int
    
    Returns
    -------
    all_kmers : numpy array
    """
    all_kmers = []
    for seq in seqs:
        #short seqs cannot be kmc'd properly: kp1-mers with both [ and ]
        assert len(seq.replace('[', '').replace(']', '')) >= lag
        kmers = [seq[i:i+lag] for i in range(len(seq)-lag)]
        all_kmers = all_kmers + list(kmers)
    return np.array(list(set(all_kmers))).astype(str)


def get_bear_probs_seqs(bear_path, seqs, train_col,
                        mc_samples=41, vans=[0.1, 1, 10], get_map=False,
                        lag=None, alphabet=None, h=None, data=None,
                        kmc_path=None, kmc_reverse=False, no_ends=False):
    """Sample posterior predictive probabilities of variants under BEAR by looping through batches of kmers.
    
    Parameters
    ----------
    bear_path : str
        Path to folder of trained BEAR model. None if using BMM.
    seqs : list
        List of sequences to get probabilities of. Must not be bytes!
    train_col : int
        Row of counts data that includes the training data
    mc_samples : int, default = 41
        Number of samples to take from the posterior predictive of the mutation probabilities.
    vans : numpy array, default = [0.1, 1, 10]
        vanilla regularization to use for BMM models.
    get_map : bool
        Gets the probability of the mutations under the MAP model under BEAR instead of sampling models from BEAR.
    lag : int, default = None
        Specify if not using BEAR.
    alphabet :str, default = None
        Specify if not using BEAR.
    h : numpy array
        For h scans if bear_folder is specified. None if using fit BEAR h.
    data : tf dataset
        Generator of kmers and counts. Specify if not using BEAR.
    kmc_path : str
        Specify the path kmc files if one wishes to use kmc to count kmers instead of cycling through whole dataset.
    kmc_reverse : bool, default=False
        Whether to include counts of the reverse complement of kmers when counting using kmc.
    no_ends: bool, default=False
        Whether or not to include starts and stops in probability calculations.
        
    Returns
    -------
    scores : numpy array of floats
        [num sequences, num models (whather or not to use BEAR + len(vans)), mc_samples].
    """            
    # load bear from bear_path
    if bear_path != None:
        lag, alphabet, h_bear, ar_func, data = load_bear(bear_path)
        if h is None:
            h = np.array([h_bear])
        len_h = len(h)
    else:
        assert ((lag is not None and alphabet is not None)
                and ((data is not None or kmc_path is not None) and len(vans) > 0))
        if kmc_path is not None:
            assert alphabet == 'dna' and train_col == 0and train_col == 0
            #TODO inflate kmc counts to include case where train col of ar_func is not 0
        len_h = 0
        ar_func = None
    alphabet_size = len(core.alphabets_en[alphabet])-1

    # pad seqs
    if not no_ends:
        seqs = [lag*'['+seq+']' for seq in seqs]

    # get list of all possible kmers
    all_kmers = _get_all_kmers_seqs(seqs, lag)
    
    # no sampling if just using the MAP
    if get_map:
        mc_samples = 1
    num_models = (ar_func is not None)*(len_h + get_map) + len(vans)

    scores = np.zeros([len(seqs), num_models, mc_samples])  
    
    if kmc_path is not None:
        counter = make_kmc_genome_counter(kmc_path, lag, reverse=kmc_reverse)
        all_counts = counter(all_kmers)[:, None, :]
        if np.all(all_counts == 0):
            print("no kmers found, are you sure you have the correct kmc file and lag?")
        pdf = get_pdf(all_kmers, all_counts, h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
        scores = _add_kmer_probs_seqs(seqs, scores, pdf, lag, all_kmers)
    else:
        seen_all_kmers = np.zeros(len(all_kmers))
        for kmers, counts in iter(data):
            kmers = kmers.numpy().astype(str)
            # first throw out kmers in the batch that can't contribute to the variant scores
            in_kmers = np.isin(kmers, all_kmers)
            print("num seen kmers in this batch:", np.sum(in_kmers))
            seen_kmers = kmers[in_kmers]
            seen_counts = counts[in_kmers]
            # make a record of having seem the kmer
            seen_all_kmers += np.isin(all_kmers, seen_kmers)
            if np.sum(in_kmers)>0:
                # get probabilities of all transitions out of each kmer
                pdf = get_pdf(seen_kmers, seen_counts, h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
                # goes through all mutants and add the probabilities contributed by this batch of kmers
                scores = _add_kmer_probs_seqs(seqs, scores, pdf, lag, seen_kmers)
        # some kmers haven't been seen but still affect the probabilities through prior values
        unseen_kmers = (seen_all_kmers == 0)
        print("num unseen kmers:", sum(unseen_kmers))
        if sum(unseen_kmers)>0:
            pdf = get_pdf(all_kmers[unseen_kmers],
                           tf.zeros([sum(unseen_kmers), train_col+1, alphabet_size+1], dtype=tf.float64),
                           h, ar_func, mc_samples, vans, train_col, alphabet, get_map)
            scores = _add_kmer_probs_seqs(seqs, scores, pdf, lag, all_kmers[unseen_kmers].astype(str))
    if get_map:
        scores = scores[..., 0]
    return scores
