from bear_model import dataloader
from bear_model import get_var_probs
import numpy as np
from scipy import stats as st
from scipy.special import logsumexp
from pkg_resources import resource_filename

def test_kmc_counter():
    f_name = resource_filename('bear_model', 'data/ex_seqs_kmap_for_var_pred.csv')
    kmc_path = f_name.split('ex_seqs_kmap_for_var_pred.csv')[0] + 'kmaps/ex_seqs'
    lag = 3
    # Sequences are 'TTTAT', 'TTCTT', 'TTTTT', 'TTTTT'
    counter = get_var_probs.make_kmc_genome_counter(kmc_path, lag, reverse=False, no_end=False)
    assert np.all(counter(np.array(['TTT', 'TTA', '[[T'])) == [[1, 0, 0, 4, 2], [0, 0, 0, 1, 0], [0, 0, 0, 4, 0]])
    
    counter = get_var_probs.make_kmc_genome_counter(kmc_path, lag, reverse=True, no_end=False)
    print(counter(np.array(['TTT', '[AT', 'AAA'])))
    assert np.all(counter(np.array(['TTT', '[AT', 'AAA'])) == [[1, 0, 0, 4, 2], [1, 0, 0, 0, 0], [4, 0, 0, 0, 3]])

def test_mc_sampling():
    f_name = resource_filename('bear_model', 'data/ex_seqs_kmap_for_var_pred.csv')
    kmc_path = f_name.split('ex_seqs_kmap_for_var_pred.csv')[0] + 'kmaps/ex_seqs'
    data = dataloader.sparse_dataloader(f_name, 'dna', 500, 1)
    alphabet_size = 4
    
    # Sequences are 'TTTAT', 'TTCTT', 'TTTTT', 'TTTTT'
    wt_seq = 'TTTAT'
    vars_ = np.array(['A3T', 'T2C'])
    vans = np.array([0.1, 1, 10])
    
    # First by mc sampling
    scores = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data,
                                          mc_samples=500000, vans=vans, lag=3, alphabet_name='dna')
    #also using kmc
    scores_kmc = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data,
                                              mc_samples=500000, vans=vans, lag=3, alphabet_name='dna', kmc_path=kmc_path)

    #calc true probs
    def get_log_dir(seen, all_, van, num_samples=500000):
        return np.average(np.log(st.beta.rvs(seen+van, all_-seen+alphabet_size*van, size=num_samples)))

    true_scores = np.empty([len(vars_), len(vans)])
    for i, van in enumerate(vans):
        true_scores[0, i] = ((2*get_log_dir(4, 7, van) + 1*get_log_dir(2, 7, van))
                             -(1*get_log_dir(1, 7, van) + 2*get_log_dir(1, 1, van)))

    for i, van in enumerate(vans):
        true_scores[1, i] = ((get_log_dir(1, 4, van) + get_log_dir(0, 1, van) + 2*get_log_dir(0, 0, van))
                             -(get_log_dir(3, 4, van) + get_log_dir(1, 7, van) + 2*get_log_dir(1, 1, van)))
        
    #assert
    fraction_error = (np.average(scores, axis=-1)-true_scores)/true_scores
    print(np.absolute(fraction_error))
    assert np.all(np.absolute(fraction_error) < 0.02)
    
    fraction_error = (np.average(scores_kmc, axis=-1)-true_scores)/true_scores
    print(np.absolute(fraction_error))
    assert np.all(np.absolute(fraction_error) < 0.02)
    
    # Now by MAP
    scores = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data, get_map=True,
                                          vans=vans, lag=3, alphabet_name='dna')
    scores_kmc = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data, get_map=True,
                                              vans=vans, lag=3, alphabet_name='dna', kmc_path=kmc_path)
    
    def get_quotient(seen, all_, van):
        return np.log((seen+van) / (all_+(alphabet_size+1)*van))
    
    for i, van in enumerate(vans):
        true_scores[0, i] = ((2*get_quotient(4, 7, van) + 1*get_quotient(2, 7, van))
                             -(1*get_quotient(1, 7, van) + 2*get_quotient(1, 1, van)))

    for i, van in enumerate(vans):
        true_scores[1, i] = ((get_quotient(1, 4, van) + get_quotient(0, 1, van) + 2*get_quotient(0, 0, van))
                             -(get_quotient(3, 4, van) + get_quotient(1, 7, van) + 2*get_quotient(1, 1, van)))
        
    print((np.average(scores, axis=-1)-np.average(scores_kmc, axis=-1))/np.average(scores, axis=-1))
    assert np.allclose(scores, true_scores)


def test_mc_sampling_seqs():
    f_name = resource_filename('bear_model', 'data/ex_seqs_kmap_for_var_pred.csv')
    kmc_path = f_name.split('ex_seqs_kmap_for_var_pred.csv')[0] + 'kmaps/ex_seqs'
    data = dataloader.sparse_dataloader(f_name, 'dna', 500, 1)
    alphabet_size = 4
    
    # Sequences are 'TTTAT', 'TTCTT', 'TTTTT', 'TTTTT'
    seqs = ['TTTAT', 'TTCAT', 'TTTTTTTTTT']
    vans = np.array([0.1, 1, 10])
    
    # First by mc sampling
    scores = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data,
                                          mc_samples=20000, vans=vans, lag=3, alphabet_name='dna')
    #also using kmc
    scores_kmc = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data,
                                              mc_samples=20000, vans=vans, lag=3, alphabet_name='dna', kmc_path=kmc_path)
    
    # Also get marg
    margs = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data, get_marg = True,
                                              vans=vans, lag=3, alphabet_name='dna')
    margs_kmc = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data, get_marg = True,
                                                  vans=vans, lag=3, alphabet_name='dna', kmc_path=kmc_path)

    #calc true probs
    N = 50000
    def get_log_dir(seen, all_, van, num_samples=N):
        return np.log(st.beta.rvs(seen+van, all_-seen+alphabet_size*van, size=num_samples))

    true_scores = np.empty([len(seqs), len(vans), N])
    for i, van in enumerate(vans):
        true_scores[0, i] = (get_log_dir(4, 4, van) + get_log_dir(4, 4, van) #starts
                             + get_log_dir(3, 4, van)
                             + get_log_dir(1, 7, van) # TTT -> A
                             + get_log_dir(1, 1, van)
                             + get_log_dir(1, 1, van))
    for i, van in enumerate(vans):
        true_scores[1, i] = (get_log_dir(4, 4, van) + get_log_dir(4, 4, van)
                             + get_log_dir(1, 4, van)
                             + get_log_dir(0, 1, van)
                             + get_log_dir(0, 0, van)
                             + get_log_dir(0, 0, van))
    for i, van in enumerate(vans):
        # have to include correlation between TTT->T and ]
        ttt_dirs = np.log(st.beta.rvs(4+van, 2+van, size=N))
        ttt_mod = np.log(st.beta.rvs(6+2*van, 1+3*van, size=N))
        true_scores[2, i] = (get_log_dir(4, 4, van) + get_log_dir(4, 4, van)
                             + get_log_dir(3, 4, van)
                             + 7 * (ttt_dirs + ttt_mod)
                             + (np.log1p(-np.exp(ttt_dirs)) + ttt_mod))
        
    #assert
    av_scores = np.average(true_scores, axis=-1)
    fraction_error = (np.average(scores, axis=-1) - av_scores) / av_scores
    assert np.all(np.absolute(fraction_error) < 0.01)
    
    fraction_error = (np.average(scores_kmc, axis=-1) - av_scores) / av_scores
    assert np.all(np.absolute(fraction_error) < 0.01)
    
    av_scores = logsumexp(true_scores, axis=-1) - np.log(N)
    fraction_error = (margs - av_scores) / av_scores
    assert np.all(np.absolute(fraction_error) < 0.01)
    
    fraction_error = (margs_kmc - av_scores) / av_scores
    assert np.all(np.absolute(fraction_error) < 0.01)
    
    # Now by MAP
    scores = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data, get_map=True,
                                          vans=vans, lag=3, alphabet_name='dna')
    scores_kmc = get_var_probs.get_bear_probs_seqs(None, seqs, 0, data=data, get_map=True,
                                                   vans=vans, lag=3, alphabet_name='dna', kmc_path=kmc_path)
    
    def get_quotient(seen, all_, van):
        return np.log((seen+van) / (all_+(alphabet_size+1)*van))
    
    true_scores = np.empty([len(seqs), len(vans)])
    for i, van in enumerate(vans):
        true_scores[0, i] = (2*get_quotient(4, 4, van) + 1*get_quotient(3, 4, van)
                             + 1*get_quotient(1, 7, van) +  2*get_quotient(1, 1, van))

    for i, van in enumerate(vans):
        true_scores[1, i] = (2*get_quotient(4, 4, van) + 1*get_quotient(1, 4, van)
                             + 1*get_quotient(0, 1, van) +  2*get_quotient(0, 0, van))
        
    for i, van in enumerate(vans):
        true_scores[2, i] = (2 * get_quotient(4, 4, van)
                             + get_quotient(3, 4, van)
                             + 7 * get_quotient(4, 7, van)
                             + get_quotient(2, 7, van))
        
    print((np.average(scores, axis=-1)-np.average(scores_kmc, axis=-1))/np.average(scores, axis=-1))
    print(scores)
    print(true_scores)
    assert np.allclose(scores, true_scores)
    