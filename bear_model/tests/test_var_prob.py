from bear_model import dataloader
from bear_model import get_var_probs
import numpy as np
from scipy import stats as st
from pkg_resources import resource_filename

def test_mc_sampling():
    f_name = resource_filename('bear_model', 'data/ex_seqs_kmap_for_var_pred.csv')
    data = dataloader.sparse_dataloader(f_name, 'prot', 500, 1)
    alphabet_size = 20
    
    # Sequences are 'MMMAM', 'MMKMM', 'MMMMM', 'MMMMM'
    wt_seq = 'MMMAM'
    vars_ = np.array(['A3M', 'M2K'])
    vans = np.array([0.1, 1, 10])
    
    # First by mc sampling
    scores = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data,
                                          mc_samples=200000, vans=vans, lag=3, alphabet='prot')

    def get_log_dir(seen, all_, van, num_samples=200000):
        return np.average(np.log(st.beta.rvs(seen+van, all_-seen+alphabet_size*van, size=num_samples)))

    true_scores = np.empty([len(vars_), len(vans)])
    for i, van in enumerate(vans):
        true_scores[0, i] = ((2*get_log_dir(4, 7, van) + 1*get_log_dir(2, 7, van))
                             -(1*get_log_dir(1, 7, van) + 2*get_log_dir(1, 1, van)))

    for i, van in enumerate(vans):
        true_scores[1, i] = ((get_log_dir(1, 4, van) + get_log_dir(0, 1, van) + 2*get_log_dir(0, 0, van))
                             -(get_log_dir(3, 4, van) + get_log_dir(1, 7, van) + 2*get_log_dir(1, 1, van)))
        
    fraction_error = (np.average(scores, axis=-1)-true_scores)/true_scores
    assert np.all(np.absolute(fraction_error) < 0.01)
    
    # Now by MAP
    scores = get_var_probs.get_bear_probs(None, wt_seq, vars_, 0, data=data, get_map=True,
                                          vans=vans, lag=3, alphabet='prot')
    
    def get_quotient(seen, all_, van):
        return np.log((seen+van) / (all_+(alphabet_size+1)*van))
    
    for i, van in enumerate(vans):
        true_scores[0, i] = ((2*get_quotient(4, 7, van) + 1*get_quotient(2, 7, van))
                             -(1*get_quotient(1, 7, van) + 2*get_quotient(1, 1, van)))

    for i, van in enumerate(vans):
        true_scores[1, i] = ((get_quotient(1, 4, van) + get_quotient(0, 1, van) + 2*get_quotient(0, 0, van))
                             -(get_quotient(3, 4, van) + get_quotient(1, 7, van) + 2*get_quotient(1, 1, van)))
        
    assert np.allclose(scores, true_scores)