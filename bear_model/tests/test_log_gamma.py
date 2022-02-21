import numpy as np
from scipy import stats as st
from bear_model import log_gamma

def ks_test(samples, conc):
    """ Tests at significance 10/6%"""
    assert st.kstest(np.exp(samples), cdf='gamma', args=[conc]).pvalue > 0.1/6
    
def test_loggamma():
    concs = np.array([0.01, 0.1, 0.5, 0.99, 1, 5, 100])
    n = 100000
    n_tile = 3

    concs_tile = (np.ones([len(concs), n]) * concs[:, None]).flatten()
    
    samples = log_gamma.log_gamma(concs_tile, size=[n_tile]).reshape([n_tile, len(concs), n])
    real = np.log(np.random.standard_gamma(np.tile(concs_tile, [n_tile, 1]))).reshape([n_tile, len(concs), n])
    for i, conc in enumerate(concs):
        ks_test(samples[:, i].flatten(), conc)