from scipy.special import gamma, gammaln
import scipy.stats as st
import numpy as np

def triple_slice_mut(b, c, d):
    # a, b are same size, c is size of sum(b), d is size of sum(c)
    # gives back same slice as a[b][c][d]
    change = np.zeros(len(b), dtype=bool)
    change_temp = np.zeros(len(c), dtype=bool)
    change_temp[c] = d
    change[b] = change_temp
    return change
    
def log_gamma_pdf(conc, xs):
    return np.exp(conc * xs - np.exp(xs) - gammaln(conc))
    
def log_gamma(concs, size=[]):
    """ Samples from a loggamma (good for small values).
    Does so by rejection sampling an exponential for negative values and
    just drawing a regular numpy gamma for positive.
    
    If f is the log_gamma pdf, F the cdf, c the concentration, out proposal is
    g(x) = F(0) c exp(cx) if x is negative and f(x) if x is positive.
    The log gamma pdf is exp(cx) exp(-exp(x)) / gamma(c).
    The renormalization we will add to g is M = max{1, 1/(gamma(c) F(0) c)}
    so Mg >= f everywhere.
    Note that when M>1 (c<~5), f/(Mg) = exp(-exp(x))>1/3 when x<0 and 
    =1/M when x>0. Thus, this sampling scheme is
    efficient for small values of c. for c > 1 then, we'll just sample normally.
    M, increasing with c, reaches a peak of ~1.5 at c=1.
    Can get a million samples in around a second."""
    
    shape = np.r_[size, np.shape(concs)].astype(int)
    concs = np.tile(concs, np.r_[size, np.ones(len(np.shape(concs)))].astype(int))
    concs = concs.flatten()
    
    n = len(concs)
    draws = np.empty(n)
    remain = np.ones(n, dtype=bool)
    n_left = np.sum(remain)
    
    # first, if conc > 1, we can just sample normally
    draws[concs >= 1] = np.log(np.random.standard_gamma(concs[concs >= 1]))
    remain = concs < 1
    n_left = np.sum(remain)
    
    log_prob_neg = np.log(st.gamma.cdf(1, concs)) # cdf is only unstable when concs are large
    log_ms = np.maximum(0, -(log_prob_neg + gammaln(concs) + np.log(concs)))
    while n_left > 0:
        # first sample from regular gamma conditional on > 0. IS ratio is 1/M.
        x_gam = np.random.standard_gamma(concs[remain])
        pos = x_gam > 1
        x_pos = np.log(x_gam[pos])
        n_pos = np.sum(pos)
        n_neg = np.sum(remain) - np.sum(pos)
        u = np.random.uniform(size=n_pos)
        is_accept_pos = u < np.exp(-log_ms[remain][pos])
        
        # conditional on negative, sample from negative gamma and RS
        u = np.random.uniform(size=n_neg)
        x_neg = - np.random.standard_gamma(np.ones(n_neg)) / concs[remain][~pos]        
        ratio = np.exp(- np.exp(x_neg) - gammaln(concs[remain][~pos])
                       - np.log(concs[remain][~pos]) - log_prob_neg[remain][~pos] - log_ms[remain][~pos])
        # print(ratio)
        is_accept_neg = u < ratio

        slice_pos = triple_slice_mut(remain, pos, is_accept_pos)
        slice_neg = triple_slice_mut(remain, ~pos, is_accept_neg)

        draws[slice_pos] = x_pos[is_accept_pos]
        remain[slice_pos] = 0
        draws[slice_neg] = x_neg[is_accept_neg]
        remain[slice_neg] = 0
        
        n_left = np.sum(remain)
    return draws.reshape(shape)
