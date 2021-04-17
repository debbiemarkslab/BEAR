from bear_model import core
import numpy as np
from scipy.special import loggamma
from scipy import stats as st


def test_dirichlet_multinomial_distribution():
    # Test that the sample size, log probability, and ml are working as
    # intended for the dirichlet multinomial dist.
    shape = np.array([3, 5])
    alphabet_size = 4
    trans_counts = np.random.poisson(size=np.r_[shape, alphabet_size+1]).astype(float)
    total_counts = np.sum(trans_counts, axis=-1)
    concentration = np.random.exponential(size=np.r_[shape[1], alphabet_size+1]).astype(float)
    sum_conc = np.sum(concentration, axis=-1)
    dist = core.tfpDirichletMultinomialPerm(total_counts.astype(float), concentration)

    shape_sam = 7
    assert np.all(dist._sample_n(shape_sam).numpy() == np.zeros(np.r_[shape_sam, shape, alphabet_size+1]))

    assert np.all(dist.ml_output().numpy() == np.tile(np.argmax(concentration, axis=-1)[None, ...], [shape[0], 1]))

    assert np.allclose(dist.counts_log_prob(trans_counts).numpy(), (np.sum(loggamma(concentration+trans_counts)
                                                                           - loggamma(concentration), axis=-1)
                                                                    - (loggamma(sum_conc+total_counts)
                                                                       - loggamma(sum_conc))))


def test_dirichlet_multinomial_distribution_tie_breaking():
    # Test tie breaking in Dirichlet Multinomial distribution (1/1000 chance this test will fail randomly)
    total_count = np.array([1]).astype(float)
    concentration = np.array([1, 0.5, 1]).astype(float)
    dist = core.tfpDirichletMultinomialPerm(total_count, concentration)
    n_trials = 1000
    trials = np.empty(n_trials)
    for i in range(n_trials):
        assert dist.ml_output().numpy() in [0, 2]
        trials[i] = dist.ml_output().numpy()
    assert np.abs(np.sum(trials - 1)/np.sqrt(n_trials)) < st.norm.ppf(0.9995)


def test_multinomial_distribution():
    # Test that the sample size, log probability, and ml are working
    # as intended for the dirichlet multinomial dist.
    shape = np.array([3, 5])
    alphabet_size = 4
    trans_counts = np.random.poisson(size=np.r_[shape, alphabet_size+1]).astype(float)
    total_counts = np.sum(trans_counts, axis=-1)
    concentration = np.random.exponential(size=np.r_[shape[1], alphabet_size+1]).astype(float)
    sum_conc = np.sum(concentration, axis=-1)
    concentration /= sum_conc
    dist = core.tfpMultinomialPerm(total_counts.astype(float), concentration)

    shape_sam = 7
    assert np.all(dist._sample_n(shape_sam).numpy() == np.zeros(np.r_[shape_sam, shape, alphabet_size+1]))

    assert np.all(dist.ml_output().numpy() == np.tile(np.argmax(concentration, axis=-1)[None, ...], [shape[0], 1]))

    assert np.allclose(dist.counts_log_prob(trans_counts).numpy(),
                       np.sum(np.log(concentration) * trans_counts, axis=-1))


def test_multinomial_distribution_tie_breaking():
    # Test tie breaking in Dirichlet Multinomial distribution (1/1000 chance this test will fail randomly)
    total_count = np.array([1]).astype(float)
    concentration = np.array([1, 0.5, 1]).astype(float)/(2.5)
    dist = core.tfpMultinomialPerm(total_count, concentration)
    n_trials = 1000
    trials = np.empty(n_trials)
    for i in range(n_trials):
        assert dist.ml_output().numpy() in [0, 2]
        trials[i] = dist.ml_output().numpy()
    assert np.abs(np.sum(trials - 1)/np.sqrt(n_trials)) < st.norm.ppf(0.9995)
