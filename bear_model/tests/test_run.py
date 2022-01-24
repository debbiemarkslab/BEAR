import os
import configparser
import numpy as np
import tensorflow.compat.v2 as tf
from pkg_resources import resource_filename
from bear_model.models import train_bear_net
from bear_model.models import train_bear_ref
from bear_model import dataloader

epsilon = tf.keras.backend.epsilon()

def test_run_net():
#     Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
#     script_name = resource_filename('bear_model', 'models/train_bear_net.py')
    config = configparser.ConfigParser()
    config.read(f_name)
    exit, ll_van, perp_van = train_bear_net.main(config)
    assert 1 == exit
    
#      Might as well also check that likelihood calculations are going well
    f_name = resource_filename('bear_model', 'data/ysd1_lag_5_file_0_preshuf.tsv')
    data = dataloader.dataloader(f_name, 'dna', 2000, 3)
    kmers, counts = next(iter(data))
    counts = counts.numpy()
    alpha = np.array([0.1, 1., 10.]) + epsilon
    calc_liks = dataloader.bmm_likelihood(data.map(lambda kmers, counts: counts), alpha)
    train_liks = calc_liks[0].numpy()
    assert np.allclose(train_liks, ll_van)
    assert np.allclose(np.exp(-train_liks/np.sum(counts[:, 0, :])), perp_van)

def test_run_ref():
#     Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
#     script_name = resource_filename('bear_model', 'models/train_bear_ref.py')
    config = configparser.ConfigParser()
    config.read(f_name)
    exit, ll_van, perp_van = train_bear_ref.main(config)
    print(ll_van)
    assert 1 == exit

#      Might as well also check that likelihood calculations are going well
    f_name = resource_filename('bear_model', 'data/ysd1_lag_5_file_0_preshuf.tsv')
    data = dataloader.dataloader(f_name, 'dna', 2000, 3)
    kmers, counts = next(iter(data))
    counts = counts.numpy()
    alpha = np.array([0.1, 1., 10.]) + epsilon
    calc_liks = dataloader.bmm_likelihood(data.map(lambda kmers, counts: counts), alpha)
    train_liks = calc_liks[0].numpy()
    assert np.allclose(train_liks, ll_van)
    assert np.allclose(np.exp(-train_liks/np.sum(counts[:, 0, :])), perp_van)