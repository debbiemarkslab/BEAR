"""
Train and evaluate AR or BEAR models that require a reference using maximal liklihood or empirical Bayes respectively.
Train reference-based Bayesian embedded autoregressive models using
empirical Bayes, and evaluate based on heldout
likelihood, perplexity, and accuracy. Usage:

``python train_bear_ref.py config.cfg``

Example config files, with descriptions of the input parameters, can be found
in the subfolder ``bear_model/models/config_files`` (`bear_stop_bear.cfg` and `bear_stop_ar.cfg`).
"""
import argparse
import configparser
import tensorflow.compat.v2 as tf
import dill
import os
import json
import subprocess
import datetime
import numpy as np
from matplotlib import pyplot as plt
from bear_model import dataloader
from bear_model import ar_funcs
from bear_model import bear_ref
from pkg_resources import resource_filename


def main(config):
    # Setup.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config['general']['out_folder'] == 'TEST':
        out_folder = os.path.join(resource_filename('bear_model', 'models/'),
                                  'out_data/logs', time_stamp)
    elif config['general']['out_folder'][-1] == '*':
        out_folder = config['general']['out_folder'][:-1]
        os.system('mkdir -p '+out_folder)
    else:
        out_folder = os.path.join(config['general']['out_folder'],
                                  'logs', time_stamp)
    tf.random.set_seed(int(config['general']['seed']))
    dtype = getattr(tf, config['general']['precision'])

    # Set up the recording mechanism for TensorBoard.
    writer = tf.summary.create_file_writer(out_folder)

    # Load data.
    if config['data']['files_path'] == 'TEST':
        files = [resource_filename('bear_model', 'data/ysd1_lag_5_file_0_preshuf.tsv')]
    else:
        files = [os.path.join(config['data']['files_path'], file) for file in os.listdir(config['data']['files_path'])
                 if file.startswith(config['data']['start_token'])]
    num_kmers = sum([(int)(subprocess.run('wc -l '+file, shell=True, capture_output=True
                                         ).stdout.decode("utf-8").split()[0]) for file in files])
    kmer_batch_size = float(config['train']['batch_size'])
    if kmer_batch_size <= 1:
        kmer_batch_size = (int)(num_kmers * kmer_batch_size)
    else:
        kmer_batch_size = (int)(kmer_batch_size)
    epochs = config['train']['epochs']
    if epochs[-1] == 's':
        epochs = int(epochs[:-1])//(1+num_kmers//kmer_batch_size)+1
    else:
        epochs = int(epochs)
    num_ds = int(config['data']['num_ds'])

    if config['data']['sparse'] == 'True':
        dataload_func = dataloader.sparse_dataloader
    else:
        dataload_func = dataloader.dataloader

    if len(files) == 1:
        data = dataload_func(files[0], config['data']['alphabet'],
                             kmer_batch_size, num_ds,
                             cache=config['train']['cache'] == 'True',
                             dtype=dtype)
    else:
        # Interleave loaded data from multiple files.
        data = tf.data.Dataset.from_tensor_slices(files)
        data = data.interleave(
            lambda x: dataload_func(x, config['data']['alphabet'],
                                    kmer_batch_size, num_ds,
                                    cache=config['train']['cache'] == 'True',
                                    dtype=dtype),
            cycle_length=4, num_parallel_calls=4, deterministic=False)
    data_train = data.repeat(epochs)
    print("data_loaded")

    # Add location of results to config.
    result_file = os.path.join(out_folder, 'results.pickle')
    config['results']['out_folder'] = out_folder
    config['results']['file'] = result_file
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)

    # Load hyperparameters.
    ds_loc = int(config['data']['train_column'])
    ds_loc_ref = int(config['data']['reference_column'])
    alphabet = config['data']['alphabet']
    lag = int(config['hyperp']['lag'])

    make_ar_func = getattr(ar_funcs, 'make_ar_func_'+config['model']['ar_func_name'])
    af_kwargs = json.loads(config['model']['af_kwargs'])

    # Settings for training
    learning_rate = float(config['train']['learning_rate'])
    optimizer_name = config['train']['optimizer_name']
    train_ar = config['train']['train_ar'] == 'True'
    acc_steps = int(config['train']['accumulation_steps'])

    # If restarting, reload params.
    if config['train']['restart'] == 'True':
        with open(config['train']['restart_path']+"results.pickle", 'rb') as fr:
            results = dill.load(fr)
        params_restart = results['params']
    else:
        params_restart = None

    # Train.
    loss_save = []
    (params, h_signed, ar_func) = bear_ref.train(
        data_train, num_kmers, epochs, ds_loc, ds_loc_ref, alphabet, lag, make_ar_func, af_kwargs,
        learning_rate, optimizer_name, train_ar=train_ar, acc_steps=acc_steps,
        params_restart=params_restart, writer=writer, loss_save=loss_save, dtype=dtype)
    
    # plot training curve
    plt.figure(figsize=[10, 10])
    plt.xlabel("steps", fontsize=30)
    plt.ylabel("loss", fontsize=30)
    plt.plot(loss_save)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'loss.png'), dpi=200)

    # Add learned h to results in config.
    config['results']['h'] = str(tf.math.exp(h_signed).numpy())
    tau = tf.math.exp(params[1])
    config['results']['error_rate'] = str(1-tf.math.exp(-tau).numpy())
    nw = tf.math.exp(params[2])
    config['results']['stop_rate'] = str(1/((nw/(1+nw)).numpy()))
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)

    # Save results.
    result_file = os.path.join(out_folder, 'results.pickle')
    with open(result_file, 'wb') as rw:
        dill.dump({'params': params}, rw)

    if config['test']['test'] == 'True':
        # Load settings for testing.
        ds_loc_test = int(config['data']['test_column'])
        van_reg = np.array(json.loads(config['test']['van_reg']))

        (ll_ear, ll_ar, ll_van, perp_ear, perp_ar, perp_van,
         acc_ear, acc_ar, acc_van) = bear_ref.evaluation(
            data, ds_loc, ds_loc_test, ds_loc_ref, alphabet,
            tf.math.exp(h_signed), ar_func, van_reg)

        # Save config.
        config['results']['heldout_perplex_BEAR'] = str(perp_ear.numpy())
        config['results']['heldout_perplex_AR'] = str(perp_ar.numpy())
        config['results']['heldout_perplex_BMM'] = json.dumps(perp_van.numpy().tolist())
        config['results']['heldout_loglikelihood_BEAR'] = str(ll_ear.numpy())
        config['results']['heldout_loglikelihood_AR'] = str(ll_ar.numpy())
        config['results']['heldout_loglikelihood_BMM'] = json.dumps(ll_van.numpy().tolist())
        config['results']['heldout_accuracy_BEAR'] = str(acc_ear.numpy())
        config['results']['heldout_accuracy_AR'] = str(acc_ar.numpy())
        config['results']['heldout_accuracy_BMM'] = json.dumps(acc_van.numpy().tolist())
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)
            
    if config['test']['train_test']=='True':
        ds_loc_train = -1
        ds_loc_test = ds_loc
        van_reg = np.array(json.loads(config['test']['van_reg']))

        (ll_ear, ll_ar, ll_van, perp_ear, perp_ar, perp_van,
         acc_ear, acc_ar, acc_van) = bear_ref.evaluation(
            data, ds_loc_train, ds_loc_test, ds_loc_ref, alphabet,
            tf.math.exp(h_signed), ar_func, van_reg)

        # Save config.
        config['results']['perplex_BEAR'] = str(perp_ear.numpy())
        config['results']['perplex_AR'] = str(perp_ar.numpy())
        config['results']['perplex_BMM'] = json.dumps(perp_van.numpy().tolist())
        config['results']['loglikelihood_BEAR'] = str(ll_ear.numpy())
        config['results']['loglikelihood_AR'] = str(ll_ar.numpy())
        config['results']['loglikelihood_BMM'] = json.dumps(ll_van.numpy().tolist())
        config['results']['accuracy_BEAR'] = str(acc_ear.numpy())
        config['results']['accuracy_AR'] = str(acc_ar.numpy())
        config['results']['accuracy_BMM'] = json.dumps(acc_van.numpy().tolist())
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)
            
        # Return liks for testing
        return 1, ll_van.numpy(), perp_van.numpy()
 
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('configPath')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
