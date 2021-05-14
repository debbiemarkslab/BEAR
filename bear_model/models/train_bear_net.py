"""
Train and evaluate AR or BEAR models using maximal liklihood or empirical Bayes respectively. Usage:

``python train_bear_net.py config.cfg``

Example config files, with descriptions of the input parameters, can be found
in the subfolder ``bear_model/models/config_files`` (`bear_lin_bear.cfg`, `bear_lin_ar.cfg`,
`bear_cnn_bear.cfg`, and `bear_cnn_ar.cfg`).
"""
import argparse
import configparser
import tensorflow.compat.v2 as tf
import dill
import os
import json
import subprocess
import datetime
from bear_model import dataloader
from bear_model import ar_funcs
from bear_model import bear_net


def main(config):
    # Setup.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder = os.path.join(config['general']['out_folder'],
                              'logs', time_stamp)
    tf.random.set_seed(int(config['general']['seed']))
    dtype = getattr(tf, config['general']['precision'])

    # Set up the recording mechanism for TensorBoard.
    writer = tf.summary.create_file_writer(out_folder)

    # Load data.
    files = [config['data']['files_path']+file for file in os.listdir(config['data']['files_path'])
             if file.startswith(config['data']['start_token'])]
    num_kmers = sum([(int)(subprocess.check_output(['wc', '-l', file]).split()[0]) for file in files])
    epochs = int(config['train']['epochs'])
    num_ds = int(config['data']['num_ds'])

    kmer_batch_size = float(config['train']['batch_size'])
    if kmer_batch_size <= 1:
        kmer_batch_size = (int)(num_kmers * kmer_batch_size)
    else:
        kmer_batch_size = (int)(kmer_batch_size)

    if len(files) == 1:
        data = dataloader.dataloader(files[0], config['data']['alphabet'],
                                     kmer_batch_size, num_ds,
                                     cache=(bool)((int)(config['train']['cache'])),
                                     dtype=dtype)
    else:
        # Interleave loaded data from multiple files.
        data = tf.data.Dataset.from_tensor_slices(files)
        data = data.interleave(
            lambda x: dataloader.dataloader(x, config['data']['alphabet'],
                                            kmer_batch_size, num_ds,
                                            cache=(bool)((int)(config['train']['cache'])),
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
    ds_loc = json.loads(config['data']['train_column'])
    alphabet = config['data']['alphabet']
    lag = int(config['hyperp']['lag'])

    make_ar_func = getattr(ar_funcs, 'make_ar_func_'+config['model']['ar_func_name'])
    af_kwargs = {}
    for key in config['model']:
        if key != 'ar_func_name':
            af_kwargs[key] = (float)(config['model'][key])

    # Settings for training
    learning_rate = float(config['train']['learning_rate'])
    optimizer_name = config['train']['optimizer_name']
    train_ar = (bool)((int)(config['train']['train_ar']))
    acc_steps = int(config['train']['accumulation_steps'])

    # If restarting, reload params.
    if (bool)((int)(config['train']['restart'])):
        with open(config['train']['restart_path']+"results.pickle", 'rb') as fr:
            results = dill.load(fr)
        params_restart = results['params']
    else:
        params_restart = None

    # Train.
    (params, h_signed, ar_func) = bear_net.train(
        data_train, num_kmers, epochs, ds_loc, alphabet, lag, make_ar_func, af_kwargs,
        learning_rate, optimizer_name, train_ar=train_ar, acc_steps=acc_steps,
        params_restart=params_restart, writer=writer, dtype=dtype)

    # Add learned h to results in config.
    config['results']['h'] = str(tf.math.exp(h_signed))
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)

    # Save results.
    result_file = os.path.join(out_folder, 'results.pickle')
    with open(result_file, 'wb') as rw:
        dill.dump({'params': params}, rw)

    if bool((int)(config['test']['test'])):
        # Load settings for testing.
        ds_loc_test = json.loads(config['data']['test_column'])
        van_reg = float(config['test']['van_reg'])

        (ll_ear, ll_ar, ll_van, perp_ear, perp_ar, perp_van,
         acc_ear, acc_ar, acc_van) = bear_net.evaluation(
            data, ds_loc, ds_loc_test, alphabet,
            tf.math.exp(h_signed), ar_func, van_reg)

        # Save config.
        config['results']['heldout_perplex_BEAR'] = str(perp_ear.numpy())
        config['results']['heldout_perplex_AR'] = str(perp_ar.numpy())
        config['results']['heldout_perplex_BMM'] = str(perp_van.numpy())
        config['results']['heldout_loglikelihood_BEAR'] = str(ll_ear.numpy())
        config['results']['heldout_loglikelihood_AR'] = str(ll_ar.numpy())
        config['results']['heldout_loglikelihood_BMM'] = str(ll_van.numpy())
        config['results']['heldout_accuracy_BEAR'] = str(acc_ear.numpy())
        config['results']['heldout_accuracy_AR'] = str(acc_ar.numpy())
        config['results']['heldout_accuracy_BMM'] = str(acc_van.numpy())
        with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
            config.write(cw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('configPath')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
