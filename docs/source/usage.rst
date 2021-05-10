#####
Usage
#####

This is a library for building Bayesian embedded autoregressive (BEAR) models,
proposed in Amin et al. (link TODO). It provides (1) a script for extracting
summary statistics from large sequence datasets (which relies on KMC),
(2) a python package with tools for building BEAR models in general
(which relies on TensorFlow), and (3) scripts for training and evaluating
specific example BEAR models. To use the package, follow the installation
instructions and clone the `package repository`_.

.. _package repository: https://github.com/AlanNawzadAmin/BEAR

###########################
Preprocessing sequence data
###########################

.. automodule:: bear_model.summarize


####################
Building BEAR models
####################
********
bear_net
********
The ``bear_net`` submodule contains functions to train, evaluate, and deploy
BEAR (and AR) models.
Two example autoregressive functions are implemented in the submodule ``ar_funcs``:
a linear function and a simple convolutional neural network (CNN),
:func:`bear_model.ar_funcs.make_ar_func_linear` and
:func:`bear_model.ar_funcs.make_ar_func_cnn` respectively.

The standard workflow for training a BEAR model is to first load a
file of preprocessed kmer transition counts using the ``dataloader`` submodule:

.. autofunction:: bear_model.dataloader.dataloader

The loaded datafile may then be used to train an AR of BEAR model using the ``bear_net`` submodule:

.. autofunction:: bear_model.bear_net.train

To evaluate the performance of the trained model, use:

.. autofunction:: bear_model.bear_net.evaluation

To save a model and its learned parameters, use the python module ``dill``.
To recover the misspecification diagnostic h and the autoregressive function
from the list of saved parameters, use the ``change_scope_params`` function:

.. autofunction:: bear_model.bear_net.change_scope_params

Other autoregressive functions may be implemented by creating a
``make_ar_func_...`` function based on the examples
:func:`bear_model.ar_funcs.make_ar_func_linear` and
:func:`bear_model.ar_funcs.make_ar_func_cnn`.

**************
bear_ref
**************

One may also build an autoregressive function from a reference sequence:
A Jukes-Cantor mutation model (with error rate :math:`\tau`) from a reference
sequence is first used to create an AR function, :math:`\tilde f`, on those
transitions that are not to :math:`\$`, the stop symbol, using the transition
counts in the reference :math:`c_{\text{ref}}`:

.. math::
    \tilde{f}(b; k) = e^{-\tau}\frac{c_{\text{ref}, b, k}}{\sum_{b'\neq\$}c_{\text{ref}, b', k}}+\left(1-e^{-\tau}\right)\frac{1}{\#\{b'\neq\$\}}

for :math:`b\neq\$` and :math:`\tilde{f}(\$; k)=0`, where :math:`\$` is the stop symbol.
This is then combined with an AR function :math:`g` that can efficiently calculate transition probabilities from a one hot encoding of a kmer:

.. math::
    f(b;k) = \left(\nu g(b; k) + (1-\nu)\tilde{f}(b;k)\right)/(\nu+1)

for some :math:`\nu\in(0, 1)`. :math:`g` is usually chosen to be a function that always predicts a stop, implemented in :func:`bear_model.ar_funcs.make_ar_func_stop`.
To allow this model to be trained efficiently the reference sequence is preprocessed with the training and testing data to allow :math:`c_{\text{ref}}` to make up a column of the loaded data.
The submodule ``bear_ref`` allows one to train models of this type using

.. autofunction:: bear_model.bear_ref.train

:func:`bear_model.bear_ref.evaluate` and :func:`bear_model.bear_net.change_scope_params` amy be used in analogy with their coresponding functions in ``bear_net``.

###################
Example BEAR models
###################

########
Tutorial
########

In this tutorial, we will apply BEAR models to whole genome
sequencing data from the Salmonella bacteriophage YSD1. The data and the
reference assembly are from
`Dunstan et al. (2019) <https://doi.org/10.1111/mmi.14396>`_; the NCBI SRA
link is
`here <https://www.ncbi.nlm.nih.gov/sra/?term=ERR956946>`_.

**Part 1: preprocessing**

First, download the example dataset and extract its contents (we assume
throughout that you are in the ``bear_model`` folder and the data is in
the ``data`` subfolder).

``wget https://marks.hms.harvard.edu/bear/ysd1_example.tar.gz``

``tar -xvf ysd1_example.tar.gz -C data``

There are five data files in the dataset, corresponding to
training sequence data (`1_train.fasta` and `2_train.fasta`),
testing data (`1_test.fasta` and `2_test.fasta`), and the genome reference
assembly (`virus_reference.fna`).
The file `datalist.csv` lists the dataset paths, data types (all fasta),
and the data groups (training, testing, and reference).
We provide `datalist.csv` as input to the summary statistic script,

``python summarize.py data/datalist.csv data/ysd1 -l 5``

This extracts kmer transition counts up to lag 5 (inclusive).
The counts for each lag will be found in the files `data/ysd1_lag_1_file_0.tsv`,
`data/ysd1_lag_2_file_0.tsv`, `data/ysd1_lag_3_file_0.tsv`,
`data/ysd1_lag_4_file_0.tsv`, and `data/ysd1_lag_5_file_0.tsv`. Each line
consists of an individual kmer and the transition counts in each data group.
Finally, before training a BEAR model, datasets should be shuffled so that
kmers are sampled randomly. In Linux,

``shuf data/ysd1_lag_5_file_0.tsv -o data/ysd1_6_lag_5_file_0_shuf.tsv``

On a Mac, replace ``shuf`` with ``gshuf`` (you may first need to install
GNU coreutils, e.g. ``brew install coreutils``).
A preshuffled version is provided in the file
``models/data/shuffled_virus_kmers_lag_5.tsv``
to ensure this tutorial is reproducible.

**Part 2: training**

The scripts ``bear/models/train_bear_net.py`` and
``bear/models/train_bear_reference.py`` implement the above workflows
for ``bear_net`` and ``bear_ref`` respectively.
These scripts may be used to train on preprocessed transition count data in
one or multiple files.
They each may be used from the command line and accept a .cfg file specifying
the training and testing parameters.

Example config files are located in the config_files folder in the bear/models
folder.
All 6 config files decribe the same training regimen and differ only in
the AR functions they use through the ar_func_name variable under
[model] - linear, cnn, or stop - and whether they decribe training an AR or BEAR
model through the train_ar variable under [train].
The examples may be run by navigating to the bear/models directory and then
using the command line to run one of

``python train_bear_net.py config_files/bear_lin_ar.cfg``

``python train_bear_net.py config_files/bear_lin_bear.cfg``

``python train_bear_net.py config_files/bear_cnn_ar.cfg``

``python train_bear_net.py config_files/bear_cnn_bear.cfg``

``python train_bear_ref.py config_files/bear_stop_ar.cfg``

``python train_bear_ref.py config_files/bear_stop_bear.cfg``

These each should not take more than a few minutes to run on any setup.
These scripts output a folder titled with the time they were run to models/out_data/logs.
Each of these folders contain:

* A pickle file with the trained parameters. They may be recoveded using ``dill``.
* A progress file that may be used to visualize the training curve using tensorboard.
* A config file containing the parameters of the run as well as the performance of the model and :math:`h`.
  The output of ``models/train_bear_reference.py`` also include learned error and stop rates derived from :math:`\tau` and :math:`\nu` as defined above.
  Note the evaluations of the trained AR function and h as parameters of both a BEAR and AR model will be included regardless of the value of train_ar, as well as the performance of a vanilla Bayesian Markov model (BMM).

One may compare the results of their runs to the following table:

==============  ==========  ======== ======
Experiment      Perplexity  Accuracy h
==============  ==========  ======== ======
Linear AR       3.99        32.9%    -
CNN AR          3.85        35.6%    -
Reference AR    3.84        36.5%    -
BMM             3.79        36.8%    -
Linear BEAR     3.79        36.8%    0.0433
CNN BEAR        3.79        36.8%    0.0116
Reference BEAR  3.79        36.8%    0.0142
==============  ==========  ======== ======

A few things to note:

* In comparison with the values reported in the publication for a lag of 13, the perplexities are much larger and the learned :math:`h`'s are much smaller.
  This is due to the fact that the closest (in KL) AR model of lag 5, expectedly, has a much larger perplexity than the best fit of lag 13, and is also much closer to those module specified by linear and CNN models.
* As demonstrated in the publication, with enough data, the relative benefit of the EAR over the AR is minimal. As the lag is only 5 in this example, the data is enough to make this the case.
  However, the BEAR still outperforms the BMM in the fifth or sixth decimal space (not shown in table).
* The value of :math:`h` is in proportion to expected misspecification of the parametric AR model.
  Interestingly, in this simple case, the CNN is better specified than the reference.
  Running ``python train_bear_ref.py config_files/bear_cnn_bear.cfg`` indeed yields a substantially lower :math:`h` at 0.00422.
* The learned stop rate outputted with the reference AR model is near 151 (not shown in table).
  150 is the read length so that 1/151 transitions are stops.
* AR models trained using empirical Bayes (train_ar=1) perform better as BEAR models than those trained using maximum likelihood (train_ar=0) (not shown in table).
  The same is true for AR models trained using maximal likelihood evaluated as AR models.
