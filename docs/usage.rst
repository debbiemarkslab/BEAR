############
Introduction
############

This is a library for building Bayesian embedded autoregressive (BEAR) models,
proposed in Amin et al. (link TODO). It provides (1) a script for extracting
summary statistics from large sequence datasets (which relies on KMC),
(2) a python package with tools for building BEAR models in general
(which relies on TensorFlow), and (3) scripts for training and evaluating
specific example BEAR models. To use the package, follow the
:ref:`installation instructions<Installation>`
and clone the `package repository`_.
To get started with an example dataset, jump to the :ref:`tutorial<Tutorial>`.

.. _package repository: https://github.com/AlanNawzadAmin/BEAR

###########################
Preprocessing sequence data
###########################
************
summarize.py
************

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

The loaded dataset can then be used to train an AR or BEAR model with the
``bear_net`` submodule:

.. autofunction:: bear_model.bear_net.train

One then may evaluate the performance of the trained model:

.. autofunction:: bear_model.bear_net.evaluation

One may also evaluate the performance of the trained BEAR model for mutiple values of the concentration parameter :math:`h`:

.. autofunction:: bear_model.bear_net.h_scan

To recover the concentration parameter/misspecification diagnostic :math:`h` and
the learned autoregressive function
from the outputted list of parameters, use the ``change_scope_params`` function:

.. autofunction:: bear_model.bear_net.change_scope_params

To embed your own autoregressive function, create a new
``make_ar_func_...`` function using the examples
:func:`bear_model.ar_funcs.make_ar_func_linear` and
:func:`bear_model.ar_funcs.make_ar_func_cnn` as templates.

********
bear_ref
********

Besides standard autoregressive models like linear models
and neural networks, one can also build an autoregressive model based
on a reference genome. In particular, we can predict the next base by looking up the
previous :math:`L` bases :math:`k` in the reference, normalizing the observed transition
counts :math:`\#_{\text{ref}}(k,b)` to form a probability, and accounting for
noise using a Jukes-Cantor mutation model (with error rate :math:`\tau`).
This gives the autoregressive function

.. math::
    \tilde{f}_{k,b} = e^{-\tau}\frac{\#_{\text{ref}}(k,b)}{\sum_{b'\neq\$}\#_{\text{ref}}(k,b')}+\left(1-e^{-\tau}\right)\frac{1}{|\mathcal{B}|}

for :math:`b\neq\$` and :math:`\tilde{f}_{k,\$}=0`, where :math:`\$` is the stop
symbol and :math:`|\mathcal{B}|` is the alphabet size (excluding the stop symbol).
When :math:`\sum_{b'\neq\$}\#_{\text{ref}}(k,b') = 0`, we default to
:math:`\tilde{f}_{k,b} = 1/|\mathcal{B}|`.
Reference genomes do not include stop symbols, and so do not provide predictions
of read length; to account for this problem in a generalized way we introduce another
AR function :math:`g` and combine it with the reference-based prediction,

.. math::
    f_{k,b} = \nu g_{k,b} + (1-\nu)\tilde{f}_{k,b}

for some :math:`\nu\in(0, 1)`. The function :math:`g` is typically chosen to
predict a stop symbol with probability 1; this particular function is implemented
in :func:`bear_model.ar_funcs.make_ar_func_stop`.
To train the model efficiently, we preprocess the reference sequence
along with the training and testing data, such that
:math:`\#_{\text{ref}}` forms a column of the summarized data.
The submodule ``bear_ref`` trains this reference-based
BEAR model, optimizing the hyperparameters :math:`\tau` and :math:`\nu` as well as
any additional parameters in :math:`g`.

.. autofunction:: bear_model.bear_ref.train

The functions :func:`bear_model.bear_ref.evaluate` and
:func:`bear_model.bear_net.change_scope_params`  are analogous to the functions
in ``bear_net`` with the same names.

**********************************
Getting probabilities of mutations
**********************************
Having trained a BEAR or BMM model using ``bear_net`` (``bear_ref`` not yet supported) one can calculate the probability of new sequences or the probability of mutations to a wild type sequence.
One can do this by sampling AR models from BEAR and calculating probabilities ffor each of these AR models, or by simply calculating the probabilities under the MAP model under a BEAR or BMM model.
One may calculaate the probability fo given sequences:

.. autofunction:: bear_model.get_var_probs.get_bear_probs_seqs

or a list of mutations of a wild type sequence:

.. autofunction:: bear_model.get_var_probs.get_bear_probs

###################
Example BEAR models
###################

We have also written scripts that will perform the above workflow 
- loading data, training, evaluating, and saving the list of parameters using ``dill`` -
with parameters specified in a config file.
Config files contain information about the training and testing data; parameters for training and testing; and parameters of the AR model.
Example config  files may be found in ``bear_model/models/config_files``.
Descriptions of the outputs of these scripts in given in the tutorial below.

************************
bear_model/models/train_bear_net.py
************************

.. automodule:: bear_model.models.train_bear_net

************************
bear_model/models/train_bear_ref.py
************************

.. automodule:: bear_model.models.train_bear_ref

If the data includes a reference, the same config file may be used by either of the above scripts,
with the specified AR model interpreted as :math:`g` when used with ``bear_model/models/train_bear_ref.py``.
These scripts are easiest to use through the command line, as shown in the tutorial below.

########
Tutorial
########

In this tutorial, we will apply BEAR models to whole genome
sequencing data from the Salmonella bacteriophage YSD1. The data and the
reference assembly are from
`Dunstan et al. (2019) <https://doi.org/10.1111/mmi.14396>`_; the NCBI SRA
record is
`here <https://www.ncbi.nlm.nih.gov/sra/?term=ERR956946>`_.

**Part 1: preprocessing**

First, navigate to the folder ``bear_model`` and download the example dataset and extract its contents (we assume
throughout that you are in the ``bear_model`` folder).

``wget https://marks.hms.harvard.edu/bear/ysd1_example.tar.gz``

``tar -xvf ysd1_example.tar.gz -C data``

There should now be five data files in the ``data`` subfolder,
corresponding to
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
Finally, before training a non-vanilla BEAR model, datasets should be shuffled.
In Linux,

``shuf data/ysd1_lag_5_file_0.tsv -o data/ysd1_lag_5_file_0_shuf.tsv``

On a Mac, replace ``shuf`` with ``gshuf`` (you may first need to install
GNU coreutils, via e.g. ``brew install coreutils``).
Note that shuffling is done in memory, so when using large lags on large datasets,
you can use the **-mf** flag
in `summarize.py` to ensure its output files are sufficiently small.
A preshuffled dataset is provided in the file
``models/data/ysd1_lag_5_file_0_preshuf.tsv``
to ensure that part 2 of this tutorial is reproducible and can be run
independently of KMC.

**Part 2: training**

Now we can train AR or BEAR model via maximal likelihood or empirical Bayes respectively.
We will be using
Config files for training three different AR models and their corresponding
BEAR model can be found in the folder ``bear_model/models/config_files``.
You can run these examples on your own shuffled dataset from part 1 by editing
the config files to set
``start_token = ysd1_lag_5_file_0_shuf.tsv``; by default the config files use the
example preshuffled file.
Note that in these examples we fix the model lag at the small value of 5 so
that it can be trained quickly; normally we would choose the lag based on maximum
marginal likelihood.
All 6 config files use the same training protocol and differ only in
the AR functions they use (see config file section ``[model]``)
and whether they are BEAR or AR models (see config file entry ``train_ar``).
To run the examples, move to the ``bear_models`` directory and run the following:

**Linear AR** ``python models/train_bear_net.py models/config_files/bear_lin_ar.cfg``

**CNN AR** ``python models/train_bear_net.py models/config_files/bear_cnn_ar.cfg``

**Reference AR** ``python models/train_bear_ref.py models/config_files/bear_stop_ar.cfg``

**Linear BEAR** ``python models/train_bear_net.py models/config_files/bear_lin_bear.cfg``

**CNN BEAR** ``python models/train_bear_net.py models/config_files/bear_cnn_bear.cfg``

**Reference BEAR** ``python models/train_bear_ref.py models/config_files/bear_stop_bear.cfg``

Each example should not take more than a few minutes to run.
The scripts each output a folder named with the time at which they were run in
``out_data/logs`` (two folders may already be present if you tested your installation).
This output folders contains:

* A progress file that can be used to visualize the training curve using
  TensorBoard.
* A config file that consists of the input config file appended with the
  training results, in the section ``[results]``.
  (Note that even when the BEAR model is the model that is trained, the
  perplexity, log likelihood and accuracy of the embedded AR model
  and vanilla BEAR model (BMM) are also reported; when the AR model is trained,
  the performance of the corresponding BEAR model with :math:`h=1` is reported.)
* A pickle file with the learned hyperparameters. These hyperparameters can be
  recovered using the ``dill`` package.

The performance results from these example models should match closely the
following table:

==============  ==========  ======== ======
Experiment      Perplexity  Accuracy h
==============  ==========  ======== ======
Linear AR       3.99        32.9%    N/A
CNN AR          3.85        35.8%    N/A
Reference AR    3.84        36.5%    N/A
BMM             3.79        36.8%    N/A
Linear BEAR     3.79        36.8%    0.0433
CNN BEAR        3.79        36.8%    0.0119
Reference BEAR  3.79        36.8%    0.0142
==============  ==========  ======== ======

A few things to note from looking at the results:

* In the paper we used a lag of 13 for this dataset, chosen by maximum marginal likelihood.
  Here, using a lag of 5, the perplexities are much larger and the
  :math:`h` values are much smaller.
  This is due to the fact that the closest (in KL) AR model of lag 5,
  unsurprisingly, has a much larger perplexity than the lag 13 model, and
  is also much closer to the linear and CNN models.
* As demonstrated in the paper, with enough data, the relative benefit of
  the BEAR over the vanilla BEAR is minimal. Since the lag is only 5 in this example, the
  data is sufficiently large (relative to the model flexibility)
  to make this true. However, the BEAR still outperforms the vanilla BEAR in
  the fifth or sixth decimal space (not shown in table).
* The value of :math:`h` is in proportion to expected misspecification of the
  parametric AR model.
  Interestingly in this simple case, unlike most example datasets in the publication,
  the CNN is better specified than the
  reference (:math:`h=0.0116` versus :math:`h=0.0142`).
  Running a reference model with :math:`g` set to a CNN, via
  ``python train_bear_ref.py config_files/bear_cnn_bear.cfg``,
  yields an even lower :math:`h=0.00422`, suggesting the two models learn
  complementary information.
* The learned stop rate :math:`\nu` in the reference AR model is near 151
  (not shown in table).
  150 is the read length, so 1/151 transitions are stops on average.
* Embedded AR models trained in the context of a BEAR model produce better BEAR
  models than AR models trained using maximum likelihood (not shown in table).
  The same is true for AR models trained using maximal likelihood and evaluated
  as AR models rather than as BEAR models.
