============
Installation
============

To install the python package, use a Python 3 environment and run::

    $ pip install bear-model

Preprocessing fasta and fastq files requires installation of `KMC`_.
This code has been tested on KMC version 3.1.1. Mac users should run
``$ ulimit -n 2048`` to allow KMC to make a large number
of files; otherwise the preprocessing stage may fail with a `File not found`
error.

.. _KMC: https://github.com/refresh-bio/KMC/releases

To test your installation, first add the folder containing the KMC binaries
to your PATH (for instance, run ``$ export PATH="LOCATION/KMC3.1.1:$PATH"``
on Mac/Linux). Then run ``pytest`` in the BEAR directory.
All tests should pass.
