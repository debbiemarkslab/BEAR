============
Installation
============

To install the python package, use a clean Python 3 environment and run::

    pip install "git+https://github.com/debbiemarkslab/BEAR.git"

Preprocessing fasta and fastq files requires installation of `KMC`_.
This code has been tested on KMC version 3.1.1. Mac and Linux users should run
``ulimit -n 2048`` to allow KMC to make a large number
of files; otherwise the preprocessing stage may fail with a `File not found`
error. The folder with the KMC binaries should be added to the path;
for instance, run ``export PATH="LOCATION/KMC3.1.1:$PATH"``
on Mac/Linux.

.. _KMC: https://github.com/refresh-bio/KMC/releases

To test your installation, run ``pytest`` in the BEAR directory.
All tests should pass.
