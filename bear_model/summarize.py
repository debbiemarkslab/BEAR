#!/usr/bin/env python

"""
Extract summary statistics (kmer count transitions) from large nucleotide
datasets in order to train BEAR models. Usage:

``python summarize.py file out_prefix [-l L] [-r R] [-mk MK] [-mf MF] [-p P] [-t T]``

Input
-----

file : str
    The input csv file with rows in the format `FILE, GROUP, TYPE`
    where `FILE` is a path, `GROUP` is an integer, and `TYPE` is either `fa`
    (denoting fasta) or `fq` (denoting fastq). Files with the same group will
    have their counts merged. All files must contain DNA sequences
    (A, C, G, and T alone).

out_prefix : str
    The file name prefix (including path) for the output files.

-l : int, default = 10
    The maximum lag (the truncation level).
    
-nf : bool
    Do not count kmers in the forward direction.
    
-r : bool
    Also run KMC including the reverse compliment of sequences when counting.
    
-pr : bool
    Do all lags for pre and full KMCs.

-mk : float, default = 12
    Maximum amount of memory available, in gigabytes (corresponding to the
    KMC -m flag).

-mf : float, default = 0.1
    Maximum size of output files, in gigabytes.

-p : str, default = ''
    Path to KMC binaries. If these binaries are in your PATH, there is no
    need to use this option.

-t : str, default = 'tmp/'
    Folder in which to store KMC's intermediate results. A valid path MUST be
    provided.

You can run ``python summarize.py -h`` for help and more advanced options.

**Output:**

A collection of files::

    out_prefix_lag_1_0.tsv, ..., out_prefix_lag_1_N.tsv, ...,
    out_prefix_lag_L_0.tsv, ..., out_prefix_lag_L_N.tsv

where L is the maximum lag and there are N total files for each lag.
Each file is a tsv with rows of the format::

    kmer\t[[transition counts in group 0 files],[transition counts in group 1 files],...]

The symbol `[` in the kmer is the start symbol.
Each counts vector is in the order `A, C, G, T, $` where $ is the stop symbol.

.. caution:: KMC discards kmers with more than 4 billion counts, which may lead
    to errors on ultra large scale datasets.

.. caution:: The script is not intended for use on sequence data with N symbols;
    it will run but will not handle such missing data carefully.

.. caution:: The output is not lexicographically sorted, nor uniformly randomized.

"""

import argparse
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio import Seq
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import datetime
import heapq
import numpy as np
import multiprocessing
import os
import random
import subprocess
from subprocess import PIPE
from typing import Any


# --- Stage 1: Extract prefixes and suffixes from input files. ---

def load_input(in_file, file_type):

    if file_type == 'fq':
        return FastqGeneralIterator(in_file)
    elif file_type == 'fa':
        return SimpleFastaParser(in_file)


class Unit1i:
    """Processing input unit for Stage 1."""
    def __init__(self, file_num, file, group, file_type, args):

        self.file_num = file_num
        self.file = file
        self.group = group

        # Shorten file type to KMC's convention.
        self.file_type = file_type
        self.out_prefix = args.out_prefix
        self.lag = args.l
        self.reverse = args.r
        self.pr = args.pr

        # Set up file structure.
        self.file_out_names = {
                'suf': ['{}_{}_suf_{}.fastq'.format(
                            self.out_prefix, self.file_num, li+1)
                        for li in range(self.lag)]}
        if self.pr:
            self.file_out_names['pre'] = ['{}_{}_pre_{}.fastq'.format(
                            self.out_prefix, self.file_num, li+1)
                        for li in range(self.lag)]
        else:
            self.file_out_names['pre'] = '{}_{}_pre.fastq'.format(
                self.out_prefix, self.file_num)
        if self.file_type == 'fq' and not self.reverse:
            self.file_out_names['full'] = self.file
        else:
            self.file_out_names['full'] = '{}_{}_full.fastq'.format(
                                    self.out_prefix, self.file_num)
        # Format for next stage.
        if self.pr:
            self.output_units = (
                [Unit1o(self.file_out_names['full'], 'fq', self.group, 'full', li+1) for li in range(self.lag + 1)]
                + [Unit1o(self.file_out_names['pre'][li], 'fq', self.group, 'pre', li+1) for li in range(self.lag)])
        else:
            self.output_units = (
                [Unit1o(self.file_out_names['full'], 'fq', self.group, 'full', self.lag+1)]
                + [Unit1o(self.file_out_names['pre'], 'fq', self.group, 'pre', self.lag)])
        self.output_units = (self.output_units
            + [Unit1o(self.file_out_names['suf'][li], 'fq', self.group, 'suf', li+1) for li in range(self.lag)])

    def __write_out(self, file_out, not_init, name, seq):
        # Write full, if not in fastq format already.
        if self.file_type != 'fq' or self.reverse:
            if not_init:
                file_out['full'].write('\n')
            file_out['full'].write('@{}\n{}\n+\n{}'.format(
                name, seq, 'F'*len(seq)))
        # Write prefix.
        if self.pr:
            for li in range(self.lag):
                if not_init:
                    file_out['pre'][li].write('\n')
                file_out['pre'][li].write('@{}\n{}\n+\n{}'.format(
                    name, seq[:li+1], 'F'*(li+1)))
        else:
            if not_init:
                file_out['pre'].write('\n')
            file_out['pre'].write('@{}\n{}\n+\n{}'.format(
                    name, seq[:self.lag], 'F'*self.lag))
        # Write suffixes.
        for li in range(self.lag):
            if not_init:
                file_out['suf'][li].write('\n')
            file_out['suf'][li].write('@{}\n{}\n+\n{}'.format(
                name, seq[-(li+1):], 'F'*(li+1)))
    
    def __call__(self):

        # Initialize output files.
        file_out = {'suf': [open(self.file_out_names['suf'][li], 'w')
                            for li in range(self.lag)]}
        if self.pr:
            file_out['pre'] = [open(self.file_out_names['pre'][li], 'w')
                               for li in range(self.lag)]
        else:
            file_out['pre'] = open(self.file_out_names['pre'], 'w')
        if self.file_type != 'fq' or self.reverse:
            file_out['full'] = open(self.file_out_names['full'], 'w')

        # Open file.
        in_file = open(self.file, 'r')
        # Iterate through sequences in files.
        for j, elem in enumerate(load_input(in_file, self.file_type)):
            not_init = j > 0
            name, seq = elem[:2]

            self.__write_out(file_out, not_init, name, seq)
                
            if self.reverse:
                not_init = True
                seq = Seq.reverse_complement(seq)
                name = name + '_rev'
                
                self.__write_out(file_out, not_init, name, seq)

        # Close files.
        if self.pr:
            for li in range(self.lag):
                file_out['pre'][li].close()
        else:
            file_out['pre'].close()
        if self.file_type != 'fq' or self.reverse:
            file_out['full'].close()
        for li in range(self.lag):
            file_out['suf'][li].close()


class Unit1o:
    """Store outputs from Stage 1."""
    def __init__(self, file, file_type, group, seq_type, k):

        self.id = '{}_{}_{}_{}'.format(file_type, group, seq_type, k)
        self.value = file
        self.file_type = file_type


def merge_unit1o(lst, args):
    """Merge output for stage 2."""
    group_outs = defaultdict(list)
    for elem in lst:
        group_outs[elem.id].append(elem.value)

    return [Unit2i(*key.split('_'), group_outs[key], args)
            for key in group_outs]


def stage1(args):
    """Break up input files into prefixes and suffixes."""

    # Load input file.
    file_num = 0
    n_groups = 0
    indiv_files, groups = [], []
    jobs = []
    out_units = []
    with open(args.file, 'r', newline='') as file_list:
        file_list_reader = csv.reader(file_list)
        for indiv_file, group, file_type in file_list_reader:
            indiv_files.append(indiv_file)
            groups.append(int(group))
            n_groups = max([n_groups, groups[-1]])
            in_unit = Unit1i(file_num, indiv_file, groups[file_num], file_type,
                             args)
            out_units += in_unit.output_units
            if not args.s3:
                # Run job.
                p = multiprocessing.Process(target=in_unit)
                jobs.append(p)
                p.start()
            file_num += 1
    n_groups += 1

    if not args.s3:
        # Wait for all processes to finish.
        for job in jobs:
            job.join()

    # Consolidate output.
    unit2is = merge_unit1o(out_units, args)

    return n_groups, unit2is


# --- Stage 2: Count kmers with KMC. ---
class Unit2i:
    """Run KMC on a collection of input files."""
    def __init__(self, file_type, group, seq_type, k, in_files, args):

        self.group = group
        self.seq_type = seq_type
        self.k = k
        self.in_files = in_files
        self.file_type = file_type
        self.args = args

        # File structure.
        self.key = '{}_{}_{}'.format(self.group, self.seq_type, self.k)
        self.in_files_file = '{}_kmc_in_{}.txt'.format(self.args.out_prefix,
                                                       self.key)
        self.inter_file = '{}_kmc_inter_{}.res'.format(self.args.out_prefix,
                                                       self.key)
        self.sort_file = '{}_kmc_inter_{}_sort.res'.format(
                            self.args.out_prefix, self.key)
        self.out_file = '{}_kmc_out_{}.tsv'.format(self.args.out_prefix,
                                                   self.key)

    def __call__(self):

        # Write input file list.
        with open(self.in_files_file, 'w') as iw:
            iw.write('\n'.join(self.in_files))
        # Run kmc.
        kmc_call = (
            '{} -v -b -k{} -m{}'.format(
                        os.path.join(self.args.p, 'kmc'), self.k, self.args.mk)
            + ' -ci1 -cs1000000000000 -cx1000000000000 '
            + '-fq @{} {} {}'.format(
                    self.in_files_file, self.inter_file, self.args.t))
        out_kmc = subprocess.run(kmc_call, shell=True, capture_output=True)
        stdout_kmc = out_kmc.stdout.decode("utf-8")
        stderr_kmc = out_kmc.stderr.decode("utf-8")

        # Run kmc sort and dump.
        kmc_dump_call = '{} transform {} sort {} dump {}'.format(
            os.path.join(self.args.p, 'kmc_tools'), self.inter_file,
            self.sort_file, self.out_file)
        out_kmc_dump = subprocess.run(kmc_dump_call, shell=True,
                                      capture_output=True)
        stdout_kmc_dump = out_kmc_dump.stdout.decode("utf-8")
        stderr_kmc_dump = out_kmc_dump.stderr.decode("utf-8")

        # Save warning files.
        with open('{}_kmc_stdout_{}.txt'.format(self.args.out_prefix,
                                                self.key), 'w') as f:
            f.write('--- kmc ---\n')
            f.write(stdout_kmc)
            f.write('--- kmc dump ---\n')
            f.write(stdout_kmc_dump)
        with open('{}_kmc_stderr_{}.txt'.format(self.args.out_prefix,
                                                self.key), 'w') as f:
            f.write('--- kmc ---\n')
            f.write(stderr_kmc)
            f.write('--- kmc dump ---\n')
            f.write(stderr_kmc_dump)

    def get_size(self):
        # Get total size for next stage.
        out_size = os.path.getsize(self.out_file)
        return out_size

    def get_output_info(self):

        return self.out_file, int(self.group), self.seq_type, int(self.k)


def stage2(unit2is, args):
    """Run KMC on all files."""
    # Run sequentially.
    out_size = 0
    for unit2i in unit2is:
        if not args.s3:
            unit2i()
        out_size += unit2i.get_size()
    return out_size


# --- Stage 3 ---
alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, ']': 4}


class Register:
    """Merge counts with the same lag sequence."""
    def __init__(self, n_groups, n_bins_bits, lag, seq_type, args,
                 writers=None):

        # Data storage.
        self.lag_kmer = ''
        self.counts = [[0 for j in range(len(alphabet))]
                       for i in range(n_groups)]
        self.init = True
        self.n_groups = n_groups
        self.max_lag = args.l

        # Initialize writers.
        self.n_bins_bits = n_bins_bits
        self.writers = writers
        self.lag = lag
        self.out_ind = 0
        self.seq_type = seq_type

    def add(self, next_kmer, next_count, next_group, writer=None):

        next_lag_kmer = next_kmer[:-1]

        # Initialize.
        if self.init:
            self.lag_kmer = next_lag_kmer
            self.init = False

        if next_lag_kmer != self.lag_kmer:
            # Write what's already recorded.
            self.write(writer)
            # Store new lag kmer.
            self.lag_kmer = next_lag_kmer

        # Add counts.
        self.counts[next_group][alphabet[next_kmer[-1]]] += next_count

    def write(self, writer=None):

        # Convert count matrix to string.
        elem_str = '[[' + '],['.join([','.join(map(str, self.counts[i]))
                                      for i in range(self.n_groups)]) + ']]'
        if self.seq_type == 'pre':
            # Iterate over model lags.
            len_lag_kmer = len(self.lag_kmer)
            for li in range(len_lag_kmer-1, self.max_lag):
                if self.n_bins_bits > 0:
                    self.out_ind = random.getrandbits(self.n_bins_bits)
                # Prepend correct number of [ for this model lag and write.
                self.writers[li][self.out_ind].write(
                    '['*(li+1 - len_lag_kmer) + self.lag_kmer + '\t'
                    + elem_str + '\n')
        else:
            # Write lag kmer and counts.
            if self.n_bins_bits > 0:
                self.out_ind = random.getrandbits(self.n_bins_bits)
            writer[self.out_ind].write(
                    self.lag_kmer + '\t' + elem_str + '\n')

        # Wipe counts.
        for i in range(self.n_groups):
            for j in range(len(alphabet)):
                self.counts[i][j] = 0


class PreConsolidate:
    """Consolidate prefix kmer counting results."""
    def __init__(self, unit2is, n_groups, n_bin_bits, args):

        # Initialize queues.
        self.lag = args.l
        self.queue = []
        for unit2i in unit2is:
            out_file, group, seq_type, k = unit2i.get_output_info()
            if seq_type == 'pre' and k == args.l:
                self.load_onto_queue(open(out_file, 'r'), group)

        # Initialize registers and writers.
        self.n_bins = 2**n_bin_bits
        self.writers = [[open('{}_lag_{}_file_{}.tsv'.format(
                                args.out_prefix, li+1, bi), 'w')
                         for bi in range(self.n_bins)]
                        for li in range(args.l)]
        self.registers = [Register(n_groups, n_bin_bits, li+1, 'pre', args,
                                   writers=self.writers)
                          for li in range(args.l)]

    def __call__(self):

        # Iterate through heap.
        while len(self.queue) > 0:
            heapq.heappop(self.queue)(self)
        # Shut down.
        self.end()

    def load_onto_queue(self, file_handle, group, seq_type=None):
        # Load next kmer and counts.
        line = file_handle.readline()
        if line != '':
            line0, line1 = line.split('\t')
            kmer = '[' + line0
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, None))
            heapq.heappush(self.queue, next_unit)

    def send_to_registers(self, kmer, count, group, writer=None):

        for li in range(len(kmer)-1):
            self.registers[li].add(kmer[:(li+2)], count, group)

    def end(self):

        for li in range(self.lag):
            self.registers[li].write()
            for bi in range(self.n_bins):
                self.writers[li][bi].close()


class Consolidate:
    """Consolidate kmer counting results."""
    def __init__(self, unit2is, n_groups, n_bin_bits, lag, args):

        # Initialize queues.
        self.lag = lag
        self.queue = []
        self.init_queue = []
        for unit2i in unit2is:
            out_file, group, seq_type, k = unit2i.get_output_info()
            if (seq_type == 'suf' and k >= lag) or (seq_type == 'full' and k == args.l + 1):
                print(out_file, seq_type, k, lag)
                self.init_queue.append((out_file, group, seq_type))

        # Initialize register and writers.
        self.n_bins = 2**n_bin_bits
        self.writer_names = ['{}_lag_{}_file_{}.tsv'.format(
                                        args.out_prefix, lag, bi)
                             for bi in range(self.n_bins)]
        self.register = Register(n_groups, n_bin_bits, lag, seq_type, args)

    def start(self):

        # Start reading in queue.
        for elem in self.init_queue:
            self.load_onto_queue(open(elem[0], 'r'), elem[1], elem[2])

        # Open writer.
        self.writer = [open(elem, 'a') for elem in self.writer_names]

    def __call__(self):

        # Start up.
        self.start()

        # Iterate through heap.
        while len(self.queue) > 0:
            heapq.heappop(self.queue)(self, self.writer)

        # Shut down.
        self.end()

    def load_onto_queue(self, file_handle, group, seq_type):
        # Load next kmer and counts.
        line = file_handle.readline()
        if line != '':
            kmer, line1 = line.split('\t')
            if seq_type == 'suf':
                kmer += ']'
            kmer = kmer[:(self.lag+1)]
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, seq_type))
            heapq.heappush(self.queue, next_unit)

    def send_to_registers(self, kmer, count, group, writer=None):

        self.register.add(kmer, count, group, writer=writer)

    def end(self):

        self.register.write(self.writer)
        for elem in self.writer:
            elem.close()


@dataclass(order=True)
class Unit3i:
    """Individual kmer + count."""
    priority: str
    item: Any = field(compare=False)

    def __call__(self, consolidate, writer=None):
        # Load info.
        count, file_handle, group, seq_type = self.item

        # Send to registers.
        consolidate.send_to_registers(self.priority, count, group,
                                      writer=writer)

        # Add next line to queue.
        consolidate.load_onto_queue(file_handle, group, seq_type)


def compute_n_bin_bits(total_size, n_groups, mf):
    """Compute number of output files per lag and bits needed to specify
    them."""
    return int(max([np.ceil(np.log(total_size * n_groups /
                                   (mf * 1e9)) / np.log(2)), 0]))


def stage3(unit2is, total_size, n_groups, args):
    """Merge KMC output kmer counts to produce kmer-transition counts for all
    lags."""
    # Initialize registers and writers.
    n_bin_bits = compute_n_bin_bits(total_size, n_groups, args.mf)

    # Process prefixes.
    PreConsolidate(unit2is, n_groups, n_bin_bits, args)()

    # (Multi)process files.
    jobs = []
    for li in range(args.l):
        consolidate = Consolidate(unit2is, n_groups, n_bin_bits, li+1, args)
        p = multiprocessing.Process(target=consolidate)
        jobs.append(p)
        p.start()

    # Wait for all processes to finish
    for job in jobs:
        job.join()

    return 2**n_bin_bits


# --- Main. ---
def run(args):
    """Run entire summarization."""
    # Preprocess before KMC.
    print('Start: Stage 1...', datetime.datetime.now())
    n_groups, unit2is = stage1(args)

    # Run KMC.
    print('Stage 2...', datetime.datetime.now())
    total_size = stage2(unit2is, args)

    if args.s12:
        return

    # Postprocess after KMC.
    print('Stage 3...', datetime.datetime.now())
    n_bins = stage3(unit2is, total_size, n_groups, args)
    print('Finished.', datetime.datetime.now())
    return n_bins


def main(args):

    # Standard direction first.
    store_args_r = args.r
    args.r = False
    n_bins = None
    if not args.nf:
        n_bins = run(args)
    # Handle reverse case.
    n_bins_rev = None
    if store_args_r:
        args.r = True
        args.out_prefix += '_rev'
        n_bins_rev = run(args)

    return n_bins, n_bins_rev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Preprocess for collapsed BEAR training.")
    parser.add_argument('file',
                        help=('Input file: csv of individual files and their'
                              + ' group number.'))
    parser.add_argument('out_prefix', help='Prefix for output files.')
    parser.add_argument('-l', default=10, type=int,
                        help='Maximum lag of BEAR model.')
    parser.add_argument('-mk', default=12, type=float,
                        help='Maximum memory available to KMC (Gb)')
    parser.add_argument('-mf', default=0.1, type=float,
                        help='Maximum memory of final dataset chunks (Gb).')
    parser.add_argument('-p', default='',
                        help=('Path to folder with kmc scripts' +
                              ' (kmc and kmc_dump).'))
    parser.add_argument('-nf', action='store_true', default=False,
                        help='Do not compute the forward direction.')
    parser.add_argument('-r', action='store_true', default=False,
                        help='Compute reverse direction.')
    parser.add_argument('-pr', action='store_true', default=False,
                        help='KMC shorter pres as well.')
    parser.add_argument('-t', default='tmp/',
                        help=('Temporary directory for use by KMC. '
                              + 'Defaults to tmp/'))
    parser.add_argument('-s12', action='store_true', default=False,
                        help='Only run stages 1 and 2.')
    parser.add_argument('-s3', action='store_true', default=False,
                        help='Only run stage 3.')
    args = parser.parse_args()
    main(args)
