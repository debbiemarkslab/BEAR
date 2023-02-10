#!/usr/bin/env python

"""
Extract summary statistics (kmer count transitions) from large nucleotide
datasets in order to train BEAR models. Usage:

``python summarize.py file out_prefix [-l L] [-nf NF] [-r R] [-pr PR] [-mk MK] [-mf MF] [-p P] [-t T]``

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
import itertools
from subprocess import PIPE
from typing import Any


# --- Stage 1: Extract prefixes and suffixes from input files. ---

def load_input(in_file, file_type):
    if file_type == 'fq':
        return FastqGeneralIterator(in_file)
    elif file_type == 'fa':
        return SimpleFastaParser(in_file)


class KMC_input_fqs:
    """
    For each input fa or fq file, we write a set of fq files of various types:
    'full' is just each full sequence, 'pre' are the prefixes of length 1 to lag,
    'suf' are the suffixes of length 1 to lag + 1. That is, up to 2*lag + 1 files per input.
    
    Each of these files is given a group id: 'fq_{}_{}_{}'.format(group, seq_type, k)
    where seq_type is one of 'full', 'pre', 'suf'.
    Later all files of the same type and group will be concatenated and put through KMC.
    
    Calling this class writes these output files.
    """
    def __init__(self, file_num, file, group, file_type, out_prefix, lag, reverse, pr):

        self.file_num = file_num
        self.file = file
        self.group = group

        # Shorten file type to KMC's convention.
        self.file_type = file_type
        self.out_prefix = out_prefix
        self.lag = lag
        self.reverse = reverse
        self.pr = pr

        # Set up output file names.
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

        # Get group ids for output files
        if self.pr:
            self.file_out_id_keys = (
                [id_key(self.file_out_names['full'], self.group, 'full', li+1)
                 for li in range(self.lag + 1)]
                + [id_key(self.file_out_names['pre'][li], self.group, 'pre', li+1)
                   for li in range(self.lag)])
        else:
            self.file_out_id_keys = (
                [id_key(self.file_out_names['full'], self.group, 'full', self.lag+1)]
                + [id_key(self.file_out_names['pre'], self.group, 'pre', self.lag)])
        self.file_out_id_keys = (self.file_out_id_keys
            + [id_key(self.file_out_names['suf'][li], self.group, 'suf', li+1)
               for li in range(self.lag)])

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


class id_key:
    """Store outputs from Stage 1."""
    def __init__(self, file, group, seq_type, k):
        self.id = '{}_{}_{}'.format(group, seq_type, k)
        self.path = file


def merge_id_keys(all_id_keys, out_prefix, kmc_path, temp_path, memory):
    """Make a dictionary with lists of the paths to all files of the same id_key,
    "0_pre_2" for example. Then initialize a KMC run for all sequence files of the
    same id_key."""
    keys_outs = defaultdict(list)
    for id_key in all_id_keys:
        keys_outs[id_key.id].append(id_key.path)

    return [KMC_run(*key.split('_'), keys_outs[key],
                    out_prefix, kmc_path, temp_path, memory)
            for key in keys_outs]


def preprocess_seq_files(seq_list_file, lag, reverse, pr, out_prefix,
                         kmc_path, temp_path, kmc_memory, run_step):
    """ Break up input files into files to input to KMC,
    including prefixes and suffixes."""
    # Load list of input files -- 
    # first column is paths to the files, second is their "group" and
    # third is their type 'fa' or 'fq'.
    files_list = np.genfromtxt(seq_list_file, delimiter=',', dtype=str)
    file_num = len(files_list)
    n_groups = int(np.max(files_list[:, 1].astype(int)) + 1)
    all_kmc_input_file_id_keys = []
    jobs = []
    for file_num, (indiv_file, group, file_type) in enumerate(files_list):
        # Format sequences before inputting into KMC. This involves creating from each
        # input seuqnece file, prefixes and suffixes of various lengths.
        # We create ids for each of these reformatted sequence files.
        # Calling in_files will actually write these output files.
        in_files = KMC_input_fqs(file_num, indiv_file, group, file_type,
                                 out_prefix, lag, reverse, pr)
        all_kmc_input_file_id_keys += in_files.file_out_id_keys
         
        # Write the output file if not done already.
        if run_step:
            p = multiprocessing.Process(target=in_files)
            jobs.append(p)
            p.start()

    # Wait for all processes to finish.
    if run_step:
        for job in jobs:
            job.join()

    # Initialize KMC runs for each group and sequence type (pre, suf, full).
    kmc_runs = merge_id_keys(all_kmc_input_file_id_keys,
                             out_prefix, kmc_path, temp_path, kmc_memory)

    return n_groups, kmc_runs


# --- Stage 2: Count kmers with KMC. ---
class KMC_run:
    """An object of the inputs and outputs of running KMC on a set of files.
    The output will be a lexicographically sorted list of kmers and
    their counts in a tsv. Calling KMC_run runs the KMC job.
    KMC_run.get_size gets the size of the output tsv."""
    def __init__(self, group, seq_type, k, in_files,
                 out_prefix, kmc_path, temp_path, memory):
        self.group = group
        self.seq_type = seq_type
        self.k = k
        self.in_files = in_files
        self.out_prefix = out_prefix
        self.temp_path = temp_path
        self.memory = memory
        self.kmc_path = kmc_path

        # File structure.
        self.key = '{}_{}_{}'.format(self.group, self.seq_type, self.k)
        self.in_files_file = '{}_kmc_in_{}.txt'.format(self.out_prefix,
                                                       self.key)
        self.inter_file = '{}_kmc_inter_{}.res'.format(self.out_prefix,
                                                       self.key)
        self.sort_file = '{}_kmc_inter_{}_sort.res'.format(
                            self.out_prefix, self.key)
        self.out_file = '{}_kmc_out_{}.tsv'.format(self.out_prefix,
                                                   self.key)

    def __call__(self):

        # Write concatenated input file list.
        with open(self.in_files_file, 'w') as iw:
            iw.write('\n'.join(self.in_files))
        # Run kmc.
        kmc_call = (
            '{} -v -b -k{} -m{}'.format(
                        os.path.join(self.kmc_path, 'kmc'), self.k, self.memory)
            + ' -ci1 -cs1000000000000 -cx1000000000000 '
            + '-fq @{} {} {}'.format(
                    self.in_files_file, self.inter_file, self.temp_path))
        out_kmc = subprocess.run(kmc_call, shell=True, capture_output=True)
        stdout_kmc = out_kmc.stdout.decode("utf-8")
        stderr_kmc = out_kmc.stderr.decode("utf-8")

        # Run kmc sort and dump.
        kmc_dump_call = '{} transform {} sort {} dump {}'.format(
            os.path.join(self.kmc_path, 'kmc_tools'), self.inter_file,
            self.sort_file, self.out_file)
        out_kmc_dump = subprocess.run(kmc_dump_call, shell=True,
                                      capture_output=True)
        stdout_kmc_dump = out_kmc_dump.stdout.decode("utf-8")
        stderr_kmc_dump = out_kmc_dump.stderr.decode("utf-8")

        # Save warning files.
        with open('{}_kmc_stdout_{}.txt'.format(self.out_prefix,
                                                self.key), 'w') as f:
            f.write('--- kmc ---\n')
            f.write(stdout_kmc)
            f.write('--- kmc dump ---\n')
            f.write(stdout_kmc_dump)
        with open('{}_kmc_stderr_{}.txt'.format(self.out_prefix,
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


def run_kmc(kmc_runs, run_step):
    """Run KMC on every set of files."""
    # Run sequentially.
    total_out_size = 0
    for kmc_run in kmc_runs:
        if run_step:
            kmc_run()
        total_out_size += kmc_run.get_size()
    return total_out_size


# --- Stage 3 ---
alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, ']': 4}


class Register:
    """ Object that takes and writes counts.
    Has two modes: seq_type != 'pre':
        In this case, writers is None and a list of n_bins writers is
        specified when calling write, max_lag is not used,
        and each kmer is randomly written to one of the writers.
    If seq_type = 'pre':
        writers is 2D: max_lag x 2 ** n_bin_bits.
        A prefix of length l gets written to l+1, l+2, ..., max_lag.
        Note prefixes are prepended with a '[' so a prefix of length
        l gets written to l, l+1, ... .
    """
    def __init__(self, n_groups, n_bin_bits, seq_type, max_lag,
                 writers=None):
        # Initialize parameters
        self.n_groups = n_groups
        self.max_lag = max_lag #used only if seq_type = 'pre'
        
        # Initialize current kmer and its counts.
        self.lag_kmer = ''
        self.counts = [[0 for j in range(len(alphabet))]
                       for i in range(n_groups)]
        self.init = True

        # Initialize writers.
        self.n_bin_bits = n_bin_bits
        self.writers = writers
        self.out_ind = 0
        self.seq_type = seq_type

    def add_next_kmer_count(self, next_kmer, next_count, next_group, writer=None):
        # Add counts and dump if new kmer.
        next_lag_kmer = next_kmer[:-1]

        # Initialize.
        if self.init:
            self.lag_kmer = next_lag_kmer
            self.init = False

        # Check if this is a new k-mer.
        if next_lag_kmer != self.lag_kmer:
            # Write what's already recorded, empty counts.
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
            # Write the prefix count for all sets greater than the kmer len.
            len_lag_kmer = len(self.lag_kmer)
            for li in range(len_lag_kmer-1, self.max_lag):
                if self.n_bin_bits > 0:
                    self.out_ind = random.getrandbits(self.n_bin_bits)
                # Prepend correct number of [ for this model lag and write.
                self.writers[li][self.out_ind].write(
                    '['*(li+1 - len_lag_kmer) + self.lag_kmer + '\t'
                    + elem_str + '\n')
        else:
            # Write lag kmer and counts toa. random bin.
            if self.n_bin_bits > 0:
                self.out_ind = random.getrandbits(self.n_bin_bits)
            writer[self.out_ind].write(
                    self.lag_kmer + '\t' + elem_str + '\n')

        # Wipe counts.
        for i in range(self.n_groups):
            for j in range(len(alphabet)):
                self.counts[i][j] = 0


class Pre_Consolidate:
    """Consolidate prefix kmer counting results."""
    def __init__(self, kmc_runs, n_groups, n_bin_bits, max_lag, out_prefix):

        # Load the KMC run coresponding to the prefixes of length lag
        # of each group.
        self.max_lag = max_lag
        self.queue = []
        for kmc_run in kmc_runs:
            out_file, group, seq_type, k = kmc_run.get_output_info()
            if seq_type == 'pre' and k == self.max_lag:
                self.load_onto_queue(open(out_file, 'r'), group, 'pre')

        # Initialize registers and writers.
        self.n_bins = 2 ** n_bin_bits
        self.writers = [[open('{}_lag_{}_file_{}_kmer_start_.tsv'.format(
                              out_prefix, li+1, bi), 'w')
                         for bi in range(self.n_bins)]
                        for li in np.arange(self.max_lag)]
        # Registers take in kmer counts line-by-line and write to the writers.
        # Make a register for each kmer lag
        self.registers = [Register(n_groups, n_bin_bits, 'pre', self.max_lag,
                                   writers=self.writers)
                          for li in np.arange(self.max_lag)]

    def __call__(self):
        # Iterate through heap.
        while len(self.queue) > 0:
            heapq.heappop(self.queue)(self)
        # Write final kmer counts and close writers.
        self.end()

    def load_onto_queue(self, file_handle, group, seq_type):
        # Load next kmer and counts onto queue.
        line = file_handle.readline()
        if line != '':
            line0, line1 = line.split('\t')
            kmer = '[' + line0
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, None))
            heapq.heappush(self.queue, next_unit)

    def send_to_registers(self, kmer, count, group, writer=None):
        # Send beginning of kmer to each register.
        for li in np.arange(len(kmer)-1):
            self.registers[li].add_next_kmer_count(kmer[:(li+2)], count, group)

    def end(self):
        # Write final kmer counts and close writers.
        for li in range(self.max_lag):
            self.registers[li].write()
            for bi in range(self.n_bins):
                self.writers[li][bi].close()


class Consolidate:
    """Consolidate kmer counting results."""
    def __init__(self, kmc_runs, n_groups, n_bin_bits, lag, max_lag, out_prefix,
                 kmer_start):

        # Initialize queues.
        self.max_lag = max_lag
        self.lag = lag
        self.len_start = len(kmer_start)
        self.kmer_start = kmer_start
        # get suffixes for lag and full for max_lag.
        self.queue = []
        self.init_queue = []
        self.no_kmers_seen = True # tracks if the file is empty
        for kmc_run in kmc_runs:
            out_file, group, seq_type, k = kmc_run.get_output_info()
            if ((seq_type == 'suf' and k >= lag)
                or (seq_type == 'full' and k == max_lag + 1)):
                self.init_queue.append((out_file, group, seq_type))

        # Initialize register.
        self.n_bins = 2 ** n_bin_bits
        self.writer_names = ['{}_lag_{}_file_{}_kmer_start_{}.tsv'.format(
                                 out_prefix, lag, bi, self.kmer_start)
                             for bi in range(self.n_bins)]
        self.register = Register(n_groups, n_bin_bits, seq_type, max_lag)

    def start(self):
        # Start reading in queue.
        for out_file, group, seq_type in self.init_queue:
            self.load_onto_queue(open(out_file, 'r'), group, seq_type)

        # Open writers.
        if self.kmer_start == '':
            self.writer = [open(elem, 'a') for elem in self.writer_names]
        else:
            self.writer = [open(elem, 'w') for elem in self.writer_names]

    def __call__(self):

        # Put jobs on heap and open writers.
        self.start()

        # Iterate through queue.
        while len(self.queue) > 0:
            heapq.heappop(self.queue)(self, self.writer)

        # Write final kmer counts and close writers.
        if not self.no_kmers_seen:
            self.end()

    def load_onto_queue(self, file_handle, group, seq_type):
        # Load next kmer and counts.
        line = file_handle.readline()
        # go to first instance of kmer_start, or end of file
        while line[:self.len_start] != self.kmer_start and line != '':
            line = file_handle.readline()
        if line != '':
            self.no_kmers_seen = False
            kmer, line1 = line.split('\t')
            if seq_type == 'suf':
                kmer += ']'
            kmer = kmer[:(self.lag+1)]
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, seq_type))
            heapq.heappush(self.queue, next_unit)

    def send_to_registers(self, kmer, count, group, writer=None):
        self.register.add_next_kmer_count(kmer, count, group, writer=writer)

    def end(self):
        # Write final kmer counts and close writers.
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
    them. That is, 2 ** n_bin_bits = n_bins."""
    approx_n_chunks = total_size * n_groups / (mf * 1e9)
    n_chunks_2 = int(max([
        np.ceil(np.log(approx_n_chunks) / np.log(2)), 0]))
    return n_chunks_2


def get_starts(L):
    """ Get all kmers of the DNA alphabet of length L. """
    if L < 1:
        return ['']
    else:
        list_ = L * [['A', 'C', 'G', 'T']]
        return [''.join(letters)
                for letters in itertools.product(*list_)]

def stage3(kmc_runs, total_size, n_groups, lag, chunk_size, len_start,
           out_prefix):
    """Merge KMC output kmer counts to produce kmer-transition counts for all
    lags."""
    # Initialize registers and writers.
    n_bin_bits = compute_n_bin_bits(total_size, n_groups, chunk_size)

    # Process prefixes.
    Pre_Consolidate(kmc_runs, n_groups, n_bin_bits, lag,
                    out_prefix)()

    # (Multi)process files.
    # Multiprocess across kmer starts and lags
    jobs = []
    for li in range(lag):
        len_comp = np.min([len_start, np.max([li-4, 0])])
        for kmer_start in get_starts(len_comp):
            consolidate = Consolidate(kmc_runs, n_groups, n_bin_bits, li+1,
                                      lag, out_prefix, kmer_start)
            p = multiprocessing.Process(target=consolidate)
            jobs.append(p)
            p.start()
            
    # Wait for all processes to finish
    for job in jobs:
        job.join()

    # Concatenate different starts
    n_bins = 2 ** n_bin_bits
    for li in range(lag):
        len_comp = np.min([len_start, np.max([li-4, 0])])
        kmers = get_starts(len_comp)
        if '' not in kmers:
            kmers += ['']
        for bi in range(n_bins):
            fnames = ['{}_lag_{}_file_{}_kmer_start_{}.tsv'.format(
                      out_prefix, li+1, bi, k) for k in kmers]
            out_fname = '{}_lag_{}_file_{}.tsv'.format(
                out_prefix, li+1, bi)
            command = 'rm ' + out_fname
            subprocess.run(command, shell=True)
            command = 'cat {} > {}'.format(
                ' '.join(fnames), out_fname)
            subprocess.run(command, shell=True)

    return 2 ** n_bin_bits


# --- Main. ---
def run(args):
    """Run entire summarization."""
    # Preprocess before KMC.
    print('Start: Stage 1...', datetime.datetime.now())
    n_groups, kmc_runs = preprocess_seq_files(
        args.file, args.l, args.r, args.pr, args.out_prefix,
        args.p, args.t, args.mk, not args.s3)

    # Run KMC.
    print('Stage 2...', datetime.datetime.now())
    total_size = run_kmc(kmc_runs, not args.s3)

    if args.s12:
        return

    # Postprocess after KMC.
    print('Stage 3...', datetime.datetime.now())
    n_bins = stage3(kmc_runs, total_size, n_groups, args.l, args.mf,
                    args.ls, args.out_prefix)
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
    parser.add_argument('-ls', default=4, type=int,
                        help='Length at which to split BEAR task. Breaks into'
                              +' 4**ls tasks.')
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
