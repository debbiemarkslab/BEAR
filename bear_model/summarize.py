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
    
-ls : int, default = 3
    Length of sequence to split jobs by.
    
-nf : bool
    Do not count kmers in the forward direction.
    
-r : bool
    Also run KMC including the reverse compliment of sequences when counting.
    
-pr : bool
    Do all lags for pre and full KMCs. Highly recommended.

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
    
-s12 : bool
    Run only stages 1 and 2.

-s3 : bool
    Run only stage 3.
    
-s3_o : bool
    If only one group, speed up runtime but do not process ends.
    
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
from collections import defaultdict, Counter
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

    def fix_offset(self, offset):
        # 1. calc the potential offsets
        max_number_length = 100
        tab_char_len = 1
        new_line_char_len = 0
        max_slide = 4 * (int(self.k) + tab_char_len + max_number_length + new_line_char_len) # M_O
        max_slide = min(offset,max_slide)
        current_slide_len = max_slide

        file = open(self.out_file, 'r')
        file.seek(offset - max_slide)

        # if we are not at the begin of the file
        if max_slide < offset:
            # we are throwing out the first line
            first_line = next(file)
            first_line_length = len(first_line) + new_line_char_len
            current_slide_len -= first_line_length # M_1

        # now we will keep reading lines until we hit the offset
        number_chars_read  = 0
        lines_seen = []
        while number_chars_read <  current_slide_len:
            line = next(file)
            lines_seen.append(line)
            number_chars_read += len(line) + new_line_char_len
        
        # now we want to just get 4 lines (max) back
        if len(lines_seen) > 4:
            amount_to_remove = sum([ len(i) + new_line_char_len for i in lines_seen[:-4]])
            current_slide_len -= amount_to_remove 
            lines_seen = lines_seen[-4:]

        # now we need to get the last kmer in the block of 4 lines
        last_line = lines_seen[-1]
        last_kmer = last_line[0:(int(self.k) -1)]
        for line in lines_seen[:-1]:
            if line[0:(int(self.k) -1)] != last_kmer:
                current_slide_len -= len(line) + new_line_char_len

        
        return offset - current_slide_len
            
    def count_start_kmers(self, s3_once, max_len_start):
        if s3_once:
            # need to check that this gives you the size in bytes vs lines
            file_size = self.get_size()
            
            #offset are the actual byte offsets
            offset_counts = [{'':0}]
            for li in range(max_len_start):
                offset_counts.append(defaultdict())
                for j, kmer in enumerate(get_starts(li + 1)):
                    # approximate to initialize
                    approximate_offset = int(j * (file_size / 4**(li+1)))
                    if approximate_offset != 0:
                        offset_counts[li+1][kmer] = self.fix_offset(approximate_offset)
                    else:
                        offset_counts[li+1][kmer] = 0

            
            # calculate the bytes within each chunk
            bytes_per_chunk = defaultdict()
            for li in range(max_len_start):
                kmers = get_starts(li + 1)
                for j in range(len(kmers)):
                    if j + 1 == len(kmers):
                        offset_1 = file_size
                        offset_2 = offset_counts[li+1][kmers[j]]
                    else:
                        offset_1 = offset_counts[li+1][kmers[j+ 1]]
                        offset_2 = offset_counts[li+1][kmers[j]]
                    
                    bytes = offset_1 - offset_2
                    bytes_per_chunk[kmers[j]] = bytes
            
            bytes_per_chunk[''] = file_size
            self.offset_nums = offset_counts
            self.bytes_per_chunk = bytes_per_chunk
            self.s3_chunk = True

        else:
            start_counts = []
            offset_counts = []
            for li in range(max_len_start):
                start_counts.append(Counter())
                offset_counts.append(Counter())
                for kmer in get_starts(li + 1):
                    start_counts[li][kmer] = 0
                    offset_counts[li][kmer] = 0
            handel = open(self.out_file, 'r')
            line = handel.readline()
            n_lines = 0
            while line != '':
                n_lines += 1
                for li in range(max_len_start):
                    start_counts[li][line[:li+1]] += 1
                    offset_counts[li][line[:li+1]] += len(line)
                line = handel.readline()
            handel.close()
            
            self.start_counts = [{'':n_lines}] + start_counts
            self.offset_nums = [{'':0}]
            for li in range(max_len_start):
                kmer_starts = list(offset_counts[li].keys())
                assert np.all(np.argsort(kmer_starts) == np.arange(4 ** (li+1)))
                offset_nums = [offset_counts[li][kmer] for kmer in kmer_starts]
                self.offset_nums.append({kmer : np.sum(offset_nums[:i]) for i, kmer
                                        in enumerate(kmer_starts)})
            self.s3_chunk = False
        
    def get_file_handel(self, kmer_start):
        len_start = len(kmer_start)
        file = open(self.out_file, 'r')
        file.seek(self.offset_nums[len_start][kmer_start])

        if self.s3_chunk:
            n_bytes = self.bytes_per_chunk[kmer_start]
            byte_counter = 0
            while byte_counter < n_bytes:
                line = next(file)
                byte_counter += len(line)
                yield line

        else:
            n_lines = self.start_counts[len_start][kmer_start]
            line_counter = 0
            while line_counter < n_lines:
                line_counter += 1
                yield next(file)

    def get_size(self):
        # Get total size for next stage.
        out_size = os.path.getsize(self.out_file)
        return out_size

    def get_output_info(self):
        return self.get_file_handel, int(self.group), self.seq_type, int(self.k)


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
            get_handel, group, seq_type, k = kmc_run.get_output_info()
            if seq_type == 'pre' and k == self.max_lag:
                self.load_onto_queue(get_handel(''), group, 'pre')

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
        try:
            line = next(file_handle)
            line0, line1 = line.split('\t')
            kmer = '[' + line0
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, None))
            heapq.heappush(self.queue, next_unit)
        except StopIteration:
            1

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
                 kmer_start, pr, s3_once):

        # Initialize queues.
        self.max_lag = max_lag
        self.lag = lag
        self.len_start = len(kmer_start)
        self.kmer_start = kmer_start
        self.s3_once = s3_once
        # get suffixes for lag and full for max_lag.
        self.queue = []
        self.init_queue = []
        self.no_kmers_seen = True # tracks if the file is empty
        for kmc_run in kmc_runs:
            get_handel, group, seq_type, k = kmc_run.get_output_info()
            suf_lags = np.arange(lag, max_lag + 1) if not pr else [lag]
            if self.s3_once:
                suf_lags = []
            full_lag = 1 + (max_lag if not pr else lag)
            if ((seq_type == 'suf' and k in suf_lags)
                or (seq_type == 'full' and k == full_lag)):
                self.init_queue.append((get_handel, group, seq_type))

        # Initialize register.
        self.n_bins = 2 ** n_bin_bits
        self.writer_names = ['{}_lag_{}_file_{}_kmer_start_{}.tsv'.format(
                                 out_prefix, lag, bi, self.kmer_start)
                             for bi in range(self.n_bins)]
        self.register = Register(n_groups, n_bin_bits, seq_type, max_lag)

    def start(self):
        # Start reading in queue.
        for get_handel, group, seq_type in self.init_queue:
            self.load_onto_queue(get_handel(self.kmer_start), group, seq_type)

        # Open writers.
        if self.kmer_start == '' and not self.s3_once:
            self.writer = [open(elem, 'a') for elem in self.writer_names]
        else:
            self.writer = [open(elem, 'w') for elem in self.writer_names]

    def __call__(self):

        # Put jobs on heap and open writers.
        self.start()

        # Iterate through queue.
        if not self.s3_once:
            while len(self.queue) > 0:
                heapq.heappop(self.queue)(self, self.writer)
        else:
            assert len(self.init_queue) == 1
            get_handel, group, seq_type = self.init_queue[0]
            assert group == 0
            assert seq_type == 'full'
            # this is a fh for the start of the full file
            self.consolidate_once(get_handel(self.kmer_start))

        # Write final kmer counts and close writers.
        if not self.no_kmers_seen:
            self.end()

    def load_onto_queue(self, file_handle, group, seq_type):
        # Load next kmer and counts.
        try:
            line = next(file_handle)
            self.no_kmers_seen = False
            kmer, line1 = line.split('\t')
            if seq_type == 'suf':
                kmer += ']'
            kmer = kmer[:(self.lag+1)]
            count = int(line1[:-1])
            next_unit = Unit3i(kmer, (count, file_handle, group, seq_type))
            heapq.heappush(self.queue, next_unit)
        except StopIteration:
            1
            
    def consolidate_once(self, file_handle):
        # Consolidate in the case of no ends and one group
        for line in file_handle:
            # Load next kmer and counts.
            self.no_kmers_seen = False
            kmer, line1 = line.split('\t')
            kmer = kmer[:(self.lag+1)]
            count = int(line1[:-1])
            self.send_to_registers(kmer, count, 0, writer=self.writer)

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


def compute_n_bin_bits(total_size, n_groups, mf, len_start):
    """Compute number of output files per lag and bits needed to specify
    them. That is, 2 ** n_bin_bits = n_bins."""
    approx_n_chunks = total_size * n_groups / (mf * 1e9)
    n_chunks_2 = int(max([
        np.ceil(np.log(approx_n_chunks) / np.log(2)) - 2 * len_start, 0]))
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
           out_prefix, pr, s3_once=False):
    """Merge KMC output kmer counts to produce kmer-transition counts for all
    lags."""
    if s3_once:
        assert n_groups == 1, "Too many groups"

    # Initialize registers and writers.
    n_bin_bits = compute_n_bin_bits(total_size, n_groups, chunk_size, len_start)
    
    # Get start indices for KMC runs
    for kmc_run in kmc_runs:
        *_, k = kmc_run.get_output_info()
        len_comp = np.min([len_start, k])
        kmc_run.count_start_kmers(s3_once, len_comp)

    # Process prefixes.
    if not s3_once:
        Pre_Consolidate(kmc_runs, n_groups, n_bin_bits, lag,
                        out_prefix)()

    # (Multi)process files.
    # Multiprocess across kmer starts and lags
    jobs = []
    for li in range(lag):
        len_comp = np.min([len_start, np.max([li-4, 0])])
        for kmer_start in get_starts(len_comp):
            consolidate = Consolidate(kmc_runs, n_groups, n_bin_bits, li+1,
                                      lag, out_prefix, kmer_start, pr, s3_once)
            p = multiprocessing.Process(target=consolidate)
            jobs.append(p)
            p.start()
            
    # Wait for all processes to finish
    for job in jobs:
        job.join()

    # Concatenate different starts
    # n_bins = 2 ** n_bin_bits
    # for li in range(lag):
    #     len_comp = np.min([len_start, np.max([li-4, 0])])
    #     kmers = get_starts(len_comp)
    #     if '' not in kmers:
    #         kmers += ['']
    #     for bi in range(n_bins):
    #         fnames = ['{}_lag_{}_file_{}_kmer_start_{}.tsv'.format(
    #                   out_prefix, li+1, bi, k) for k in kmers]
    #         out_fname = '{}_lag_{}_file_{}.tsv'.format(
    #             out_prefix, li+1, bi)
    #         command = 'rm ' + out_fname
    #         subprocess.run(command, shell=True)
    #         command = 'cat {} > {}'.format(
    #             ' '.join(fnames), out_fname)
    #         subprocess.run(command, shell=True)

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
                    args.ls, args.out_prefix, args.pr, args.s3_o)
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
    parser.add_argument('-s3_o', action='store_true', default=False,
                        help='Only run stage 3 without ends for a single group.')
    args = parser.parse_args()
    main(args)
