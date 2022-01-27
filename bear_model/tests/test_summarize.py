from bear_model import summarize
from bear_model.tests import check_summarize

from Bio import Seq
from collections import defaultdict
import csv
import json
import numpy as np
import os
from pkg_resources import resource_filename


exdata_path = resource_filename('bear_model', 'tests/exdata')


def setup_args(pr=None):
    # --- Run preprocess code. ---
    class Args:

        def __init__(self, file, out_prefix, nf=None, l=None, mk=None, mf=None,
                     p=None, r=None, t=None, d1=None, d2=None, n=None, pr=pr,
                     s3=False, s12=False, num=10):
            self.file = file
            self.out_prefix = out_prefix
            self.nf = nf
            self.l = l
            self.mk = mk
            self.mf = mf
            self.p = p
            self.r = r
            self.t = t
            self.d1 = False
            self.d2 = False
            self.n = n
            self.pr = pr
            self.s12 = s12
            self.s3 = s3
            self.num = num

    max_lag = 10
    in_file_set = os.path.join(exdata_path, 'infiles.csv')
    out_prefix = os.path.join(exdata_path, 'out/out')
    os.makedirs(os.path.join(exdata_path, 'out'), exist_ok=True)
    os.makedirs(os.path.join(exdata_path, 'tmp'), exist_ok=True)
    args = Args(in_file_set, out_prefix, nf=False, l=max_lag, mk=2,
                mf=2, p='', r=True, t=os.path.join(exdata_path, 'tmp/'))
    return args, max_lag, in_file_set, out_prefix


def test_main():
    np.random.seed(1)
    # --- Set up input. ---
    in_file_set = os.path.join(exdata_path, 'infiles.csv')
    n_in_files = 5
    n_seqs_per_file = [3, 2, 2, 4, 2]
    groups = [0, 0, 2, 1, 1]
    file_types = ['fa', 'fq', 'fq', 'fa', 'fq']
    len_seqs = (14, 18)
    in_file_names = [os.path.join(exdata_path,
                                  'infile_{}.{}'.format(j, file_types[j]))
                     for j in range(n_in_files)]
    name_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    nmi = 0
    seqs = [['' for si in range(n_seqs_per_file[fi])]
            for fi in range(n_in_files)]
    with open(in_file_set, 'w') as infiles:
        for fi in range(n_in_files):
            with open(in_file_names[fi], 'w') as file_w:
                for si in range(n_seqs_per_file[fi]):
                    len_seq = np.random.randint(len_seqs[0], len_seqs[1])
                    for li in range(len_seq):
                        seqs[fi][si] += np.random.choice(['A', 'T', 'G', 'C'])
                    if file_types[fi] == 'fq':
                        file_w.write('@{}\n{}\n+\n{}\n'.format(
                            name_alphabet[nmi], seqs[fi][si], 'F'*len_seq))
                    elif file_types[fi] == 'fa':
                        file_w.write('>{}\n{}\n'.format(
                            name_alphabet[nmi], seqs[fi][si]))
                    nmi += 1
            infiles.write('{},{},{}\n'.format(in_file_names[fi], groups[fi],
                                              file_types[fi]))

    # --- Run. ---
    args, max_lag, in_file_set, out_prefix = setup_args()
    nbins, nbins_rev = summarize.main(args)

    # --- Count kmers in memory. ---
    n_groups = max(groups) + 1
    alphabet = summarize.alphabet
    kmer_counts = [defaultdict(lambda: [[0 for j in range(len(alphabet))]
                                        for i in range(n_groups)])
                   for li in range(max_lag)]
    kmer_counts_rev = [defaultdict(lambda: [[0 for j in range(len(alphabet))]
                                            for i in range(n_groups)])
                       for li in range(max_lag)]
    for li in range(max_lag):
        lag = li + 1
        for fi in range(n_in_files):
            for si in range(len(seqs[fi])):
                full_seq = '['*lag + seqs[fi][si] + ']'
                for j in range(lag, len(full_seq)):
                    lag_kmer = full_seq[(j-lag):j]
                    next_letter = full_seq[j]
                    kmer_counts[li][lag_kmer][groups[fi]][
                            alphabet[next_letter]] += 1
                    kmer_counts_rev[li][lag_kmer][groups[fi]][
                            alphabet[next_letter]] += 1
                # Reverse.
                full_seq = '['*lag + Seq.reverse_complement(seqs[fi][si]) + ']'
                for j in range(lag, len(full_seq)):
                    lag_kmer = full_seq[(j-lag):j]
                    next_letter = full_seq[j]
                    kmer_counts_rev[li][lag_kmer][groups[fi]][
                            alphabet[next_letter]] += 1

    # --- Check results. ---
    kmer_counts_check = [dict() for li in range(max_lag)]
    kmer_counts_check_rev = [dict() for li in range(max_lag)]
    for li in range(max_lag):
        for bi in range(nbins):
            out_file = '{}_lag_{}_file_{}.tsv'.format(out_prefix, li+1, bi)
            with open(out_file, 'r', newline='') as out_counts_file:
                out_reader = csv.reader(out_counts_file, delimiter='\t')
                for lag_kmer, count_str in out_reader:
                    if lag_kmer in kmer_counts_check[li]:
                        assert False
                    kmer_counts_check[li][lag_kmer] = json.loads(count_str)
        for bi in range(nbins_rev):
            # Reverse.
            out_file = '{}_rev_lag_{}_file_{}.tsv'.format(out_prefix, li+1, bi)
            with open(out_file, 'r', newline='') as out_counts_file:
                out_reader = csv.reader(out_counts_file, delimiter='\t')
                for lag_kmer, count_str in out_reader:
                    if lag_kmer in kmer_counts_check_rev[li]:
                        assert False
                    kmer_counts_check_rev[li][lag_kmer] = json.loads(count_str)

    # --- Compare. ---
    for li in range(max_lag):
        lag_kmers = set(list(kmer_counts_check[li].keys()) +
                        list(kmer_counts[li].keys()))
        for lag_kmer in lag_kmers:
            assert lag_kmer in kmer_counts[li]
            assert lag_kmer in kmer_counts_check[li]
            print(lag_kmer)
            for gi in range(n_groups):
                for j in range(len(alphabet)):
                    assert (kmer_counts[li][lag_kmer][gi][j] ==
                            kmer_counts_check[li][lag_kmer][gi][j])
        # Reverse.
        lag_kmers = set(list(kmer_counts_check_rev[li].keys()) +
                        list(kmer_counts_rev[li].keys()))
        for lag_kmer in lag_kmers:
            assert lag_kmer in kmer_counts_rev[li]
            assert lag_kmer in kmer_counts_check_rev[li]
            for gi in range(n_groups):
                for j in range(len(alphabet)):
                    assert (kmer_counts_rev[li][lag_kmer][gi][j] ==
                            kmer_counts_check_rev[li][lag_kmer][gi][j])


def test_check():
    # --- Compare using large scale check. ---
    args, *_ = setup_args()
    check_summarize.main(args)
    
    args, *_ = setup_args(pr=True)
    check_summarize.main(args)
