from bear_model import summarize

import os
from Bio import Seq
import argparse
from collections import defaultdict
import csv
import json


def get_input_info(args):
    input_info = []
    with open(args.file, 'r', newline='') as file_list:
        file_list_reader = csv.reader(file_list)
        for indiv_file, group, file_type in file_list_reader:
            input_info.append((indiv_file, int(group), file_type))
    return input_info


def get_output_info(args):
    # Dummy run of stages 1 and 2.
    args.s3 = False
    n_groups, kmc_runs = summarize.preprocess_seq_files(
        args.file, args.l, args.r, args.pr, args.out_prefix,
        args.p, args.t, args.mk, not args.s3)
    total_size = summarize.run_kmc(kmc_runs, not args.s3)
    # Compute number of output files.
    n_bin_bits = summarize.compute_n_bin_bits(
                    total_size, n_groups, args.mf, args.ls)
    n_bins = 2**n_bin_bits
    # Construct output file names.
    path = '/'.join(args.out_prefix.split('/')[:-1])
    def get_start_token(lag):
        start_token = '{}_lag_{}_file_'.format(args.out_prefix, lag)
        return start_token.split('/')[-1]
    output_info = [[os.path.join(path, file) for file in os.listdir(path)
                    if file.startswith(get_start_token(li+1))]
                   for li in range(args.l)]
    return output_info


def extract_input_counts(args):
    if args.s3_o is None:
        compare_kmer = lambda k, b: True
    else:
        compare_kmer = lambda k, b: (b != ']' and '[' not in k)

    input_info = get_input_info(args)
    groups = [elem[1] for elem in input_info]
    n_groups = max(groups) + 1
    alphabet = summarize.alphabet
    kmer_counts = [defaultdict(lambda: [[0 for j in range(len(alphabet))]
                                        for i in range(n_groups)])
                   for li in range(args.l)]
    for li in range(args.l):
        lag = li + 1
        for fi in range(len(input_info)):
            indiv_file, group, file_type = input_info[fi]
            for si, elem in enumerate(summarize.load_input(
                                open(indiv_file, 'r'), file_type)):
                seq = elem[1]
                full_seq = '['*lag + seq + ']'
                for j in range(lag, len(full_seq)):
                    lag_kmer = full_seq[(j-lag):j]
                    next_letter = full_seq[j]
                    if compare_kmer(lag_kmer, next_letter):
                        kmer_counts[li][lag_kmer][group][
                                alphabet[next_letter]] += 1
                if args.r:
                    seq = Seq.reverse_complement(seq)
                    full_seq = '['*lag + seq + ']'
                    for j in range(lag, len(full_seq)):
                        lag_kmer = full_seq[(j-lag):j]
                        next_letter = full_seq[j]
                        if compare_kmer(lag_kmer, next_letter):
                            kmer_counts[li][lag_kmer][group][
                                    alphabet[next_letter]] += 1
    return kmer_counts


def extract_output_counts(args):
    output_info = get_output_info(args)
    kmer_counts_check = [dict() for li in range(args.l)]
    for li in range(args.l):
        for bi in range(len(output_info[li])):
            out_file = output_info[li][bi]
            with open(out_file, 'r', newline='') as out_counts_file:
                out_reader = csv.reader(out_counts_file, delimiter='\t')
                for lag_kmer, count_str in out_reader:
                    if lag_kmer in kmer_counts_check[li]:
                        assert False
                    kmer_counts_check[li][lag_kmer] = json.loads(count_str)

    return kmer_counts_check


def compare_counts(kmer_counts, kmer_counts_check, args):
    for li in range(args.l):
        lag_kmers = set(list(kmer_counts_check[li].keys()) +
                        list(kmer_counts[li].keys()))
        for lag_kmer in lag_kmers:
            assert lag_kmer in kmer_counts[li]
            assert lag_kmer in kmer_counts_check[li]
            for gi in range(max([
                        len(kmer_counts[li][lag_kmer]),
                        len(kmer_counts_check[li][lag_kmer])])):
                for j in range(len(summarize.alphabet)):
                    # print(li, lag_kmer, gi, j)
                    # print(kmer_counts[li][lag_kmer][gi][j])
                    # print(kmer_counts_check[li][lag_kmer][gi][j])
                    assert (kmer_counts[li][lag_kmer][gi][j] ==
                            kmer_counts_check[li][lag_kmer][gi][j])


def main(args):
    store_args_r = args.r
    args.r = False
    kmer_counts = extract_input_counts(args)
    kmer_counts_check = extract_output_counts(args)
    compare_counts(kmer_counts, kmer_counts_check, args)

    # Handle reverse case.
    if store_args_r:
        args.r = True
        args.out_prefix += '_rev'
        kmer_counts = extract_input_counts(args)
        kmer_counts_check = extract_output_counts(args)
        compare_counts(kmer_counts, kmer_counts_check, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Preprocess for collapsed EAR training.")
    parser.add_argument('file',
                        help=('Input file. csv of individual files and their'
                              + ' group number.'))
    parser.add_argument('out_prefix', help='Prefix for output files.')
    parser.add_argument('-l', default=10, type=int,
                        help='Maximum lag of EAR model.')
    parser.add_argument('-mk', default=12, type=int,
                        help='Maximum memory available to KMC (Gb)')
    parser.add_argument('-mf', default=0.1, type=float,
                        help='Maximum memory of final dataset chunks (Gb).')
    parser.add_argument('-p', default='.',
                        help=('Path to folder with kmc scripts' +
                              '(kmc and kmc_dump).'))
    parser.add_argument('-r', action='store_true', default=False,
                        help='Also compute reverse direction.')
    parser.add_argument('-t', default='tmp',
                        help='Temporary directory for KMC.')
    parser.add_argument('-s12', action='store_true', default=False,
                        help='Only run stages 1 and 2.')
    parser.add_argument('-s3', default=True,
                        help='Only run stage 3.')
    parser.add_argument('-num', default=10, type=int,
                        help='Number of random kmers to check.')
    args = parser.parse_args()
    main(args)
