import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import xlogy
from Bio import SeqIO
from Bio import Seq
from . import core
from . import get_var_probs

def assemble_no_ends(seqs_fa_file, lengths_to_gen, num_to_gen, bear_path, kmc_path,
                     h=None, reverse=True, save_folder=None):
    """Generate sequences from a seed using a BEAR model without the possibility of the sequence stopping.
    Note: this is not the same thing as conditioning on not stopping, and is best seen as generating to a limited length
    in a model where stops are uncommon.
    For each generated sequence, an AR model is drawn from BEAR and the sequence is generated according to that model.
    
    Parameters
    ----------
    seqs_fa_file : str
        file name of fasta containing starting sequences.
    lengths_to_gen : numpy array
        Shape [len(seqs), 2] for how much to generate forward and backwards.
    num_to_gen : int
        Number of sequences ot generate for each seed sequence.
    bear_path : str
    kmc_path : str
        Include .res !
    h : float, defualt=None
        Optinally change learned h in BEAR
    reverse : add reverse counts to KMC counter (BEAR is still forward).
    save_folder : str, default=None
        Output sequences into fasta file seqs.fa, and plot site-wise entropies of generated sequences.
        
    Returns
    -------
    gen_seqs : numpy array
        Array of shape [len(seqs), num_to_gen]
    sw_ent : list
        List of length len(seqs) where each entry is the length of the coresponding generated sequences.
        Each entry contains the sitewise entropy of the num_to_gen generated sequences.
        """
    lag, alphabet_name, h_bear, ar_func, data = get_var_probs.load_bear(bear_path)
    if h is None:
        h = h_bear
    h = np.array([h])
    vans = []
    get_map = False
    train_col = 0
    alphabet = core.alphabets_en[alphabet_name][:-1]
    alphabet_size = len(alphabet)
    #load kmc
    counter = get_var_probs.make_kmc_genome_counter(kmc_path, lag, reverse=reverse, no_end=True)
    
    # load seqs
    fwd_seqs = np.array([str(seq.seq) for seq in SeqIO.parse(seqs_fa_file, 'fasta')])
    # repeat seqs
    fwd_seqs = np.repeat(fwd_seqs, num_to_gen * np.ones(len(fwd_seqs), dtype=int))
    lengths_to_gen_rep = np.repeat(np.array(lengths_to_gen).T, num_to_gen * np.ones(2*len(lengths_to_gen), dtype=int)).reshape([2, -1])
    # make reverse
    rev_seqs = np.array([Seq.reverse_complement(seq) for seq in fwd_seqs])
    
    flanks = []
    for seqs, length_to_gen in zip([rev_seqs, fwd_seqs], lengths_to_gen_rep):
        # instantiate new sequences
        new_seq = (len(seqs)) * ['']
        new_seq_lens = np.zeros(len(seqs))
        inds = np.squeeze(np.argwhere(new_seq_lens < length_to_gen)) 
        
        # get kmers
        end_kmers = np.array([seq[-lag:] for seq in seqs[inds]])
        all_kmers = []#np.array(list(set(end_kmers))).astype(str)
        all_pdf_start = True

        while len(inds) > 0:
            # TODO: pdf needs to change with inds!
            #new kmers should 
            new_kmers = np.unique(end_kmers[~np.isin(end_kmers, all_kmers)])
            all_kmers = np.r_[all_kmers, new_kmers]
            # get counts
            counts = counter(new_kmers)[:, None, :]
            # pake a new pdf
            mc_samples = len(end_kmers)
            if len(new_kmers) > 0:
                new_kmers_pdf = get_var_probs.get_pdf(new_kmers, counts, h, ar_func, mc_samples,
                                                      vans, train_col, alphabet_name, get_map, get_df=True)
                # concatenate pdfs
                if not all_pdf_start:
                    all_pdf = pd.concat([all_pdf, new_kmers_pdf])
                else:
                    all_pdf = new_kmers_pdf
                    all_pdf_start = False
            # make pdf function
            pdf_func = get_var_probs.df_to_func(all_pdf, 1, mc_samples, summed=False)
            # query into pdf
            end_kp1mers = get_var_probs.cross_str_arrays(end_kmers, alphabet, exch='X')
            trans_log_probs = pdf_func(end_kp1mers).reshape([len(end_kmers), len(alphabet), len(end_kmers)])
            trans_log_probs = np.einsum('ijk,ik->ij', trans_log_probs, np.eye(len(end_kmers)))
            # transition
            new_letters = alphabet[np.argmax(np.random.gumbel(size=np.shape(trans_log_probs)) + trans_log_probs, axis=-1)]
            # add to end_kmers and drop finished columns in all_pdf
            end_kmers = np.array([end[1:]+l for end, l, ind in zip(end_kmers, new_letters, inds)
                                  if new_seq_lens[ind] + 1 < length_to_gen[ind]])
            all_pdf = all_pdf.iloc[:, [new_seq_lens[ind] + 1 < length_to_gen[ind] for ind in inds]]
            all_pdf.columns = np.arange(len(all_pdf.columns)) # drop the column names
            # add to new seq
            for j, ind in enumerate(inds):
                new_seq[ind] = new_seq[ind] + new_letters[j]
                new_seq_lens[ind] += 1
            inds = np.squeeze(np.argwhere(new_seq_lens < length_to_gen))
        flanks.append(new_seq)
    
    gen_seqs = [Seq.reverse_complement(left_seq) + seed_seq + right_seq
                for left_seq, right_seq, seed_seq in zip(flanks[0], flanks[1], seqs)]
    gen_seqs = np.array(gen_seqs).reshape([-1, num_to_gen])
    sw_ent = []
    for seqs in gen_seqs:
        gen_seqs_probs = np.average(core.tf_one_hot(seqs, alphabet_name).numpy(), axis=0)
        sw_ent.append(-np.sum(xlogy(gen_seqs_probs, gen_seqs_probs), axis=-1))
    
    if save_folder is not None:
        os.system('mkdir -p ' + save_folder)
        with open(os.path.join(save_folder, 'seqs.fa'), 'w+') as f:
            for i, seqs in enumerate(gen_seqs):
                for j, seq in enumerate(seqs):
                    f.write('>seq{}_rep{}\n{}\n'.format(i, j, seq))
                   
        plt.figure(figsize=[10, 5])
        plt.xlabel("entropy", fontsize=15)
        plt.ylabel("position", fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        xlim = [0, 0]
        for ent, length_to_gen in zip(sw_ent, lengths_to_gen):
            xs = np.arange(len(ent)) + - length_to_gen[0]
            xlim[1] = np.max([xlim[1], np.max(xs)])
            xlim[0] = np.min([xlim[0], np.min(xs)])
            plt.plot(xs, ent, color='blue', linewidth=1, alpha=0.1)
        plt.plot(xlim, np.log(alphabet_size) * np.ones(2), color='black', linewidth=2)
        plt.xlim(xlim)
        ylim = plt.ylim()
        plt.ylim([0, ylim[1]])
        plt.savefig(os.path.join(save_folder, 'entropy.png'), dpi=200)
        plt.xlim([-10, 0])
        plt.savefig(os.path.join(save_folder, 'entropy_zoom.png'), dpi=200)
        plt.show()
        plt.close()
    return gen_seqs, sw_ent
