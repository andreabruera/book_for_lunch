import mne
import numpy
import os
import scipy

from mne import stats
from scipy import stats

f_one = 'overall_5000_ITGPT2medium_top_four_big_simple_decoding_rsa_False_only_verbs_False_book_fast_whole_trial_flattened_pairwise_all_no_reduction_avg.txt'
f_one = 'book_polysemy_5000_ITGPT2medium_top_four_big_simple_decoding_rsa_False_only_verbs_False_book_fast_whole_trial_flattened_pairwise_all_no_reduction_avg.txt'

f_two = 'overall_5000__simple_decoding_rsa_False_book_fast_whole_trial_flattened_pairwise_all_no_reduction_avg.txt'
f_two = 'book_polysemy_5000_book_fast_fasttext_simple_decoding_rsa_False_only_verbs_False_book_fast_whole_trial_flattened_pairwise_all_fisher_avg.txt'

with open(os.path.join('breakdown_decoding', f_one)) as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
val_one = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in lines}
with open(os.path.join('breakdown_decoding', f_two)) as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
val_two = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in lines}

scores = dict()
for k, v_one in val_one.items():
    v_two = val_two[k]
    sig = scipy.stats.wilcoxon(v_one, v_two, alternative='greater')[1]
    scores[k] = sig
corr_p = mne.stats.fdr_correction(list(scores.values()))[1]
corr_p = {k : round(v, 4) for k, v in zip(scores.keys(), corr_p)}
import pdb; pdb.set_trace()
