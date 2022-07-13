import numpy
import os
import scipy

from scipy import stats

with open('concreteness_book_single_words_en.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
    words = {l[0] : float(l[1]) for l in lines}
    stds = {l[0] : float(l[2]) for l in lines}

with open('book_fast_stimuli_ratings.tsv') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]

with open('../book_fast_single_stimuli_ratings.tsv', 'w') as o:
    for w in lines[0]:
        o.write('\t{}'.format(w))
    o.write('\n')
    for l in lines[1:]:
        conc = (words[l[0].split()[0]] + words[l[0].split()[1]]) / 2
        std = (stds[l[0].split()[0]] + stds[l[0].split()[1]]) / 2
        o.write('{}\t{}\t0.\t0.\t{}\t{}\t0.\t0.\n'.format(l[0], l[1], conc, std))

### Compute correlations
with open('book_fast_stimuli_ratings.tsv') as i:
    lines_one = [l.strip().split('\t') for l in i.readlines()]
with open('../book_fast_single_stimuli_ratings.tsv') as i:
    lines_two = [l.strip().split('\t') for l in i.readlines()]
one = {l[0] : float(l[4]) for l in lines_one[1:]}
two = {l[0] : float(l[4]) for l in lines_two[1:]}
one_values = [v for k, v in one.items()]
two_values = [two[k] for k in one.keys()]
print(scipy.stats.pearsonr(one_values, two_values))
