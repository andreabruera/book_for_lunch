import itertools
import numpy
import os
import scipy

from scipy import stats

names = list()
all_vecs = list()
for model in ['concreteness', 'concretenesssingle']:
    with open('resources/{}_model.vectors'.format(model)) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    vecs = {l[0] : numpy.array(l[1:], dtype=numpy.float32) for l in lines}
    conc_vecs = [v[1] for v in sorted(vecs.items(), key=lambda item : item[0])]
    assert len(conc_vecs) == 42
    corrs = list()
    for v_i, v in enumerate(conc_vecs):
        for v_i_two, v_two in enumerate(conc_vecs):
            corr = scipy.stats.pearsonr(v, v_two)[0]
            corrs.append(corr)
    all_vecs.append(corrs)
    names.append(model)

models = ['gpt2', 'fasttext']
for model in models:
    comp_vecs = dict()
    if model == 'gpt2':
        folder = os.path.join('resources', 'ITGPT2medium_top_four_span_average')
    else:
        folder = os.path.join('resources', 'fasttext_concatenated')
    for f in os.listdir(folder):
        with open(os.path.join(folder, f)) as i:
            lines = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float32)
        phrase = f.replace('_', ' ').split('.')[0]
        comp_vecs[phrase] = numpy.average(lines, axis=0)
    comp_vecs = {k : v for k, v in comp_vecs.items() if k in vecs.keys()}
    comp_vecs = [v[1] for v in sorted(comp_vecs.items(), key=lambda item : item[0])]
    assert len(comp_vecs) == 42
    corrs = list()
    for v_i, v in enumerate(comp_vecs):
        for v_i_two, v_two in enumerate(comp_vecs):
            corr = scipy.stats.pearsonr(v, v_two)[0]
            corrs.append(corr)
    all_vecs.append(corrs)
    names.append(model)

for c in itertools.combinations(range(len(names)), 2):
    corr = scipy.stats.pearsonr(all_vecs[c[0]], all_vecs[c[1]])
    print((names[c[0]], names[c[1]]))
    print(corr)
