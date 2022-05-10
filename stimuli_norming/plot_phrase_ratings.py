import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats

out = 'phrase_rating_plots'
os.makedirs(out, exist_ok=True)

for noun_type in ['book', 'lunch']:

    f = '{}_fast_stimuli.tsv'.format(noun_type)
    with open(f) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    exp = {tuple(k[0].replace("'", ' ').split()) : k[1] for k in lines}
    exp = {(k[0], k[2]) if len(k)==3 else k : v for k, v in exp.items()}

    f = '{}_fast_stimuli_ratings.tsv'.format(noun_type)
    with open(f) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    headings = lines[0]
    data = lines[1:]
#data = list()
    missing = 0
    '''
    for d_i, d in enumerate(all_data):

        line = d[0].replace("'", ' ').split()
        line = tuple([line[0], line[-1]]) if len(line)==3 else tuple(line)


        if line not in exp.keys():
            missing += 1
            print(line)
        else:
            data.append(d)
    '''
    header = {h : numpy.array([d[h_i] for d in data], dtype=numpy.float64) for h_i, h in enumerate(headings) if 'avg' in h}
    mapper = {d[0] : d[1] for d in data}
    colors = {'dot_concrete' : 'orange', 'dot_abstract' : 'yellow',
              'simple_concrete' : 'black', 'simple_abstract' : 'darkgray',
              'verb_concrete' : 'blue', 'verb_abstract' : 'lightblue'}

### All ratings
    fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
    for i, val in enumerate(header.values()):
        ax.violinplot(val/7, positions=[i], showmeans=True, showextrema=True)
        avg = round(numpy.average(val)/7, 3)
        #print(val.shape)
        mdn = round(numpy.median(val)/7, 3)
        ax.text(x=i+0.2, y=avg, s='avg: {}\nmdn: {}'.format(avg, mdn), ha='left', va='center')

    ax.set_xticks(range(len(header)))
    ax.set_xticklabels(header, ha='right', rotation=45)
    ax.set_ylim(ymin=0., ymax=1.1)
    ax.set_title('Ratings for {}-noun_type polysemy cases'.format(noun_type))

    pyplot.savefig(os.path.join(out, '{}_ratings.jpg'.format(noun_type)))
    pyplot.clf()
    pyplot.close()

### Concreteness
    header = {h : [d[h_i] for d in data] for h_i, h in enumerate(headings) if 'std' not in h}
    fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
    data = zip(header['phrase'], header['concreteness_avg'])
    data = sorted(data, key=lambda item : item[1])
    for i, phr_val in enumerate(data):
        ax.scatter(x=i, y=float(phr_val[1])/7, color=colors[mapper[phr_val[0]]], s=100.)
        ax.text(x=i, y=(float(phr_val[1])/7)-0.2, s=phr_val[0], rotation=90, va='top', ha='center', fontsize=15)
    ax.scatter(x=0., y=.71, color='gray', s=100.)
    ax.scatter(x=1., y=.71, color='black', s=100.)
    ax.text(x=2., y=.71, s='simple V + simple N', va='center', ha='left', fontsize=15)
    ax.scatter(x=0, y=.74, color='lightblue', s=100.)
    ax.scatter(x=1., y=.74, color='blue', s=100.)
    ax.text(x=2., y=.74, s='coercing V + simple N', va='center', ha='left', fontsize=15)
    ax.scatter(x=0., y=.765, color='yellow', s=100.)
    ax.scatter(x=1., y=.765, color='orange', s=100.)
    ax.text(x=2., y=.765, s='coercing V + dot N', va='center', ha='left', fontsize=15)
    ax.text(x=0., y=.785, s='information', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
    ax.text(x=1., y=.785, s='object', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
    ax.tick_params(axis='y', which='major', labelsize=15)

    ax.hlines(xmin=0., xmax=len(data), y=[0.,.2,.4,.6,.8,1.], alpha=0.3, color='darkgray', linestyles='dashed')
    ax.set_ylabel('Average concreteness rating', fontsize=20)
    ax.set_ylim(top=1.1, bottom=0.)
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Concreteness ratings for {}-noun_type polysemy cases\navg number of raters per item: 45'.format(noun_type), fontsize=20.)

    pyplot.savefig(os.path.join(out, 'concreteness_{}_ratings.jpg'.format(noun_type)))
    pyplot.clf()
    pyplot.close()
    conc_data = data.copy()

### Imageability
    #header = {h : [d[h_i] for d in data] for h_i, h in enumerate(headings) if 'std' not in h}
    fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
    data = zip(header['phrase'], header['imageability_avg'])
    data = sorted(data, key=lambda item : item[1])
    for i, phr_val in enumerate(data):
        ax.scatter(x=i, y=float(phr_val[1])/7, color=colors[mapper[phr_val[0]]], s=100.)
        ax.text(x=i, y=(float(phr_val[1])/7)-0.2, s=phr_val[0], rotation=90, va='top', ha='center', fontsize=15)
    ax.scatter(x=0., y=.71, color='gray', s=100.)
    ax.scatter(x=1., y=.71, color='black', s=100.)
    ax.text(x=2., y=.71, s='simple V + simple N', va='center', ha='left', fontsize=15)
    ax.scatter(x=0, y=.74, color='lightblue', s=100.)
    ax.scatter(x=1., y=.74, color='blue', s=100.)
    ax.text(x=2., y=.74, s='coercing V + simple N', va='center', ha='left', fontsize=15)
    ax.scatter(x=0., y=.765, color='yellow', s=100.)
    ax.scatter(x=1., y=.765, color='orange', s=100.)
    ax.text(x=2., y=.765, s='coercing V + dot N', va='center', ha='left', fontsize=15)
    ax.text(x=0., y=.785, s='information', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
    ax.text(x=1., y=.785, s='object', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
    ax.tick_params(axis='y', which='major', labelsize=15)

    ax.hlines(xmin=0., xmax=len(data), y=[0.,.2,.4,.6,.8,1.], alpha=0.3, color='darkgray', linestyles='dashed')
    ax.set_ylabel('Average imageability rating', fontsize=20)
    ax.set_ylim(top=1.1, bottom=0.)
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Imageability ratings for {}-noun_type polysemy cases\navg number of raters per item: 45'.format(noun_type), fontsize=20.)

    pyplot.savefig(os.path.join(out, 'imageability_{}_ratings.jpg'.format(noun_type)))
    pyplot.clf()
    pyplot.close()
    conc_data = sorted(conc_data, key=lambda item:item[0])
    data = sorted(data, key=lambda item:item[0])
    data = zip(header['phrase'], header['concreteness_avg'])
    import pdb; pdb.set_trace()
    print(scipy.stats.spearmanr([float(d[1]) for d in data], [float(d[1]) for d in conc_data]))
