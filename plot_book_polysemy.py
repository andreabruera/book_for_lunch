import matplotlib
import mne
import numpy
import os
import re
import scipy

from matplotlib import pyplot
from scipy import stats

folder = 'average_trials'
possibilities = ['book', 'catalogue', 'magazine', 'drawing']

collector = dict()
for root, direc, filez in os.walk(folder):
    for fil in filez:
        if 'results' in fil:
            with open(os.path.join(root, fil)) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            if list(set([len(k) for k in lines]))[0] > 1:
                header = lines[0]
                data = lines[1:]
                collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                current_collector = dict()
                for poss in possibilities:
                    #collector = {h : d for h, d in collector.items() if len(h.split('_'))<4}
                    #collector = {h : d for h, d in collector.items() if 'dot' in h}
                    #collector = {h : d for h, d in collector.items() if 'simple' in h}
                    #collector = {h : d for h, d in collector.items() if 'verb' in h}
                    #current_collector = {h : d for h, d in collector.items() if poss in h}
                    keyz = ['overall_accuracy',
                            poss,
                            ]
                    for h in keyz:
                        try:
                            current_collector[h] = collector[h]
                        except KeyError:
                            #print(h)
                            continue
                if len(current_collector.keys()) > 1:
                    if 'senses' in root:
                        all_ceiling = [0.703, 0.625, 0.625, 0.625, 0.562]
                        language_ceiling = [0.684, 0.562, 0.688, 0.688, 0.75]
                    else:
                        all_ceiling = [0.679, 0.618, 0.594, 0.562, 0.406]
                        language_ceiling = [0.684, 0.667, 0.594, 0.75, 0.469]

                    #if 'ceiling' in root and 'senses' in root:
                    #    import pdb; pdb.set_trace()
                    fold = root.split('/')[-6]

                    fig, ax = pyplot.subplots(figsize=(20, 9), constrained_layout=True)

                    positions = list(range(len(current_collector.keys())))
                    ceiling_pos = positions.copy()
                    ceiling_pos[0] = -.5
                    ceiling_pos[-1] = len(ceiling_pos) - .6
                    try:
                        if 'language_areas' in root:
                            ax.fill_between(ceiling_pos, language_ceiling, [1. for i in all_ceiling],
                                            color='lightgray', alpha=.3)
                        elif 'all' in root:
                            ax.fill_between(ceiling_pos, all_ceiling, [1. for i in all_ceiling],
                                            color='lightgray', alpha=0.3)
                    except ValueError:
                        continue
                    ps = list()
                    avgs = list()
                    mdns = list()
                    txt_collector = list()
                    for pos, data in zip(positions, current_collector.items()):
                        data = numpy.array(data[1], dtype=numpy.float64)
                        txt_collector.append(data)
                        ax.violinplot(data, positions=[pos], showmeans=False, showextrema=False)
                        avg = round(numpy.average(data), 3)
                        p = scipy.stats.wilcoxon(data-.5, alternative='greater')[1]
                        mdn = round(numpy.median(data), 3)
                        ps.append(p)
                        avgs.append(avg)
                        mdns.append(mdn)
                        conf_interval = stats.t.interval(0.95, len(data)-1, loc=numpy.mean(data), scale=stats.sem(data))
                        ax.vlines(pos, ymin=conf_interval[0], ymax=conf_interval[1], colors='dimgray', lw=4)
                        ax.scatter(x=pos, y=numpy.average(data), s=50, color= 'white', marker='H',zorder=3)
                    ps = mne.stats.fdr_correction(ps)[1]
                    for pos, p, avg, mdn in zip(positions, ps, avgs, mdns):
                        ax.text(x=pos, y=-.3+(0.17), s=avg, fontsize=15,
                                ha='center', va='center')
                        if p<0.05:
                            weight = 'bold'
                        else:
                            weight = 'normal'
                        ax.text(x=pos, y=-.3+(0.1), s=round(p, 4), fontsize=15,
                                ha='center', va='center', fontweight=weight)
                        '''
                        ax.text(x=pos,
                                #+0.2,
                                #y=avg,
                                y=0.2,
                                s='avg: {}\n'
                                  #'mdn: {}\n'
                                  'p (FDR): {}'.format(avg, round(p, 4)),
                                fontsize=15,
                                #ha='left',
                                ha='center',
                                va= 'center')
                        '''
                    ax.set_ylim(ymin=-.3, ymax=1.)
                    ax.set_xlim(xmin=-1.5, xmax=max(positions)+.5)
                    ax.hlines(xmin=-.5, xmax=max(positions)+0.4, y=0.5, alpha=0.5, color='darkgray', linestyle='dashdot')
                    ax.set_xticks(positions)
                    x_ticks = [t.replace('concrete', 'object')\
                                .replace('abstract', 'information')\
                                .replace('_', ' ').capitalize() for t in current_collector.keys()]
                    x_ticks = [x_ticks[0]] + ['{}\n(Info vs Obj)'.format(l) for l in x_ticks[1:]]
                    '''
                    x_ticks_final = list()
                    for x_t in x_ticks:
                        length = len(x_t.split())
                        if length == 1 or 'all' in x_t:
                            x_ticks_final.append('Aggregate\n{}'.format(x_t).replace('accuracy', ''))
                        elif length == 2:
                            x_ticks_final.append(x_t.replace(' ', ' vs '))
                        else:
                            final_x = [x_t.split()[i] for i in [0,1,3]]
                            final_x.insert(1, ' ')
                            final_x.insert(3, 'vs')
                            x_ticks_final.append(' '.join(final_x))

                    x_ticks_final = [re.sub('\s+', ' ', t) for t in x_ticks_final]
                    x_ticks_final = [t.replace(' ', '\n') for t in x_ticks_final]
                    '''
                    ax.set_xticklabels(x_ticks,
                                       #rotation=45,
                                       ha='center',
                                       fontsize=20,
                                       #fontweight='bold'
                                       )
                    ax.set_ylabel('Leave-2-out accuracy', fontsize=20, fontweight='bold', y=0.62, ha='center')
                    details = root.split('/')
                    details = ', '.join(['Dataset: {}'.format(details[-6]),
                                         '{} Voxels from {}'.format(details[-7].split('_')[0], details[-3]),
                                         'Model: {}'.format(' '.join(details[-7].split('_')[1:4]))])
                    title = 'Decoding from fMRI to word vectors\n{}'.format(details)
                    if 'senses' in root:
                        title = title.replace('word vectors', 'senses word vectors')
                    if 'ceiling' in root:
                        title = title.replace('word vectors', 'fMRI (noise ceiling)')
                    if 'rsa_True' in root:
                        title = title.replace('Decoding', 'RSA Decoding')
                    ratings = re.findall(r'familiarity|vector_familiarity|concreteness|vector_concreteness|imageability|vector_imageability|frequency|vector_frequency|pairwise_word_vectors', root)
                    if len(ratings) > 0:
                        title = title.replace('word vectors', ratings[0])
                    ax.set_title(title, fontsize=20, pad=40)

                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params('y', labelsize=15, length=0)
                    ax.tick_params('y', length=0)
                    ax.text(x=-.9, y=.8, s='Noise\nceiling', fontsize=18,
                            ha='center', va='center', fontweight='bold')
                    ax.text(x=-.9, y=0.5, s='Chance level', fontsize=18,
                            ha='center', va='center', fontweight='bold')
                    ax.text(x=-.9, y=-.3+(0.17), s='Average', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=-.9, y=-.3+(.1-0.01), s='P-value\n(FDR)', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.hlines(y=-.16, xmin=-.5, xmax=max(positions)+.4,
                              alpha=.4)
                    for i in range(2):
                        ax.get_yticklabels()[0].set_visible(False)
                    '''
                    ax.get_yticklabels()[-1].set_visible(False)
                    '''
                    ax.vlines(x=[.5], ymin=0.1, ymax=0.9, alpha=0.4)

                    '''
                    ax.text(x=.5, y=0.9, s='Overall', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=2.5, y=0.9, s='Coercion', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=4.5, y=0.9, s='Transparent', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=6.5, y=0.9, s='Light Verbs', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    '''

                    if 'encoding' not in root:
                        out = 'breakdown_decoding'
                        os.makedirs(out, exist_ok=True)
                    else:
                        out = 'breakdown_encoding'
                        os.makedirs(out, exist_ok=True)
                    pyplot.savefig(os.path.join(out,
                        'book_polysemy_{}.jpg'.format(root.replace('/', '_')[15:])))
                    pyplot.clf()
                    pyplot.close()

                    ### Write to txtos.path.join(out,
                    txt_out = os.path.join(out,
                            'book_polysemy_{}.txt'.format(root.replace('/', '_')[15:]))
                    txt_collector = {k : v for k, v in zip(x_ticks, txt_collector)}
                    with open(txt_out, 'w') as o:
                        for k, v in txt_collector.items():
                            o.write('{}\t'.format(k.replace('\n', ' ')))
                            for val in v:
                                o.write('{}\t'.format(val))
                            o.write('\n')
