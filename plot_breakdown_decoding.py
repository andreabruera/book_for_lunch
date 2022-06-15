import matplotlib
import mne
import numpy
import os
import re
import scipy

from matplotlib import pyplot
from scipy import stats

folder = 'average_trials'
possibilities = ['dot', 'verb', 'simple']
builder = ['_concrete', '_abstract']

whole_collector = dict()
for root, direc, filez in os.walk(folder):
    if 'book' in root:
        ### Averaging senses
        if 'senses' in root:
            if 'fedorenko' in root:
                all_ceiling = [0.685, 0.683, 0.743, 0.723, 0.633, 0.619, 0.722, 0.744]
            elif 'general' in root:
                all_ceiling = [0.662, 0.665, 0.71, 0.691, 0.619, 0.613, 0.708, 0.728]
            elif 'control' in root:
                all_ceiling = []
            else:
                all_ceiling = []
        ### Full set of phrases
        else:
            if 'fedorenko' in root:
                all_ceiling = [0.701, 0.697, 0.685, 0.675, 0.666, 0.651, 0.75, 0.753]
            elif 'general' in root:
                all_ceiling = [0.674, 0.67, 0.638, 0.626, 0.669, 0.657, 0.712, 0.709]
            elif 'control' in root:
                all_ceiling = [0.659, 0.657, 0.643, 0.625, 0.643, 0.642, 0.696, 0.698]
            else:
                all_ceiling = [0.679, 0.678, 0.652, 0.636, 0.696, 0.704, 0.694, 0.695]
                language_ceiling = [0.684, 0.685, 0.67, 0.672, 0.682, 0.684, 0.704, 0.719]
    elif 'lunch' in root:
        all_ceiling = [0.702, 0.699, 0.681, 0.685, 0.715, 0.722, 0.687, 0.685]
        language_ceiling = [0.712, 0.712, 0.705, 0.724, 0.722, 0.735, 0.706, 0.698]
    for fil in filez:
        ### Considering only cases when using the top 4 layers
        if 'results' in fil:
            with open(os.path.join(root, fil)) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            if list(set([len(k) for k in lines]))[0] > 1:

                txt_collector = list()

                details = root.split('/')
                if 1 == 1:
                    #if 'vector' in details[-7]:
                    #print(details)
                    details = ', '.join(['Dataset: {}'.format(details[-6]),
                                         '{} Voxels from {}'.format(details[-7].split('_')[0], details[-3]),
                                         'Model: {}'.format(' '.join(details[-7].split('_')[1:4]))])
                    model_details = re.sub('\W', '_', root)

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
                                #'{}{}'.format(poss, builder[0]),
                                #'{}{}'.format(poss, builder[1]),
                                'abstract_concrete',
                                poss,
                                '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                        for h in keyz:
                            try:
                                current_collector[h] = collector[h]
                            except KeyError:
                                continue
                    fold = root.split('/')[-6]
                    whole_collector[model_details] = list()

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
                    for pos, data in zip(positions, current_collector.items()):
                        data = numpy.array(data[1], dtype=numpy.float64)
                        whole_collector[model_details].append([data])
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
                        ax.text(x=pos, y=0.17, s=avg, fontsize=15,
                                ha='center', va='center')
                        if p<0.05:
                            weight = 'bold'
                        else:
                            weight = 'normal'
                        ax.text(x=pos, y=0.1, s=round(p, 4), fontsize=15,
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
                    #ax.set_ylim(ymin=0.0, ymax=.9)
                    ax.set_ylim(ymin=0.0, ymax=1.)
                    ax.set_xlim(xmin=-1.5, xmax=max(positions)+.5)
                    ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=0.5, alpha=0.5, color='darkgray', linestyle='dashdot')
                    ax.set_xticks(positions)
                    x_ticks = [t.replace('concrete', 'object')\
                                .replace('abstract', 'information')\
                                .replace('_', ' ').capitalize() for t in current_collector.keys()]
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
                    for x_i, x in enumerate(x_ticks_final):
                        whole_collector[model_details][x_i].append(x)
                    ax.set_xticklabels(x_ticks_final,
                                       #rotation=45,
                                       ha='center',
                                       fontsize=20,
                                       #fontweight='bold'
                                       )
                    ax.set_ylabel('Leave-2-out accuracy', fontsize=20, fontweight='bold', y=0.62, ha='center')
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
                    ax.text(x=-.9, y=0.17, s='Average', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=-.9, y=.1-0.01, s='P-value\n(FDR)', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.hlines(y=.14, xmin=-.4, xmax=max(positions)+.2,
                              alpha=.4)
                    ### < 0.6 tick labels
                    for i in range(3):
                        ax.get_yticklabels()[0].set_visible(False)
                    ### 1.0 tick label
                    #ax.get_yticklabels()[-1].set_visible(False)
                    ax.vlines(x=[1.5, 3.5, 5.5], ymin=0.1, ymax=0.9, alpha=0.4)

                    ax.text(x=.5, y=0.95, s='Overall', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=2.5, y=0.95, s='Coercion', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=4.5, y=0.95, s='Transparent', fontsize=18,
                            fontweight='bold', ha='center', va='center')
                    ax.text(x=6.5, y=0.95, s='Light Verbs', fontsize=18,
                            fontweight='bold', ha='center', va='center')

                    if 'encoding' not in root:
                        out = 'breakdown_decoding'
                        os.makedirs(out, exist_ok=True)
                    else:
                        out = 'breakdown_encoding'
                        os.makedirs(out, exist_ok=True)
                    pyplot.savefig(os.path.join(out,
                        'overall_{}.jpg'.format(root.replace('/', '_')[15:])))
                    pyplot.clf()
                    pyplot.close()
                    ### Write to txt
                    txt_out = os.path.join(out,
                            'overall_{}.txt'.format(root.replace('/', '_')[15:]))
                    txt_collector = {k : v for k, v in zip(x_ticks_final, txt_collector)}
                    with open(txt_out, 'w') as o:
                        for k, v in txt_collector.items():
                            o.write('{}\t'.format(k.replace('\n', ' ')))
                            for val in v:
                                o.write('{}\t'.format(val))
                            o.write('\n')
import pdb; pdb.set_trace()
