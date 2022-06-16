import argparse
import matplotlib
import mne
import numpy
import os
import re
import scipy

from matplotlib import pyplot
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--methodology', choices=[
                    'encoding', 'decoding', 
                    'rsa_encoding', 'rsa_decoding'],
                    required=True,
                    help = 'Encoding instead of decoding?')
args = parser.parse_args()

folder = os.path.join('results', 'vector_{}'.format(args.methodology))
possibilities = ['dot', 'verb', 'simple']
builder = ['_concrete', '_abstract']

whole_collector = dict()
### Collecting all results
for root, direc, filez in os.walk(folder):
    for fil in filez:
        if 'results' in fil:
            relevant_details = root.split('/')
            dataset = relevant_details[-1]
            if dataset not in whole_collector.keys():
                whole_collector[dataset] = dict()
            spatial_analysis = relevant_details[-2]
            if spatial_analysis not in whole_collector[dataset].keys():
                whole_collector[dataset][spatial_analysis] = dict()
            computational_model = relevant_details[-3]
            if computational_model not in whole_collector[dataset][spatial_analysis].keys():
                whole_collector[dataset][spatial_analysis][computational_model] = dict()
            features = relevant_details[-4]
            if features not in whole_collector[dataset][spatial_analysis][computational_model].keys():
                whole_collector[dataset][spatial_analysis][computational_model][features] = dict()
            senses = relevant_details[-5]
            if senses not in whole_collector[dataset][spatial_analysis][computational_model][features].keys():
                whole_collector[dataset][spatial_analysis][computational_model][features][senses] = dict()
            analysis = relevant_details[-6]
            if analysis not in whole_collector[dataset][spatial_analysis][computational_model][features][senses].keys():
                whole_collector[dataset][spatial_analysis][computational_model][features][senses][analysis] = dict()
            methodology = relevant_details[-7]
            '''
             'results',
             'vector_{}'.format(args.methodology),
             args.analysis, 
             'senses_{}'.format(args.senses),
             '{}_{}'.format(args.feature_selection, n_dims), 
             args.computational_model, 
             args.spatial_analysis,
             args.dataset, 
             )
            '''
            with open(os.path.join(root, fil)) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            assert methodology not in whole_collector[dataset][spatial_analysis][computational_model][features][senses][analysis].keys()
            whole_collector[dataset][spatial_analysis][computational_model][features][senses][analysis][methodology] = lines

### Plotting main violin plots

for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for computational_model, c_data in s_data.items():
            for features, f_data in c_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        for methodology, lines in a_data.items():

                            if 'book' in dataset:
                                if 'fedorenko' in spatial_analysis:
                                    cmap = 'BuGn_r'
                                elif 'general' in spatial_analysis:
                                    cmap = 'YlGnBu_r'
                                elif 'control' in spatial_analysis:
                                    cmap = 'YlOrBr_r'
                                else:
                                    cmap = 'RdPu_r'
                            else:
                                cmap = 'bone'

                            header = lines[0]
                            data = lines[1:]
                            collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                            current_collector = dict()
                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        'abstract_concrete',
                                        poss,
                                        '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                for h in keyz:
                                    try:
                                        current_collector[h] = collector[h]
                                    except KeyError:
                                        continue
                            ### Plotting
                            fig, ax = pyplot.subplots(figsize=(20, 9), constrained_layout=True)

                            ### Setting the title
                            if 'decoding' in methodology:
                                title = 'Decoding from fMRI to {}\nusing {} features from {} - {}'.format(
                                         computational_model, features.split('_')[1], analysis, spatial_analysis)
                            elif 'encoding' in methodology:
                                title = 'Encoding from {} to fMRI\nusing {} features from {} - {}'.format(
                                         computational_model, features.split('_')[1], analysis, spatial_analysis)
                            if 'True' in senses:
                                title =  '{}\naggregating phrases for word senses'.format(title)
                            title = title.replace('_', ' ')
                            ax.set_title(title, fontsize=20, pad=40)

                            ### Creating the output path
                            out_path  = os.path.join('plots', methodology, analysis, 
                                                     senses, features, spatial_analysis, 
                                                     dataset)
                            os.makedirs(out_path, exist_ok=True)
                            out_path = os.path.join(out_path, '{}.jpg'.format(computational_model)) 

                            positions = list(range(len(current_collector.keys())))

                            ### Plotting the ceiling, if available
                            try:
                                ceiling_data = whole_collector[dataset][spatial_analysis]['ceiling'][features][senses][methodology]
                                ceiling_pos = positions.copy()
                                ceiling_pos[0] = -.5
                                ceiling_pos[-1] = len(ceiling_pos) - .6
                                ax.fill_between(ceiling_pos, all_ceiling, [1. for i in all_ceiling],
                                                color='lightgray', alpha=0.3)
                            except KeyError:
                                print('No ceiling data available for {}, {}, {}, {}, {}, {}'.format(
                                       dataset, spatial_analysis, features, senses, methodology, analysis))
                            ps = list()
                            avgs = list()
                            mdns = list()
                            for pos, data in zip(positions, current_collector.items()):
                                data = numpy.array(data[1], dtype=numpy.float64)
                                ax.vlines(x=pos, ymin=min(data), ymax=max(data), color='lightgray')
                                parts = ax.violinplot(data, positions=[pos], showmeans=False, showextrema=False)
                                for pc in parts['bodies']:
                                    if pos > 1:
                                        colors = numpy.linspace(.1, .7, 6)
                                        pc.set_facecolor(pyplot.get_cmap(cmap)(colors[pos-2]))
                                    elif pos == 0:
                                        pc.set_facecolor('darkslategrey')
                                    elif pos == 1:
                                        pc.set_facecolor('slategrey')
                                    pc.set_edgecolor('lightgray')
                                    pc.set_alpha(.45)
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
                            ax.set_ylim(ymin=0.0, ymax=1.)
                            ax.set_xlim(xmin=-1.5, xmax=max(positions)+.5)
                            ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=0.5, alpha=0.5, color='darkgray', linestyle='dashdot')
                            ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], alpha=0.2, color='darkgray', linestyle='dashdot')
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
                            #x_ticks_final = [t.replace('Dot', 'Coercion').replace('Verb', 'Transparent').replace('Simple', 'Light Verb') for t in x_ticks_final]
                            x_ticks_final = [t.replace('Dot', '').replace('Verb', '').replace('Simple', '').strip() for t in x_ticks_final]
                            ax.set_xticklabels(x_ticks_final,
                                               #rotation=45,
                                               ha='center',
                                               fontsize=20,
                                               #fontweight='bold'
                                               )
                            ax.set_ylabel('Leave-2-out accuracy', fontsize=20, fontweight='bold', y=0.62, ha='center')

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
                            ax.set_yticks(numpy.linspace(0.1, 1, 10))
                            for i in range(2):
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

                            pyplot.savefig(os.path.join(out_path))
                            pyplot.clf()
                            pyplot.close()

### Plotting polysemy analyses
