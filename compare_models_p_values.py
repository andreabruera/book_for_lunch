import argparse
import itertools
import matplotlib
import mne
import numpy
import os
import re
import scipy
import statsmodels
import warnings

from matplotlib import pyplot
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--methodology', choices=[
                    'encoding', 'decoding', 
                    'rsa_encoding', 'rsa_decoding'],
                    required=True,
                    help = 'Encoding instead of decoding?')
args = parser.parse_args()

scipy.special.seterr(all='raise')
warnings.simplefilter('error')

folder = os.path.join('results', 'full_results_vector_{}'.format(args.methodology))

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
            methodology = relevant_details[-7]
            if methodology not in whole_collector[dataset][spatial_analysis].keys():
                whole_collector[dataset][spatial_analysis][methodology] = dict()
            features = relevant_details[-4]
            if features not in whole_collector[dataset][spatial_analysis][methodology].keys():
                whole_collector[dataset][spatial_analysis][methodology][features] = dict()
            senses = relevant_details[-5]
            if senses not in whole_collector[dataset][spatial_analysis][methodology][features].keys():
                whole_collector[dataset][spatial_analysis][methodology][features][senses] = dict()
            analysis = relevant_details[-6]
            if analysis not in whole_collector[dataset][spatial_analysis][methodology][features][senses].keys():
                whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis] = dict()
            computational_model = relevant_details[-3]
            with open(os.path.join(root, fil)) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            if computational_model not in whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis].keys():
                whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis][computational_model] = dict()
            results = {tuple(sorted((l[0], l[3]))) : int(l[-1]) for l in lines[1:]}
            ### Reorganize lines
            stimuli_mapper = {(l[0], l[3]) : [l[1], list(set([l[2], l[5]]))] for l in lines[1:] if l[1]==l[4]}
            cases = {'overall' : [tuple(sorted(k)) for k in results.keys()], 
                     'coercion' : [tuple(sorted(k)) for k, v in stimuli_mapper.items() if v[0]=='coercion'],
                     'transparent' : [tuple(sorted(k)) for k, v in stimuli_mapper.items() if v[0]=='transparent'],
                     'light verbs' : [tuple(sorted(k)) for k, v in stimuli_mapper.items() if v[0]=='light'],
                     }
            whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis][computational_model][fil.split('.')[0]] = results
mods = list()
ps = list()
### Comparisons between models
for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        models_comb = itertools.combinations(a_data.keys(), 2)
                        for model_one, model_two in models_comb:
                            res_one = a_data[model_one]
                            res_two = a_data[model_two]
                            ### Comparisons to be made: overall, coercion, transparent, light verbs
                            ### Overall
                            for case, combs in cases.items():
                                cont_table = numpy.zeros((2, 2))
                                for s in range(1, 17):
                                    all_subs_one = [res_one['sub-{:02}'.format(s)][c] for c in combs]
                                    all_subs_two = [res_two['sub-{:02}'.format(s)][c] for c in combs]
                                    for o, t in zip(all_subs_one, all_subs_two):
                                        cont_table[o, t] += 1
                                p_value = statsmodels.stats.contingency_tables.mcnemar(cont_table)
                                mods.append([model_one, model_two, spatial_analysis, case]) 
                                ps.append(vars(p_value)['pvalue'])
corr_ps = mne.stats.fdr_correction(ps)[1]
collected_mod = dict()
for m, p in zip(mods, corr_ps):
    mod = tuple(sorted((m[0], m[1])))
    if mod not in collected_mod.keys():
        collected_mod[mod] = [[m[2:], p]]
    else:
        collected_mod[mod].append([m[2:], p])

for k, v in collected_mod.items():
    print(k)
    for val in v:
        print(val)
    print('\n')
import pdb; pdb.set_trace()

'''
### Plotting main violin plots

for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        for computational_model, lines in a_data.items():

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

                            possibilities = ['dot', 'verb', 'simple']
                            builder = ['_concrete', '_abstract']

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
                            out_path  = os.path.join('plots', methodology, 'breakdown', 
                                                     analysis, 
                                                     senses, features, spatial_analysis, 
                                                     dataset)
                            os.makedirs(out_path, exist_ok=True)
                            out_path = os.path.join(out_path, '{}.jpg'.format(computational_model)) 

                            positions = list(range(len(current_collector.keys())))

                            ### Plotting the ceiling, if available
                            try:
                                ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                                ceil_header = ceiling_data[0]
                                ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                                ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                                ceiling_collector = dict()
                                for poss in possibilities:
                                    keyz = ['overall_accuracy',
                                            'abstract_concrete',
                                            poss,
                                            '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                    for h in keyz:
                                        try:
                                            ceiling_collector[h] = numpy.average(ceil_collector[h])
                                        except KeyError:
                                            continue
                                ceiling_pos = positions.copy()
                                ceiling_pos[0] = -.5
                                ceiling_pos[-1] = len(ceiling_pos) - .6
                                ax.fill_between(ceiling_pos, [v for k, v in ceiling_collector.items()], 
                                                [1. for i in ceiling_collector.keys()],
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
                                #p = scipy.stats.wilcoxon(data-.5, alternative='greater')[1]
                                p = scipy.stats.ttest_1samp(data, popmean=0.5, alternative='greater')[1]
                                mdn = round(numpy.median(data), 3)
                                ps.append(p)
                                avgs.append(avg)
                                mdns.append(mdn)
                                try:
                                    conf_interval = stats.t.interval(0.95, len(data)-1, loc=numpy.mean(data), scale=stats.sem(data))
                                except scipy.special.sf_error.SpecialFunctionError:
                                    conf_interval = [0., 0.]
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
                            
                            possibilities = ['book', 'catalogue', 'magazine', 'drawing']
                            polysemy_collector = dict()
                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        poss,
                                        ]
                                for h in keyz:
                                    try:
                                        polysemy_collector[h] = collector[h]
                                    except KeyError:
                                        continue
                            fig, ax = pyplot.subplots(figsize=(20, 9), constrained_layout=True)

                            positions = list(range(len(current_collector.keys())))

                            ### Creating the output path
                            out_path  = os.path.join('plots', methodology,
                                                     'polysemy', 
                                                     analysis, 
                                                     senses, features, spatial_analysis, 
                                                     dataset)
                            os.makedirs(out_path, exist_ok=True)
                            out_path = os.path.join(out_path, '{}.jpg'.format(computational_model)) 

                            positions = list(range(len(polysemy_collector.keys())))

                            ### Setting the title
                            if 'decoding' in methodology:
                                title = 'Decoding dot-object polysemy cases from fMRI to {}\nusing {} features from {} - {}'.format(
                                         computational_model, features.split('_')[1], analysis, spatial_analysis)
                            elif 'encoding' in methodology:
                                title = 'Encoding dot-object polysemy cases from {} to fMRI\nusing {} features from {} - {}'.format(
                                         computational_model, features.split('_')[1], analysis, spatial_analysis)
                            if 'True' in senses:
                                title =  '{}\naggregating phrases for word senses'.format(title)
                            title = title.replace('_', ' ')
                            ax.set_title(title, fontsize=20, pad=40)

                            ### Plotting the ceiling, if available
                            try:
                                ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                                ceil_header = ceiling_data[0]
                                ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                                ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                                ceiling_collector = dict()
                                for poss in possibilities:
                                    keyz = ['overall_accuracy',
                                            poss,
                                            ]
                                    for h in keyz:
                                        try:
                                            ceiling_collector[h] = numpy.average(ceil_collector[h])
                                        except KeyError:
                                            continue
                                ceiling_pos = positions.copy()
                                ceiling_pos[0] = -.5
                                ceiling_pos[-1] = len(ceiling_pos) - .6
                                ax.fill_between(ceiling_pos, [v for k, v in ceiling_collector.items()], 
                                                [1. for i in ceiling_collector.keys()],
                                                color='lightgray', alpha=0.3)
                            except KeyError:
                                pass

                            ps = list()
                            avgs = list()
                            mdns = list()
                            txt_collector = list()
                            for pos, data in zip(positions, polysemy_collector.items()):
                                data = numpy.array(data[1], dtype=numpy.float64)
                                ax.vlines(x=pos, ymin=min(data), ymax=max(data), color='lightgray')
                                txt_collector.append(data)
                                parts = ax.violinplot(data, positions=[pos], showmeans=False, showextrema=False)
                                for pc in parts['bodies']:
                                    if pos > 0:
                                        colors = numpy.linspace(.1, .7, 4)
                                        pc.set_facecolor(pyplot.get_cmap(cmap)(colors[pos-1]))
                                    elif pos == 0:
                                        pc.set_facecolor('darkslategrey')
                                    pc.set_edgecolor('lightgray')
                                    pc.set_alpha(.45)
                                avg = round(numpy.average(data), 3)
                                try:
                                    #p = scipy.stats.wilcoxon(data-.5, alternative='greater')[1]
                                    p = scipy.stats.ttest_1samp(data, popmean=0.5, alternative='greater')[1]
                                except UserWarning:
                                    import pdb; pdb.set_trace()
                                mdn = round(numpy.median(data), 3)
                                ps.append(p)
                                avgs.append(avg)
                                mdns.append(mdn)
                                try:
                                    conf_interval = stats.t.interval(0.95, len(data)-1, loc=numpy.mean(data), scale=stats.sem(data))
                                except scipy.special.sf_error.SpecialFunctionError:
                                    conf_interval = [0., 0.]
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
                            ax.set_ylim(ymin=-.3, ymax=1.)
                            ax.set_xlim(xmin=-1.5, xmax=max(positions)+.5)
                            ax.hlines(xmin=-.5, xmax=max(positions)+0.4, y=0.5, alpha=0.5, color='darkgray', linestyle='dashdot')
                            ax.set_xticks(positions)
                            x_ticks = [t.replace('concrete', 'object')\
                                        .replace('abstract', 'information')\
                                        .replace('_', ' ').capitalize() for t in polysemy_collector.keys()]
                            x_ticks = [x_ticks[0]] + ['{}\n(Info vs Obj)'.format(l) for l in x_ticks[1:]]
                            ax.set_xticklabels(x_ticks,
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
                            ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], alpha=0.2, color='darkgray', linestyle='dashdot')
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
                            ax.vlines(x=[.5], ymin=0.1, ymax=0.9, alpha=0.4)
                            ax.set_yticks(numpy.linspace(0.1, 1, 10))
                            for i in range(2):
                                ax.get_yticklabels()[0].set_visible(False)

                            pyplot.savefig(os.path.join(out_path))
                            pyplot.clf()
                            pyplot.close()
'''

### Bar plots for full comparisons
for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        for comp_model in ['concreteness', 'gpt2', 'fasttext']:

                            if spatial_analysis == 'whole_brain':
                                pass
                            else:
                                continue
                            fig, ax = pyplot.subplots(figsize=(20, 11), constrained_layout=True)

                            ### Setting the title
                            if 'decoding' in methodology:
                                title = 'Comparing decoding performance from fMRI\nusing {} features from {}\n{}'.format(
                                         features.split('_')[1], spatial_analysis, comp_model)
                            elif 'encoding' in methodology:
                                title = 'Comparing encoding performance to fMRI\nusing {} features from {}\n{}'.format(
                                         features.split('_')[1], spatial_analysis, comp_model)
                            if 'True' in senses:
                                title =  '{}\naggregating phrases for word senses'.format(title)
                            title = title.replace('_', ' ')
                            ax.set_title(title, fontsize=20, pad=20)

                            ### Creating the output path
                            out_path  = os.path.join('plots', methodology,
                                                     'comparisons_across_spatial_analyses', 
                                                     analysis, 
                                                     senses, features, dataset,
                                                     )
                            os.makedirs(out_path, exist_ok=True)
                            out_path = os.path.join(out_path, '{}_comparisons_{}.jpg'.format(comp_model, senses)) 

                            ### Comparing 3 features
                            spatials = ['whole_brain', 'fedorenko_language',
                                      'general_semantics', 'control_semantics']
                            spatials_collector = dict()
                            for m in spatials:
                                lines = whole_collector[dataset][m][methodology][features][senses][analysis][comp_model]

                                header = lines[0]
                                data = lines[1:]

                                collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                                spatials_collector[m] = dict()

                                possibilities = ['dot', 'verb', 'simple']
                                builder = ['_concrete', '_abstract']

                                for poss in possibilities:
                                    keyz = ['overall_accuracy',
                                            'abstract_concrete',
                                            poss,
                                            '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                    for h in keyz:
                                        try:
                                            spatials_collector[m][h] = collector[h]
                                        except KeyError:
                                            continue

                            positions = list(range(len(spatials_collector[m].keys())))

                            '''
                            ### Plotting the ceiling, if available
                            try:
                                ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                                ceil_header = ceiling_data[0]
                                ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                                ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                                ceiling_collector = dict()
                                for poss in possibilities:
                                    keyz = ['overall_accuracy',
                                            'abstract_concrete',
                                            poss,
                                            '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                    for h in keyz:
                                        try:
                                            ceiling_collector[h] = numpy.average(ceil_collector[h])
                                        except KeyError:
                                            continue
                                ceiling_pos = positions.copy()
                                ceiling_pos[0] = -.5
                                ceiling_pos[-1] = len(ceiling_pos) - .5
                                ax.fill_between(ceiling_pos, [v for k, v in ceiling_collector.items()], 
                                                [1. for i in ceiling_collector.keys()],
                                                color='lightgray', alpha=0.3)
                            except KeyError:
                                pass
                            '''

                            ### Plotting

                            color_map = {
                                         'whole_brain':'magenta', 
                                         'fedorenko_language':'orange', 
                                         'general_semantics':'lightseagreen', 
                                         'control_semantics' : 'mediumslateblue'
                                         }
                            scat_map = {
                                         'whole_brain':'darksalmon', 
                                         'fedorenko_language' : 'darkkhaki', 
                                         'general_semantics':'mediumseagreen',
                                         'control_semantics' : 'darkslategrey'
                                         }

                            corrections = {0 : -.3, 1 : -.1, 2 : .1, 3: .3}
                            for m_i, model in enumerate(spatials_collector.items()):
                                model_data = model[1]
                                model = model[0]
                                for pos, data in zip(positions, model_data.items()):
                                    data = numpy.array(data[1], dtype=numpy.float32)
                                    scat_data = sorted(data, key=lambda item : abs(numpy.average(data)-item))
                                    ### Dividing the participants
                                    beg_ends = [(0, 5), (5, 10), (10, 17)]
                                    scatter_corrections = {0:-.033, 1:0, 2:.033}
                                    for b_i, beg_end in enumerate(beg_ends):
                                        ax.scatter(x=[pos+corrections[m_i]+scatter_corrections[b_i] for d in data[beg_end[0]:beg_end[1]]],
                                                   y=data[beg_end[0]:beg_end[1]], zorder=2, 
                                                   edgecolors='black', linewidths=0.33,
                                                   c=scat_map[model], s=10.)
                                    ax.bar(x=pos+corrections[m_i], height=numpy.average(data),
                                           width = 0.166, align='center', zorder=1,
                                           label=model, color=color_map[model], alpha=1.-(pos*0.1))

                            ax.set_xticks(positions)
                            x_ticks = [t.replace('concrete', 'object')\
                                        .replace('abstract', 'information')\
                                        .replace('_', ' ').capitalize() for t in spatials_collector[model].keys()]
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
                            x_ticks_final = [t.replace('Dot', '')\
                                              .replace('Verb', '')\
                                              .replace('Simple', '')\
                                              .strip() for t in x_ticks_final]
                            ax.set_xticklabels(x_ticks_final,
                                               #rotation=45,
                                               ha='center',
                                               fontsize=20,
                                               #fontweight='bold'
                                               )
                            ax.hlines(xmin=-.3, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                    color='black', linestyle='dashdot')
                            ax.set_ylabel('Leave-2-out accuracy', fontsize=20, 
                                          fontweight='bold', y=0.62, ha='center',
                                          labelpad=8.)
                            ax.text(x=.5, y=0.95, s='Overall', fontsize=18,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=2.5, y=0.95, s='Coercion', fontsize=18,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=4.5, y=0.95, s='Transparent', fontsize=18,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=6.5, y=0.95, s='Light Verbs', fontsize=18,
                                    fontweight='bold', ha='center', va='center')
                            ax.vlines(x=[1.5, 3.5, 5.5], ymin=0.1, ymax=0.9, alpha=0.4, 
                                      color='gray')
                            ax.scatter(x=.35, y=1.025, s=300, color=color_map['whole_brain'], marker='s')
                            ax.text(x=.45, y=1.025, s='Whole Brain',fontsize=18,
                                    va='center', ha='left')
                            ax.scatter(x=1.6, y=1.025, s=300, color=color_map['fedorenko_language'], marker='s')
                            ax.text(x=1.7, y=1.025, s='Language Network',fontsize=18,
                                    va='center', ha='left')
                            ax.scatter(x=3.1, y=1.025, s=300, color=color_map['general_semantics'], marker='s')
                            ax.text(x=3.2, y=1.025, s='General Semantics',fontsize=18,
                                    va='center', ha='left')
                            ax.scatter(x=5.1, y=1.025, s=300, color=color_map['control_semantics'], marker='s')
                            ax.text(x=5.2, y=1.025, s='Control Semantics',fontsize=18,
                                    va='center', ha='left')

                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.tick_params('y', labelsize=15, length=0)
                            ax.tick_params('y', length=0)
                            ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], 
                                      alpha=0.2, color='darkgray', linestyle='dashdot')
                            ax.set_ylim(bottom=0.2, top=1.05)
                            ax.set_xlim(left=-.9, right=len(positions)-.5)

                            ### Computing ps, and then fdr corrections
                            scores_collector = list()
                            model_collector = list()
                            for k, v in spatials_collector.items():
                                for k_two, data in v.items():
                                    p = scipy.stats.ttest_1samp(numpy.array(data, dtype=numpy.float32),
                                                                popmean=0.5, alternative='greater')
                                    scores_collector.append(p[1])
                                    model_collector.append(k)
                            corr_ps = mne.stats.fdr_correction(scores_collector)[1]
                            corrected_collector = dict()
                            for m, p in zip(model_collector, corr_ps):
                                if m not in corrected_collector.keys():
                                    corrected_collector[m] = list()
                                corrected_collector[m].append(p)
                            corrections = {'whole_brain' : -.3, 
                                           'fedorenko_language' : -.1, 
                                           'general_semantics' : .1, 
                                           'control_semantics': .3}
                            for model, model_results in corrected_collector.items():
                                for pos, p in zip(positions, model_results):
                                    if p <= 0.05:
                                        ax.scatter(x=pos+corrections[model], y=0.23, 
                                                   marker='*', s=70, c='black', zorder=2)
                                    if p <= 0.005:
                                        ax.scatter(x=pos+corrections[model], y=0.25, 
                                                   marker='*', s=70, c='black', zorder=2)
                                    if p <= 0.0005:
                                        ax.scatter(x=pos+corrections[model], y=0.27, 
                                                   marker='*', s=70, c='black',
                                                   zorder=2)
                            
                            ax.text(x=-.7, y=0.31, ha='center', va='center', 
                                    s='p-value\n(FDR)', fontsize=12, fontweight='bold') 
                            ax.text(x=-.7, y=0.23, ha='center', va='center', 
                                    s='<0.05', fontsize=12, fontweight='bold') 
                            ax.text(x=-.7, y=0.25, ha='center', va='center', 
                                    s='<0.005', fontsize=12, fontweight='bold') 
                            ax.text(x=-.7, y=0.27, ha='center', va='center',
                                     s='<0.0005', fontsize=12, fontweight='bold') 
                            ax.text(x=-.7, y=0.5, ha='center', va='center',
                                     s='random\nbaseline', fontsize=12, fontweight='bold') 
                            '''
                            ### pairwise comparisons among models
                            scores_collector = dict()
                            for k, v in models_collector.items():
                                if k not in scores_collector.keys():
                                    scores_collector[k] = list()
                                for k_two, data in v.items():
                                    scores_collector[k].append(numpy.array(data, dtype=numpy.float32))
                            ### p-values for concreteness
                            p_val = dict()
                            for m in ['concreteness', 'gpt2']: 
                                for r_i, r in enumerate(scores_collector[m]):
                                    if m != 'gpt2':
                                        if (m, 'gpt2') not in p_val.keys():
                                            p_val[(m, 'gpt2')] =  list()
                                        ### gpt
                                        gpt_scores = scores_collector['gpt2'][r_i]
                                        p_val[(m, 'gpt2')].append(scipy.stats.ttest_rel(r, gpt_scores, alternative='greater')[1])
                                    ### fasttext
                                    if (m, 'fasttext') not in p_val.keys():
                                        p_val[(m, 'fasttext')] =  list()
                                    ft_scores = scores_collector['fasttext'][r_i]
                                    p_val[(m, 'fasttext')].append(scipy.stats.ttest_rel(r, ft_scores, alternative='greater')[1])

                            model_collector = list()
                            p_collector = list()
                            for model_comb, scores in p_val.items():
                                for score in scores:
                                    model_collector.append(model_comb)
                                    p_collector.append(score)

                            corr_ps = mne.stats.fdr_correction(p_collector)[1]
                            '''

                            pyplot.savefig(out_path)
                            pyplot.clf()
                            pyplot.close()

'''
### Bar plots for full comparisons
for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():

                        fig, ax = pyplot.subplots(figsize=(20, 11), constrained_layout=True)

                        ### Setting the title
                        if 'decoding' in methodology:
                            title = 'Comparing decoding performance from fMRI\nusing {} features from {}\n{}'.format(
                                     features.split('_')[1], analysis, spatial_analysis)
                        elif 'encoding' in methodology:
                            title = 'Comparing encoding performance to fMRI\nusing {} features from {}\n{}'.format(
                                     features.split('_')[1], analysis, spatial_analysis)
                        if 'True' in senses:
                            title =  '{}\naggregating phrases for word senses'.format(title)
                        title = title.replace('_', ' ')
                        ax.set_title(title, fontsize=20, pad=20)

                        ### Creating the output path
                        out_path  = os.path.join('plots', methodology,
                                                 'models_comparisons', 
                                                 analysis, 
                                                 senses, features, dataset,
                                                 )
                        os.makedirs(out_path, exist_ok=True)
                        out_path = os.path.join(out_path, '{}_{}.jpg'.format(spatial_analysis, senses)) 

                        ### Setting colors
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

                        ### Comparing 3 models
                        models = ['concreteness', 'gpt2', 'fasttext']
                        models_collector = dict()
                        for m in models:
                            lines = a_data[m]

                            header = lines[0]
                            data = lines[1:]

                            collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                            models_collector[m] = dict()

                            possibilities = ['dot', 'verb', 'simple']
                            builder = ['_concrete', '_abstract']

                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        'abstract_concrete',
                                        poss,
                                        '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                for h in keyz:
                                    try:
                                        models_collector[m][h] = collector[h]
                                    except KeyError:
                                        continue

                        positions = list(range(len(models_collector[m].keys())))

                        ### Plotting the ceiling, if available
                        try:
                            ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                            ceil_header = ceiling_data[0]
                            ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                            ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                            ceiling_collector = dict()
                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        'abstract_concrete',
                                        poss,
                                        '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                for h in keyz:
                                    try:
                                        ceiling_collector[h] = numpy.average(ceil_collector[h])
                                    except KeyError:
                                        continue
                            ceiling_pos = positions.copy()
                            ceiling_pos[0] = -.5
                            ceiling_pos[-1] = len(ceiling_pos) - .5
                            ax.fill_between(ceiling_pos, [v for k, v in ceiling_collector.items()], 
                                            [1. for i in ceiling_collector.keys()],
                                            color='lightgray', alpha=0.3)
                        except KeyError:
                            pass

                        ### Plotting

                        color_map = {'concreteness':'deeppink', 'gpt2':'teal', 'fasttext':'orange'}
                        scat_map = {'concreteness':'mediumvioletred', 
                                     'gpt2':'darkslategrey', 'fasttext':'peru'}

                        corrections = {0 : -.2, 1 : 0, 2: .2}
                        for m_i, model in enumerate(models_collector.items()):
                            model_data = model[1]
                            model = model[0]
                            for pos, data in zip(positions, model_data.items()):
                                data = numpy.array(data[1], dtype=numpy.float32)
                                scat_data = sorted(data, key=lambda item : abs(numpy.average(data)-item))
                                ### Dividing the participants
                                beg_ends = [(0, 5), (5, 10), (10, 17)]
                                scatter_corrections = {0:-.033, 1:0, 2:.033}
                                for b_i, beg_end in enumerate(beg_ends):
                                    ax.scatter(x=[pos+corrections[m_i]+scatter_corrections[b_i] for d in data[beg_end[0]:beg_end[1]]],
                                               y=data[beg_end[0]:beg_end[1]], zorder=2, 
                                               edgecolors='black', linewidths=0.33,
                                               c=scat_map[model], s=10.)
                                ax.bar(x=pos+corrections[m_i], height=numpy.average(data),
                                       width = 0.166, align='center', zorder=1,
                                       label=model, color=color_map[model], alpha=1.-(pos*0.1))

                        ax.set_xticks(positions)
                        x_ticks = [t.replace('concrete', 'object')\
                                    .replace('abstract', 'information')\
                                    .replace('_', ' ').capitalize() for t in models_collector[model].keys()]
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
                        ax.hlines(xmin=-.3, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                color='black', linestyle='dashdot')
                        ax.set_ylabel('Leave-2-out accuracy', fontsize=20, 
                                      fontweight='bold', y=0.62, ha='center',
                                      labelpad=8.)
                        ax.text(x=.5, y=0.95, s='Overall', fontsize=18,
                                fontweight='bold', ha='center', va='center')
                        ax.text(x=2.5, y=0.95, s='Coercion', fontsize=18,
                                fontweight='bold', ha='center', va='center')
                        ax.text(x=4.5, y=0.95, s='Transparent', fontsize=18,
                                fontweight='bold', ha='center', va='center')
                        ax.text(x=6.5, y=0.95, s='Light Verbs', fontsize=18,
                                fontweight='bold', ha='center', va='center')
                        ax.vlines(x=[1.5, 3.5, 5.5], ymin=0.1, ymax=0.9, alpha=0.4, 
                                  color='gray')
                        ax.scatter(x=.85, y=1.025, s=300, color=color_map['concreteness'], marker='s')
                        ax.text(x=.95, y=1.025, s='Concreteness vectors',fontsize=18,
                                va='center', ha='left')
                        ax.scatter(x=3.1, y=1.025, s=300, color=color_map['gpt2'], marker='s')
                        ax.text(x=3.2, y=1.025, s='Italian GPT-2',fontsize=18,
                                va='center', ha='left')
                        ax.scatter(x=5.1, y=1.025, s=300, color=color_map['fasttext'], marker='s')
                        ax.text(x=5.2, y=1.025, s='Italian fasttext',fontsize=18,
                                va='center', ha='left')

                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params('y', labelsize=15, length=0)
                        ax.tick_params('y', length=0)
                        ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], 
                                  alpha=0.2, color='darkgray', linestyle='dashdot')
                        ax.set_ylim(bottom=0.2, top=1.05)
                        ax.set_xlim(left=-.8, right=len(ceiling_pos)-.5)

                        ### Computing ps, and then fdr corrections
                        scores_collector = list()
                        model_collector = list()
                        for k, v in models_collector.items():
                            for k_two, data in v.items():
                                p = scipy.stats.ttest_1samp(numpy.array(data, dtype=numpy.float32),
                                                            popmean=0.5, alternative='greater')
                                scores_collector.append(p[1])
                                model_collector.append(k)
                        corr_ps = mne.stats.fdr_correction(scores_collector)[1]
                        corrected_collector = dict()
                        for m, p in zip(model_collector, corr_ps):
                            if m not in corrected_collector.keys():
                                corrected_collector[m] = list()
                            corrected_collector[m].append(p)
                        corrections = {'concreteness' : -.2, 'gpt2' : 0, 'fasttext': .2}
                        for model, model_results in corrected_collector.items():
                            for pos, p in zip(positions, model_results):
                                if p <= 0.05:
                                    ax.scatter(x=pos+corrections[model], y=0.23, 
                                               marker='*', s=70, c='black', zorder=2)
                                if p <= 0.005:
                                    ax.scatter(x=pos+corrections[model], y=0.25, 
                                               marker='*', s=70, c='black', zorder=2)
                                if p <= 0.0005:
                                    ax.scatter(x=pos+corrections[model], y=0.27, 
                                               marker='*', s=70, c='black',
                                               zorder=2)
                        
                        ax.text(x=-.5, y=0.31, ha='center', va='center', 
                                s='p-value\n(FDR)', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.23, ha='center', va='center', 
                                s='<0.05', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.25, ha='center', va='center', 
                                s='<0.005', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.27, ha='center', va='center',
                                 s='<0.0005', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.5, ha='center', va='center',
                                 s='random\nbaseline', fontsize=12, fontweight='bold') 
                        ### pairwise comparisons among models
                        scores_collector = dict()
                        for k, v in models_collector.items():
                            if k not in scores_collector.keys():
                                scores_collector[k] = list()
                            for k_two, data in v.items():
                                scores_collector[k].append(numpy.array(data, dtype=numpy.float32))
                        ### p-values for concreteness
                        p_val = dict()
                        for m in ['concreteness', 'gpt2']: 
                            for r_i, r in enumerate(scores_collector[m]):
                                if m != 'gpt2':
                                    if (m, 'gpt2') not in p_val.keys():
                                        p_val[(m, 'gpt2')] =  list()
                                    ### gpt
                                    gpt_scores = scores_collector['gpt2'][r_i]
                                    p_val[(m, 'gpt2')].append(scipy.stats.ttest_rel(r, gpt_scores, alternative='greater')[1])
                                ### fasttext
                                if (m, 'fasttext') not in p_val.keys():
                                    p_val[(m, 'fasttext')] =  list()
                                ft_scores = scores_collector['fasttext'][r_i]
                                p_val[(m, 'fasttext')].append(scipy.stats.ttest_rel(r, ft_scores, alternative='greater')[1])

                        model_collector = list()
                        p_collector = list()
                        for model_comb, scores in p_val.items():
                            for score in scores:
                                model_collector.append(model_comb)
                                p_collector.append(score)

                        corr_ps = mne.stats.fdr_correction(p_collector)[1]

                        pyplot.savefig(out_path)
                        pyplot.clf()
                        pyplot.close()

                        fig, ax = pyplot.subplots(figsize=(20, 11), constrained_layout=True)

                        ### Setting the title
                        if 'decoding' in methodology:
                            title = 'Comparing decoding performance for dot-object polysemy cases from fMRI\nusing {} features from {}\n{}'.format(
                                     features.split('_')[1], analysis, spatial_analysis)
                        elif 'encoding' in methodology:
                            title = 'Comparing encoding performance for dot-object polysemy cases to fMRI\nusing {} features from {}\n{}'.format(
                                     features.split('_')[1], analysis, spatial_analysis)
                        if 'True' in senses:
                            title =  '{}\naggregating phrases for word senses'.format(title)
                        title = title.replace('_', ' ')
                        ax.set_title(title, fontsize=20, pad=20)

                        ### Creating the output path
                        out_path  = os.path.join('plots', methodology,
                                                 'models_comparisons', 
                                                 analysis, 
                                                 senses, features, dataset,
                                                 )
                        os.makedirs(out_path, exist_ok=True)
                        out_path = os.path.join(out_path, 'polysemy_{}_{}.jpg'.format(spatial_analysis, senses)) 

                        ### Comparing 3 models
                        models = ['concreteness', 'gpt2', 'fasttext']
                        models_collector = dict()
                        for m in models:
                            lines = a_data[m]

                            header = lines[0]
                            data = lines[1:]

                            collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                            models_collector[m] = dict()
                            ### Polysemy
                            possibilities = ['book', 'catalogue', 'magazine', 'drawing']
                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        poss,
                                        ]
                                for h in keyz:
                                    try:
                                        models_collector[m][h] = collector[h]
                                    except KeyError:
                                        continue

                        positions = list(range(len(models_collector[m].keys())))

                        ### Plotting the ceiling, if available
                        try:
                            ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                            ceil_header = ceiling_data[0]
                            ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                            ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                            ceiling_collector = dict()
                            for poss in possibilities:
                                keyz = ['overall_accuracy',
                                        poss,
                                        ]
                                for h in keyz:
                                    try:
                                        ceiling_collector[h] = numpy.average(ceil_collector[h])
                                    except KeyError:
                                        continue
                            ceiling_pos = positions.copy()
                            ceiling_pos[0] = -.5
                            ceiling_pos[-1] = len(ceiling_pos) - .5
                            ax.fill_between(ceiling_pos, [v for k, v in ceiling_collector.items()], 
                                            [1. for i in ceiling_collector.keys()],
                                            color='lightgray', alpha=0.3)
                        except KeyError:
                            pass

                        ### Plotting

                        color_map = {'concreteness':'deeppink', 'gpt2':'teal', 'fasttext':'orange'}
                        scat_map = {'concreteness':'mediumvioletred', 
                                     'gpt2':'darkslategrey', 'fasttext':'peru'}

                        corrections = {0 : -.2, 1 : 0, 2: .2}
                        for m_i, model in enumerate(models_collector.items()):
                            model_data = model[1]
                            model = model[0]
                            for pos, data in zip(positions, model_data.items()):
                                data = numpy.array(data[1], dtype=numpy.float32)
                                scat_data = sorted(data, key=lambda item : abs(numpy.average(data)-item))
                                ### Dividing the participants
                                beg_ends = [(0, 5), (5, 10), (10, 17)]
                                scatter_corrections = {0:-.033, 1:0, 2:.033}
                                for b_i, beg_end in enumerate(beg_ends):
                                    ax.scatter(x=[pos+corrections[m_i]+scatter_corrections[b_i] for d in data[beg_end[0]:beg_end[1]]],
                                               y=data[beg_end[0]:beg_end[1]], zorder=2, 
                                               edgecolors='black', linewidths=0.33,
                                               c=scat_map[model], s=20.)
                                ax.bar(x=pos+corrections[m_i], height=numpy.average(data),
                                       width = 0.166, align='center', zorder=1,
                                       label=model, color=color_map[model], alpha=1.-(pos*0.1))

                        ax.set_xticks(positions)
                        x_ticks = [t.replace('concrete', 'object')\
                                    .replace('abstract', 'information')\
                                    .replace('_', ' ').capitalize() for t in models_collector[model].keys()]
                        x_ticks = [x_ticks[0]] + ['{}\n(Info vs Obj)'.format(l) for l in x_ticks[1:]]
                        ax.set_xticklabels(x_ticks,
                                           #rotation=45,
                                           ha='center',
                                           fontsize=20,
                                           #fontweight='bold'
                                           )
                        ax.hlines(xmin=-.3, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                color='black', linestyle='dashdot')
                        ax.set_ylabel('Leave-2-out accuracy', fontsize=20, 
                                      fontweight='bold', y=0.62, ha='center',
                                      labelpad=8.)
                        ax.scatter(x=.25, y=1.025, s=300, color=color_map['concreteness'], marker='s')
                        ax.text(x=.35, y=1.025, s='Concreteness vectors',fontsize=18,
                                va='center', ha='left')
                        ax.scatter(x=1.8, y=1.025, s=300, color=color_map['gpt2'], marker='s')
                        ax.text(x=1.9, y=1.025, s='Italian GPT-2',fontsize=18,
                                va='center', ha='left')
                        ax.scatter(x=3.1, y=1.025, s=300, color=color_map['fasttext'], marker='s')
                        ax.text(x=3.2, y=1.025, s='Italian fasttext',fontsize=18,
                                va='center', ha='left')

                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params('y', labelsize=15, length=0)
                        ax.tick_params('y', length=0)
                        ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], 
                                  alpha=0.2, color='darkgray', linestyle='dashdot')
                        ax.set_ylim(bottom=0., top=1.05)
                        ax.set_xlim(left=-.8, right=len(ceiling_pos)-.5)

                        ### Computing ps, and then fdr corrections
                        scores_collector = list()
                        model_collector = list()
                        for k, v in models_collector.items():
                            for k_two, data in v.items():
                                p = scipy.stats.ttest_1samp(numpy.array(data, dtype=numpy.float32),
                                                            popmean=0.5, alternative='greater')
                                scores_collector.append(p[1])
                                model_collector.append(k)
                        corr_ps = mne.stats.fdr_correction(scores_collector)[1]
                        corrected_collector = dict()
                        for m, p in zip(model_collector, corr_ps):
                            if m not in corrected_collector.keys():
                                corrected_collector[m] = list()
                            corrected_collector[m].append(p)
                        corrections = {'concreteness' : -.2, 'gpt2' : 0, 'fasttext': .2}
                        for model, model_results in corrected_collector.items():
                            for pos, p in zip(positions, model_results):
                                if p <= 0.05:
                                    ax.scatter(x=pos+corrections[model], y=0.03, 
                                               marker='*', s=70, c='black', zorder=2)
                                if p <= 0.005:
                                    ax.scatter(x=pos+corrections[model], y=0.05, 
                                               marker='*', s=70, c='black', zorder=2)
                                if p <= 0.0005:
                                    ax.scatter(x=pos+corrections[model], y=0.07, 
                                               marker='*', s=70, c='black',
                                               zorder=2)
                        
                        ax.text(x=-.5, y=0.11, ha='center', va='center', 
                                s='p-value\n(FDR)', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.03, ha='center', va='center', 
                                s='<0.05', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.05, ha='center', va='center', 
                                s='<0.005', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.07, ha='center', va='center',
                                 s='<0.0005', fontsize=12, fontweight='bold') 
                        ax.text(x=-.5, y=0.5, ha='center', va='center',
                                 s='random\nbaseline', fontsize=12, fontweight='bold') 
                        ### pairwise comparisons among models
                        scores_collector = dict()
                        for k, v in models_collector.items():
                            if k not in scores_collector.keys():
                                scores_collector[k] = list()
                            for k_two, data in v.items():
                                scores_collector[k].append(numpy.array(data, dtype=numpy.float32))
                        ### p-values for concreteness
                        p_val = dict()
                        for m in ['concreteness', 'gpt2']: 
                            for r_i, r in enumerate(scores_collector[m]):
                                if m != 'gpt2':
                                    if (m, 'gpt2') not in p_val.keys():
                                        p_val[(m, 'gpt2')] =  list()
                                    ### gpt
                                    gpt_scores = scores_collector['gpt2'][r_i]
                                    p_val[(m, 'gpt2')].append(scipy.stats.ttest_rel(r, gpt_scores, alternative='greater')[1])
                                ### fasttext
                                if (m, 'fasttext') not in p_val.keys():
                                    p_val[(m, 'fasttext')] =  list()
                                ft_scores = scores_collector['fasttext'][r_i]
                                p_val[(m, 'fasttext')].append(scipy.stats.ttest_rel(r, ft_scores, alternative='greater')[1])

                        model_collector = list()
                        p_collector = list()
                        for model_comb, scores in p_val.items():
                            for score in scores:
                                model_collector.append(model_comb)
                                p_collector.append(score)

                        corr_ps = mne.stats.fdr_correction(p_collector)[1]

                        pyplot.savefig(out_path)
                        pyplot.clf()
                        pyplot.close()
'''
