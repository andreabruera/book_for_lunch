import argparse
import matplotlib
import mne
import numpy
import os
import re
import scipy
import warnings

from matplotlib import font_manager, pyplot
from scipy import stats

from compare_models_p_values import compute_p_values

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

parser = argparse.ArgumentParser()
parser.add_argument('--methodology', choices=[
                    'encoding', 'decoding', 
                    'rsa_encoding', 'rsa_decoding'],
                    required=True,
                    help = 'Encoding instead of decoding?')
args = parser.parse_args()

scipy.special.seterr(all='raise')
warnings.simplefilter('error')

p_values = compute_p_values(args)

folder = os.path.join('results', 'vector_{}'.format(args.methodology))

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
            if 'False' in senses:
                if senses not in whole_collector[dataset][spatial_analysis][methodology][features].keys():
                    whole_collector[dataset][spatial_analysis][methodology][features][senses] = dict()
                analysis = relevant_details[-6]
                if analysis not in whole_collector[dataset][spatial_analysis][methodology][features][senses].keys():
                    whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis] = dict()
                computational_model = relevant_details[-3]
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
                    try:
                        lines = [l.strip().split('\t') for l in i.readlines()]
                    except UnicodeDecodeError:
                        import pdb; pdb.set_trace()
                assert computational_model not in whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis].keys()
                whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis][computational_model] = lines

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
                            ax.set_title(title, fontsize=23, pad=40)

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
                                ax.text(x=pos, y=0.17, s=avg, fontsize=18,
                                        ha='center', va='center')
                                if p<0.05:
                                    weight = 'bold'
                                else:
                                    weight = 'normal'
                                ax.text(x=pos, y=0.1, s=round(p, 4), fontsize=18,
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
                                               fontsize=23,
                                               #fontweight='bold'
                                               )
                            ax.set_ylabel('Leave-2-out accuracy', fontsize=23, fontweight='bold', y=0.62, ha='center')

                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.tick_params('y', labelsize=15, length=0)
                            ax.tick_params('y', length=0)
                            ax.text(x=-.9, y=.8, s='Noise\nceiling', fontsize=21,
                                    ha='center', va='center', fontweight='bold')
                            ax.text(x=-.9, y=0.5, s='Chance level', fontsize=21,
                                    ha='center', va='center', fontweight='bold')
                            ax.text(x=-.9, y=0.17, s='Average', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=-.9, y=.1-0.01, s='P-value\n(FDR)', fontsize=21,
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

                            ax.text(x=.5, y=0.95, s='Overall', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=2.5, y=0.95, s='Coercion', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=4.5, y=0.95, s='Transparent', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=6.5, y=0.95, s='Light Verbs', fontsize=21,
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
                            ax.set_title(title, fontsize=23, pad=40)

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
                                ax.text(x=pos, y=-.3+(0.17), s=avg, fontsize=18,
                                        ha='center', va='center')
                                if p<0.05:
                                    weight = 'bold'
                                else:
                                    weight = 'normal'
                                ax.text(x=pos, y=-.3+(0.1), s=round(p, 4), fontsize=18,
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
                                               fontsize=23,
                                               #fontweight='bold'
                                               )
                            ax.set_ylabel('Leave-2-out accuracy', fontsize=23, fontweight='bold', y=0.62, ha='center')

                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.tick_params('y', labelsize=15, length=0)
                            ax.tick_params('y', length=0)
                            ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], alpha=0.2, color='darkgray', linestyle='dashdot')
                            ax.text(x=-.9, y=.8, s='Noise\nceiling', fontsize=21,
                                    ha='center', va='center', fontweight='bold')
                            ax.text(x=-.9, y=0.5, s='Chance level', fontsize=21,
                                    ha='center', va='center', fontweight='bold')
                            ax.text(x=-.9, y=-.3+(0.17), s='Average', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=-.9, y=-.3+(.1-0.01), s='P-value\n(FDR)', fontsize=21,
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
                            ax.set_title(title, fontsize=23, pad=20)

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
                                            '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                            ]
                                    for h in keyz:
                                        try:
                                            spatials_collector[m][h] = collector[h]
                                        except KeyError:
                                            continue

                            positions = list(range(len(spatials_collector[m].keys())))

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
                                               fontsize=23,
                                               #fontweight='bold'
                                               )
                            ax.hlines(xmin=-.3, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                    color='black', linestyle='dashdot')
                            ax.set_ylabel('Leave-2-out accuracy', fontsize=23, 
                                          fontweight='bold', y=0.62, ha='center',
                                          labelpad=8.)
                            ax.text(x=.5, y=0.95, s='Overall', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=2.5, y=0.95, s='Coercion', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=4.5, y=0.95, s='Transparent', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.text(x=6.5, y=0.95, s='Light Verbs', fontsize=21,
                                    fontweight='bold', ha='center', va='center')
                            ax.vlines(x=[1.5, 3.5, 5.5], ymin=0.1, ymax=0.9, alpha=0.4, 
                                      color='gray')
                            ax.scatter(x=.35, y=1.025, s=300, color=color_map['whole_brain'], marker='s')
                            ax.text(x=.45, y=1.025, s='Whole Brain',fontsize=21,
                                    va='center', ha='left')
                            ax.scatter(x=1.6, y=1.025, s=300, color=color_map['fedorenko_language'], marker='s')
                            ax.text(x=1.7, y=1.025, s='Language Network',fontsize=21,
                                    va='center', ha='left')
                            ax.scatter(x=3.1, y=1.025, s=300, color=color_map['general_semantics'], marker='s')
                            ax.text(x=3.2, y=1.025, s='General Semantics',fontsize=21,
                                    va='center', ha='left')
                            ax.scatter(x=5.1, y=1.025, s=300, color=color_map['control_semantics'], marker='s')
                            ax.text(x=5.2, y=1.025, s='Control Semantics',fontsize=21,
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

                            pyplot.savefig(out_path)
                            pyplot.clf()
                            pyplot.close()
'''
p_val_collector = dict()

### Bar plots for aggregate comparisons
for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        #for choices in [['concreteness', 'concretenesssingle', 'gpt2'],
                        #                ['concreteness', 'gpt2', 'fasttext']]:
                        for choices in [['concreteness', 'concretenesssingle', 'gpt2', 'fasttext']]:
                            for choice, comp_model in enumerate(choices):

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
                                #ax.set_title(title, fontsize=23, pad=20)

                                ### Creating the output path
                                out_path  = os.path.join('plots', methodology,
                                                         'comparisons_across_spatial_analyses', 
                                                         analysis, 
                                                         senses, features, dataset,
                                                         )
                                os.makedirs(out_path, exist_ok=True)
                                if choice == 0:
                                    out_path = os.path.join(out_path, '{}_comparisons_{}.jpg'.format(comp_model, senses)) 
                                else:
                                    out_path = os.path.join(out_path, '{}_single_concreteness_comparisons_{}.jpg'.format(comp_model, senses)) 

                                ### Comparing 3 features
                                spatials = ['whole_brain', 'fedorenko_language',
                                          'general_semantics', 'control_semantics']
                                spatials_collector = dict()
                                ceiling_collector = dict()
                                abs_conc_collector = dict()

                                for m in spatials:
                                    print(m)
                                    lines = whole_collector[dataset][m][methodology][features][senses][analysis][comp_model]

                                    header = lines[0]
                                    data = lines[1:]

                                    collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                                    spatials_collector[m] = dict()
                                    ceiling_collector[m] = dict()
                                    abs_conc_collector[m] = dict()

                                    possibilities = ['dot', 'verb', 'simple']
                                    builder = ['_concrete', '_abstract']
                                    ### Collecting abs/conc

                                    for poss in possibilities:
                                        keyz = [
                                                'abstract_concrete',
                                                '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                                ]
                                        for h in keyz:
                                            try:
                                                abs_conc_collector[m][h] = collector[h]
                                            except KeyError:
                                                continue

                                    for poss in possibilities:
                                        keyz = ['overall_accuracy',
                                                #'abstract_concrete',
                                                poss,
                                                #'{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                                ]
                                        for h in keyz:
                                            try:
                                                spatials_collector[m][h] = collector[h]
                                            except KeyError:
                                                continue
                                        ### Plotting the ceiling, if available
                                        try:
                                            ceiling_data = whole_collector[dataset][m][methodology][features][senses][analysis]['ceiling']
                                            ceil_header = ceiling_data[0]
                                            ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                                            ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                                            for poss in possibilities:
                                                keyz = ['overall_accuracy',
                                                        #'abstract_concrete',
                                                        poss,
                                                        #'{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                                        ]
                                                for h in keyz:
                                                    try:
                                                        ceiling_collector[m][h] = numpy.average(ceil_collector[h])
                                                    except KeyError:
                                                        continue
                                        except KeyError:
                                            pass

                                ### compute correlations
                                abs_conc = [float(val) for k_one, v_one in abs_conc_collector.items() for k_two, v_two in v_one.items() for val in v_two]
                                spat = [float(val) for k_one, v_one in spatials_collector.items() for k_two, v_two in v_one.items() for val in v_two]
                                corr = scipy.stats.pearsonr(abs_conc, spat)[0]
                                print([comp_model, senses, corr, round(numpy.average(abs_conc), 3), 
                                       'overall: {}'.format(round(numpy.average(spat), 3))])

                                positions = list(range(len(spatials_collector[m].keys())))

                                ### Plotting

                                color_map = {
                                             'whole_brain':'palevioletred', 
                                             'fedorenko_language':'chocolate', 
                                             'general_semantics':'steelblue', 
                                             'control_semantics' : 'khaki'
                                             }
                                scat_map = {
                                             'whole_brain':'pink', 
                                             'fedorenko_language' : 'sandybrown', 
                                             'general_semantics':'lightblue',
                                             'control_semantics' : 'beige'
                                             }

                                corrections = {'whole_brain' : -.3, 'fedorenko_language' : -.1, 
                                               'general_semantics' : .1, 'control_semantics' : .3}
                                for m_i, model in enumerate(spatials_collector.items()):
                                    model_data = model[1]
                                    model_name = model[0]
                                    for pos, data in zip(positions, model_data.items()):
                                        space_name = data[0]
                                        data = numpy.array(data[1], dtype=numpy.float32)
                                        scat_data = sorted(data, key=lambda item : abs(numpy.average(data)-item))
                                        ### Dividing the participants
                                        beg_ends = [(0, 5), (5, 10), (10, 17)]
                                        scatter_corrections = {0:-.033, 1:0, 2:.033}
                                        for b_i, beg_end in enumerate(beg_ends):
                                            ax.scatter(x=[pos+corrections[model_name]+scatter_corrections[b_i] for d in data[beg_end[0]:beg_end[1]]],
                                                       y=data[beg_end[0]:beg_end[1]], zorder=2, 
                                                       edgecolors='black', linewidths=0.33,
                                                       c=scat_map[model_name], s=40.)
                                        ax.bar(x=pos+corrections[model_name], 
                                               height=numpy.average(data)-.2,
                                               width = 0.166, align='center', zorder=1,
                                               color=color_map[model_name], 
                                               alpha=1.-(pos*0.15),
                                               bottom=0.2)
                                        ### ceiling
                                        ceil_value = ceiling_collector[model_name][space_name]
                                        #data = numpy.array(data[1], dtype=numpy.float32)
                                        ax.bar(x=pos+corrections[model_name], bottom=ceil_value,
                                               height=.9-ceil_value,
                                               width = 0.166, align='center', zorder=1,
                                               color='gray', alpha=0.1)


                                ax.set_xticks(positions)
                                ax.set_xticklabels(['Overall', 'Coercion', 'Transparent', 'Light Verbs'],
                                                   ha='center',
                                                   fontsize=28,
                                                   fontweight='bold',
                                                   )
                                '''
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
                                                   fontsize=23,
                                                   #fontweight='bold'
                                                   )
                                '''
                                ax.hlines(xmin=-.4, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                        color='black', linestyle='dashdot')
                                ax.set_ylabel('Leave-2-out accuracy', fontsize=23, 
                                              fontweight='bold', y=0.5, ha='center',
                                              labelpad=8.)
                                '''
                                ax.text(x=0., y=0.95, s='Overall', fontsize=21,
                                        fontweight='bold', ha='center', va='center')
                                ax.text(x=1., y=0.95, s='Coercion', fontsize=21,
                                        fontweight='bold', ha='center', va='center')
                                ax.text(x=2., y=0.95, s='Transparent', fontsize=21,
                                        fontweight='bold', ha='center', va='center')
                                ax.text(x=3., y=0.95, s='Light Verbs', fontsize=21,
                                        fontweight='bold', ha='center', va='center')
                                '''
                                ax.vlines(x=[0.5, 1.5, 2.5], ymin=0.1, ymax=.9, alpha=0.4, 
                                          color='gray')

                                ### Legend
                                ax.scatter(x=-.25, y=1.025-.1, s=300, color=color_map['whole_brain'], marker='s')
                                ax.text(x=-.2, y=1.025-.1, s='Whole Brain',fontsize=21,
                                        va='center', ha='left')
                                ax.scatter(x=.25, y=1.025-.1, s=300, color=color_map['fedorenko_language'], marker='s')
                                ax.text(x=.3, y=1.025-.1, s='Language Network',fontsize=21,
                                        va='center', ha='left')
                                ax.scatter(x=.95, y=1.025-.1, s=300, color=color_map['general_semantics'], marker='s')
                                ax.text(x=1., y=1.025-.1, s='General Semantics',fontsize=21,
                                        va='center', ha='left')
                                ax.scatter(x=1.65, y=1.025-.1, s=300, color=color_map['control_semantics'], marker='s')
                                ax.text(x=1.7, y=1.025-.1, s='Control Semantics',fontsize=21,
                                        va='center', ha='left')

                                ax.scatter(x=2.45, 
                                           y=.925, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.5, y=.925, s='/', 
                                        ha='center', va='center',
                                        fontsize=21)
                                ax.scatter(x=2.55, 
                                           y=.925, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.6, y=.925, s='/', 
                                        ha='center', va='center',
                                        fontsize=21)
                                ax.scatter(x=2.65, 
                                           y=.925, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.7, y=.925, s='p<0.05 / 0.005 / 0.0005', 
                                        ha='left', va='center',
                                        fontsize=21)

                                ax.spines['right'].set_visible(False)
                                ax.spines['top'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.tick_params('y', labelsize=18, length=0)
                                ax.tick_params('y', length=0)
                                ax.tick_params('x', bottom=False, labelbottom=True,
                                               pad=20)
                                ax.hlines(xmin=-.4, xmax=max(positions)+0.5, 
                                          y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, ], 
                                          alpha=0.2, color='darkgray', linestyle='dashdot')
                                ax.set_ylim(bottom=0.08, top=.95)
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
                                for model, model_results in corrected_collector.items():
                                    for pos, p in zip(positions, model_results):
                                        p_val_collector[(comp_model, model, pos)] = p
                                        if p <= 0.0005:
                                            ax.scatter(x=pos+corrections[model]-0.04, 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]-0., 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]+0.04, 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                        elif p > 0.0005 and p <= 0.005:
                                            ax.scatter(x=pos+corrections[model]-0.02, 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]+0.02, 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                        elif p > 0.005 and p <= 0.05:
                                            ax.scatter(x=pos+corrections[model], 
                                                       y=.23, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                
                                ax.text(x=-.7, y=0.31, ha='center', va='center', 
                                        s='p-value\n(FDR)', fontsize=21, fontweight='bold') 
                                '''
                                ax.text(x=-.7, y=0.23, ha='center', va='center', 
                                        s='<0.05', fontsize=13, fontweight='bold') 
                                ax.text(x=-.7, y=0.25, ha='center', va='center', 
                                        s='<0.005', fontsize=13, fontweight='bold') 
                                ax.text(x=-.7, y=0.27, ha='center', va='center',
                                         s='<0.0005', fontsize=13, fontweight='bold') 
                                '''
                                ax.text(x=-.7, y=0.23, ha='center', va='center',
                                         s='against\nbaseline', fontsize=18, 
                                         fontweight='bold', 
                                         style='italic') 
                                ax.text(x=-.7, y=0.125, ha='center', va='center',
                                         s='brain\nnetwork\ncomparisons', fontsize=18, 
                                         fontweight='bold', 
                                         style='italic') 
                                ax.text(x=-.7, y=0.5, ha='center', va='center',
                                         s='Random\nbaseline', fontsize=21, 
                                         fontweight='bold', 
                                         )
                                ax.text(x=-.7, y=.85, ha='center', va='center',
                                         s='Ceiling\nscores', fontsize=21, 
                                         fontweight='bold', 
                                         )
                                ### pairwise comparisons among models
                                ### Plotting p-values lines
                                heights = {
                                          'whole_brain' : .1,
                                          'fedorenko_language' : 0.125,
                                          'general_semantics' : 0.15,
                                          'control_semantics' : 0.175,
                                          }
                                cases = ['overall', 
                                         'coercion',
                                         'transparent',
                                         'light verbs',
                                         ]
                                for c_i, c in enumerate(cases):
                                    for h in heights:
                                        for h_two in heights:
                                            if h != h_two:
                                                key = tuple(sorted([h, h_two]))
                                                for val in p_values[key]:
                                                    if c in val[0] and comp_model in val[0]:
                                                        if val[1] <= 0.0005:
                                                            ax.scatter(x=c_i+corrections[h]-0.033, 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)
                                                            ax.scatter(x=c_i+corrections[h]-0., 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)
                                                            ax.scatter(x=c_i+corrections[h]+0.033, 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)
                                                        elif val[1] > 0.0005 and val[1] <= 0.005:
                                                            ax.scatter(x=c_i+corrections[h]-0.016, 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)
                                                            ax.scatter(x=c_i+corrections[h]+0.016, 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)
                                                        elif val[1] < 0.05 and val[1] >= 0.005:
                                                            ax.scatter(x=c_i+corrections[h], 
                                                                       y=heights[h_two], 
                                                                       marker='*', s=100, 
                                                                       edgecolors='black', 
                                                                       linewidths=.35,
                                                                       c=color_map[h_two], zorder=2)

                                for i in range(2):
                                    ax.get_yticklabels()[0].set_visible(False)
                                pyplot.savefig(out_path)
                                pyplot.clf()
                                pyplot.close()

### Bar plots for full comparisons
for dataset, d_data in whole_collector.items():
    for spatial_analysis, s_data in d_data.items():
        for methodology, m_data in s_data.items():
            for features, f_data in m_data.items():
                for senses, sense_data in f_data.items():
                    for analysis, a_data in sense_data.items():
                        all_possibilities = {'senses' : ['book', 'catalogue', 'magazine', 'drawing'],
                                             #'abs_con' : [
                                             #             'abstract', 'concrete', 
                                             #             'dot_abstract', 'dot_concrete',
                                             #             'verb_abstract', 'verb_concrete',
                                             #             'simple_abstract', 'simple_concrete'
                                             #             ],
                                             'info_obj' : ['dot', 'verb', 'simple'],
                                             'all' : ['dot', 'verb', 'simple']}
                        
                        all_models = [['concreteness', 'gpt2', 'fasttext'], 
                                      ['concreteness', 'concretenesssingle']]
                        for models_i, models in enumerate(all_models):
                            for poss_label, possibilities in all_possibilities.items():
                                if poss_label in ['all', 'info_obj']:
                                    max_value = 0.95
                                else:
                                    max_value = 1.05

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
                                #ax.set_title(title, fontsize=23, pad=20)

                                ### Creating the output path
                                out_path  = os.path.join('plots', methodology,
                                                         'models_comparisons', 
                                                         analysis, 
                                                         senses, features, dataset,
                                                         )
                                os.makedirs(out_path, exist_ok=True)
                                if poss_label == 'all':
                                    out_path = os.path.join(out_path, '{}_{}.jpg'.format(spatial_analysis, senses)) 
                                elif poss_label == 'info_obj':
                                    out_path = os.path.join(out_path, 'info_obj_{}_{}.jpg'.format(spatial_analysis, senses)) 
                                elif poss_label == 'abs_con':
                                    out_path = os.path.join(out_path, 'abs_con_{}_{}.jpg'.format(spatial_analysis, senses)) 
                                else:
                                    out_path = os.path.join(out_path, 'polisemy_{}_{}.jpg'.format(spatial_analysis, senses)) 
                                if models_i == 1:
                                    out_path = out_path.replace('{}_{}'.format(spatial_analysis, senses),
                                                                '{}_{}_single'.format(spatial_analysis, senses))

                                ### Comparing 3 models
                                models_collector = dict()
                                ceiling_collector = dict()
                                for m in models:
                                    lines = a_data[m]

                                    header = lines[0]
                                    data = lines[1:]

                                    collector = {h : [d[h_i] for d in data] for h_i, h in enumerate(header)}
                                    models_collector[m] = dict()
                                    ceiling_collector[m] = dict()

                                    builder = ['_concrete', '_abstract']

                                    for poss in possibilities:
                                        if poss_label == 'info_obj':
                                            keyz = [
                                                    #'overall_accuracy',
                                                    'abstract_concrete',
                                                    #poss,
                                                    '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                                    ]
                                        else:
                                            keyz = ['overall_accuracy',
                                                    #'abstract_concrete',
                                                    poss,
                                                    #'{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                                    ]
                                        for h in keyz:
                                            try:
                                                models_collector[m][h] = collector[h]
                                            except KeyError:
                                                continue
                                            ### Plotting the ceiling, if available
                                            try:
                                                ceiling_data = whole_collector[dataset][spatial_analysis][methodology][features][senses][analysis]['ceiling']
                                                ceil_header = ceiling_data[0]
                                                ceil_data = numpy.array(ceiling_data[1:], dtype=numpy.float32)
                                                ceil_collector = {h : [d[h_i] for d in ceil_data] for h_i, h in enumerate(ceil_header)}
                                                for poss in possibilities:
                                                    if poss_label == 'info_obj':
                                                        keyz = [
                                                                #'overall_accuracy',
                                                                'abstract_concrete',
                                                                #poss,
                                                                '{}{}_{}{}'.format(poss, builder[1], poss, builder[0])
                                                                ]
                                                    else:
                                                        keyz = ['overall_accuracy',
                                                                #'abstract_concrete',
                                                                poss,
                                                                #'{}{}_{}{}'.format(poss, builder[1], poss, builder[0])]
                                                                ]
                                                    for h in keyz:
                                                        try:
                                                            ceiling_collector[m][h] = numpy.average(ceil_collector[h])
                                                        except KeyError:
                                                            continue
                                            except KeyError:
                                                pass
                                if poss_label == 'abs_con':
                                    for k, v in models_collector.items():
                                        del v['overall_accuracy']

                                positions = list(range(len(models_collector[m].keys())))

                                ### Plotting

                                color_map = {'concreteness':'gray', 'concretenesssingle' : 'black',
                                             'gpt2':'goldenrod', 'fasttext':'lightskyblue'}
                                scat_map = {'concreteness':'gainsboro', 'concretenesssingle' : 'white',
                                             'gpt2':'wheat', 'fasttext':'powderblue'}

                                if models_i == 0:
                                    corrections = {'concreteness' : -.2, 'concretenesssingle' : .2,
                                                   'gpt2' : 0, 'fasttext': .2}
                                else:
                                    corrections = {'concreteness' : -.1, 
                                                   'concretenesssingle' : .1,
                                                   }
                                for m_i, model in enumerate(models_collector.items()):
                                    model_data = model[1]
                                    model = model[0]
                                    for pos, data in zip(positions, model_data.items()):
                                        h = data[0]
                                        data = numpy.array(data[1], dtype=numpy.float32)
                                        scat_data = sorted(data, key=lambda item : abs(numpy.average(data)-item))
                                        ### Dividing the participants
                                        beg_ends = [(0, 5), (5, 10), (10, 17)]
                                        scatter_corrections = {0:-.033, 1:0, 2:.033}
                                        for b_i, beg_end in enumerate(beg_ends):
                                            ax.scatter(x=[pos+corrections[model]+scatter_corrections[b_i] for d in data[beg_end[0]:beg_end[1]]],
                                                       y=data[beg_end[0]:beg_end[1]], zorder=2, 
                                                       edgecolors='black', linewidths=0.33,
                                                       c=scat_map[model], s=40.)
                                        height = numpy.average(data)
                                        bottom = 0.
                                        ceil_value = ceiling_collector[model][h]
                                        ceil_height = 1.-ceil_value
                                        alpha_mult = 0.15
                                        if poss_label in ['all', 'info_obj']:
                                            height = height - .2
                                            bottom = 0.2
                                            ceil_height = .9-ceil_value
                                        if poss_label == 'abs_con':
                                            alpha_mult = 0.1
                                        ax.bar(x=pos+corrections[model], 
                                               height=height,
                                               bottom=bottom,
                                               width = 0.166, align='center', zorder=1,
                                               color=color_map[model], 
                                               alpha=1.-(pos*alpha_mult))
                                        ### ceiling
                                        #data = numpy.array(data[1], dtype=numpy.float32)
                                        ax.bar(x=pos+corrections[model], 
                                               bottom=ceil_value,
                                               height=ceil_height,
                                               width = 0.166, align='center', zorder=1,
                                               color='gray', alpha=0.1)

                                ax.set_xticks(positions)
                                ax.hlines(xmin=-.3, xmax=max(positions)+0.4, y=0.5, alpha=0.5, 
                                        color='black', linestyle='dashdot')
                                ax.set_ylabel('Leave-2-out accuracy', fontsize=23, 
                                              fontweight='bold', y=0.5, ha='center',
                                              labelpad=8.)
                                if poss_label in ['all', 'info_obj']:
                                    vlines = [0.5, 1.5, 2.5]
                                else:
                                    vlines = [0.5, 1.5, 2.5, 3.5]
                                ax.vlines(x=vlines, ymin=0.1, ymax=.9, alpha=0.4, 
                                          color='gray')

                                ### Legend
                                ax.scatter(x=0.0, y=max_value-0.025, s=300, color=color_map['concreteness'], marker='s')
                                ax.text(x=.05, y=max_value-0.025, s='Concreteness',fontsize=21,
                                        va='center', ha='left')
                                if models_i == 0:
                                    ax.scatter(x=1.7, y=max_value-0.025, s=300, color=color_map['fasttext'], marker='s')
                                    ax.text(x=1.75, y=max_value-0.025, s='fasttext',fontsize=21,
                                            va='center', ha='left')
                                    ax.scatter(x=.85, y=max_value-0.025, s=300, color=color_map['gpt2'], marker='s')
                                    ax.text(x=.9, y=max_value-0.025, s='GPT-2',fontsize=21,
                                            va='center', ha='left')
                                else:
                                    ax.scatter(x=.85, y=max_value-0.025, s=300, color=color_map['concretenesssingle'], marker='s')
                                    ax.text(x=.9, y=max_value-0.025, s='concreteness (single)',fontsize=21,
                                            va='center', ha='left')

                                ax.scatter(x=2.35, 
                                           y=max_value-.025, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.4, y=max_value-.025, s='/', 
                                        ha='center', va='center',
                                        fontsize=21)
                                ax.scatter(x=2.45, 
                                           y=max_value-.025, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.5, y=max_value-.025, s='/', 
                                        ha='center', va='center',
                                        fontsize=21)
                                ax.scatter(x=2.55, 
                                           y=max_value-.025, 
                                           marker='*', s=200, 
                                           edgecolors='black', 
                                           linewidths=.5,
                                           c='white',
                                           zorder=3)
                                ax.text(x=2.6, y=max_value-.025, s='p<0.05 / 0.005 / 0.0005', 
                                        ha='left', va='center',
                                        fontsize=21)

                                ax.spines['right'].set_visible(False)
                                ax.spines['top'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.tick_params('y', labelsize=15, length=0)
                                ax.tick_params('y', length=0)
                                ax.hlines(xmin=-.4, xmax=max(positions)+0.5, y=[0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.], 
                                          alpha=0.2, color='darkgray', linestyle='dashdot')
                                ax.set_xlim(left=-.9, right=len(positions)-.5)

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
                                for model, model_results in corrected_collector.items():
                                    for pos, p in zip(positions, model_results):
                                        #if poss_label != 'all':
                                        #    print([senses, spatial_analysis, pos, model, p])

                                        ### Actually using previously collected p-value
                                        if poss_label in ['all', 'info_obj']:
                                            p = p_val_collector[(model, spatial_analysis, pos)]
                                            star_height = 0.23
                                        else:
                                            star_height = 0.03
                                        if p <= 0.0005:
                                            ax.scatter(x=pos+corrections[model]-0.04, 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]-0., 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]+0.04, 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                        elif p > 0.0005 and p <= 0.005:
                                            ax.scatter(x=pos+corrections[model]-0.02, 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                            ax.scatter(x=pos+corrections[model]+0.02, 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                        elif p > 0.005 and p <= 0.05:
                                            ax.scatter(x=pos+corrections[model], 
                                                       y=star_height, 
                                                       marker='*', s=150, 
                                                       edgecolors='black', 
                                                       linewidths=.5,
                                                       c='white',
                                                       zorder=3)
                                ax.text(x=-.7, y=star_height, ha='center', va='center',
                                         s='against\nbaseline', fontsize=18, 
                                         fontweight='bold', 
                                         style='italic') 
                                
                                ax.text(x=-.7, y=0.31, ha='center', va='center', 
                                        s='p-value\n(FDR)', fontsize=21, fontweight='bold') 
                                ax.text(x=-.7, y=0.5, ha='center', va='center',
                                         s='Random\nbaseline', fontsize=21, 
                                         fontweight='bold', 
                                         )
                                ax.text(x=-.7, y=.85, ha='center', va='center',
                                         s='Ceiling\nscores', fontsize=21, 
                                         fontweight='bold', 
                                         )
                                ### pairwise comparisons among models
                                ### Plotting p-values lines
                                if poss_label in ['all', 'info_obj']:
                                    if poss_label == 'all':
                                        set_labels= ['Overall', 'Coercion', 'Transparent', 'Light Verbs']
                                    else:
                                        set_labels= ['Overall\ninfo vs obj', 'Coercion\ninfo vs obj', 
                                                     'Transparent\ninfo vs obj', 'Light Verbs\ninfo vs obj']
                                    ax.set_xticklabels(set_labels,
                                                       ha='center',
                                                       fontsize=28,
                                                       fontweight='bold',
                                                       )
                                    ax.set_ylim(bottom=0.08, top=.95)
                                    for i in range(2):
                                        ax.get_yticklabels()[0].set_visible(False)
                                    ax.text(x=-.7, y=0.125, ha='center', va='center',
                                             s='brain\nnetwork\ncomparisons', fontsize=18, 
                                             fontweight='bold', 
                                             style='italic') 
                                    if models == 0:
                                        heights = {
                                                  'concreteness' : 0.125,
                                                  'gpt2' : 0.15,
                                                  'fasttext' : 0.175,
                                                  }
                                    else:
                                        heights = {
                                                  'concreteness' : 0.125,
                                                  #'gpt2' : 0.15,
                                                  'concretenesssingle' : 0.15,
                                                  }
                                    cases = ['overall', 
                                             'coercion',
                                             'transparent',
                                             'light verbs',
                                             ]
                                    for c_i, c in enumerate(cases):
                                        for h in heights:
                                            for h_two in heights:
                                                if h != h_two:
                                                    key = tuple(sorted([h, h_two]))
                                                    for val in p_values[key]:
                                                        if c in val[0] and spatial_analysis in val[0]:
                                                            if val[1] <= 0.0005:
                                                                ax.scatter(x=c_i+corrections[h]-0.033, 
                                                                           y=heights[h_two], 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           marker='*', s=100, 
                                                                           c=color_map[h_two], zorder=2)
                                                                ax.scatter(x=c_i+corrections[h]-0., 
                                                                           y=heights[h_two], 
                                                                           marker='*', s=100, 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           c=color_map[h_two], zorder=2)
                                                                ax.scatter(x=c_i+corrections[h]+0.033, 
                                                                           y=heights[h_two], 
                                                                           marker='*', s=100, 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           c=color_map[h_two], zorder=2)
                                                            elif val[1] > 0.0005 and val[1] <= 0.005:
                                                                ax.scatter(x=c_i+corrections[h]-0.016, 
                                                                           y=heights[h_two], 
                                                                           marker='*', s=100, 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           c=color_map[h_two], zorder=2)
                                                                ax.scatter(x=c_i+corrections[h]+0.016, 
                                                                           y=heights[h_two], 
                                                                           marker='*', s=100, 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           c=color_map[h_two], zorder=2)
                                                            elif val[1] < 0.05 and val[1] >= 0.005:
                                                                ax.scatter(x=c_i+corrections[h], 
                                                                           y=heights[h_two], 
                                                                           marker='*', s=100, 
                                                                           edgecolors='black', 
                                                                           linewidths=.35,
                                                                           c=color_map[h_two], zorder=2)
                                elif poss_label == 'abs_con':
                                    ax.set_ylim(bottom=-.05, top=1.05)
                                    ax.set_xticklabels(['Information', 'Object', 
                                                        'Coercion\ninformation', 'Coercion\nobject',
                                                        'Transparent\ninformation', 'Transparent\nobject',
                                                        'Light verb\ninformation', 'Light verb\nobject',
                                                        ],
                                                       ha='center',
                                                       fontsize=28,
                                                       fontweight='bold',
                                                       )

                                else:
                                    ax.set_ylim(bottom=-.05, top=1.05)
                                    ax.set_xticklabels(['Overall', 'Book', 
                                                        'Catalogue', 'Magazine',
                                                        'Drawing'],
                                                       ha='center',
                                                       fontsize=28,
                                                       fontweight='bold',
                                                       )


                                pyplot.savefig(out_path)
                                pyplot.clf()
                                pyplot.close()
