import mne
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm

all_results = dict()
for mode in os.listdir('results'):
    for root, direc, filez in os.walk(os.path.join('results', mode)):
        for f in filez:
            if 'searchlight' not in root:

                #print(f)
                analysis = root.split('/')[-5]
                if analysis not in all_results.keys():
                    all_results[analysis] = dict()
                #print(analysis)

                dataset = root.split('/')[-6]
                if dataset not in all_results[analysis].keys():
                    all_results[analysis][dataset] = dict()
                #print(dataset)

                data_split = root.split('/')[-4]
                if data_split not in all_results[analysis][dataset].keys():
                    all_results[analysis][dataset][data_split] = dict()
                #print(data_split)

                n_folds = root.split('/')[-3]
                if n_folds not in all_results[analysis][dataset][data_split].keys():
                    all_results[analysis][dataset][data_split][n_folds] = dict()
                #print(n_folds)

                spatial = root.split('/')[-2]
                if spatial not in all_results[analysis][dataset][data_split][n_folds].keys():
                    all_results[analysis][dataset][data_split][n_folds][spatial] = dict()
                #print(spatial)

                feat = root.split('/')[-1]
                if feat not in all_results[analysis][dataset][data_split][n_folds][spatial].keys():
                    all_results[analysis][dataset][data_split][n_folds][spatial][feat] = dict()
                #print(feat)

                area = f.split('.')[0]

                with open(os.path.join(root, f)) as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                if analysis == 'whole_trial':
                    lines = [l[0] for l in lines]

                all_results[analysis][dataset][data_split][n_folds][spatial][feat][area] = lines

    with tqdm() as pbar:
        for analysis, analysis_results in all_results.items():
            for dataset, split_results in analysis_results.items():
                for data_split, split_results in split_results.items():
                    for n_folds, folds_results in split_results.items():
                        for spatial, spatial_results in folds_results.items():
                            for feat, dataset_results in spatial_results.items():
                                significance = 0.5
                                if feat == 'noun_soft' and dataset == 'book_fast':
                                    significance = 1/26
                                if feat == 'noun_soft' and dataset == 'lunch_fast':
                                    significance = 1/23
                                elif feat == 'verb_soft' and dataset == 'book_fast':
                                    significance = 1/9
                                elif feat == 'verb_soft' and dataset == 'lunch_fast':
                                    significance = 1/8
                                elif feat == 'soft':
                                    significance = 1/6
                                elif feat == 'stimuli':
                                    significance = 1/42
                                elif feat == 'columns':
                                    significance = 1/3
                
                                ### Plotting time-resolved
                                if analysis == 'time_resolved':

                                    labels = list(dataset_results.keys())
                                    data = numpy.array([v for k, v in dataset_results.items()], dtype=numpy.float64)
                                    #data = data[:, :, :18]

                                    averages = numpy.average(data, axis=1)
                                    #averages = averages.reshape((averages.shape[0], averages.shape[-1]))
                                    if 'fast' in dataset:
                                        xs = list(range(-1, averages.shape[1]-1))
                                    elif 'slow' in dataset:
                                        xs = list(range(-2, averages.shape[1]-2))
                                    sems = scipy.stats.sem(data, axis=1)
                                    #sems = sems.reshape((sems.shape[0], sems.shape[-1]))

                                    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(16, 9))
                                    all_ps = list()

                                    for average, sem, label, area_data in zip(averages, sems, labels, data):
                                        area_data = numpy.array(area_data)
                                        ax.plot(xs, average, label=label.replace('_', ' '))
                                        ax.fill_between(xs, average+sem, average-sem, alpha=0.1)
                                        ### P-values
                                        area_ps = list()
                                        for t_point in range(average.shape[-1]):
                                            #p_value = scipy.stats.wilcoxon(\
                                            p_value = scipy.stats.ttest_1samp(
                                                                area_data[:, t_point], \
                                                                #[significance for i in range(area_data.shape[0])], \
                                                                significance,
                                                                alternative= 'greater')[1]
                                            area_ps.append([p_value, average[t_point]])
                                        all_ps.append(area_ps)
                                    all_ps = numpy.array(all_ps)
                                    ps = all_ps[:, :, 0] #.flatten()
                                    avgs = all_ps[:, :, 1]
                                    ### FDR correction
                                    #corr_ps = mne.stats.fdr_correction(ps)[1]\
                                                           #.reshape(avgs.shape)
                                    #for l_i, l in enumerate(corr_ps):
                                    for l_i, l in enumerate(ps):
                                        corr_ps = mne.stats.fdr_correction(l)[1]
                                        for p_i, p in enumerate(corr_ps):
                                        #for p_i, p in enumerate(l):
                                            if p <= 0.05:
                                                ax.scatter(xs[p_i], \
                                                           avgs[l_i, p_i], \
                                                           edgecolors='black', \
                                                           color='white')
                                    title = 'Classification abstract vs concrete on \'{}\' polysemy cases\n'\
                                            '{} split - {} - {} - {}'.format(dataset, data_split, mode, analysis, feat)
                                    title = title.replace('_', ' ')
                                    ax.set_title(title, fontsize='xx-large', pad=50., fontweight='bold')
                                    ax.set_ylabel('Classification accuracy', fontsize='large')
                                    ax.set_xlabel('Seconds', fontsize='large')
                                    ax.set_xticks(xs)
                                    ax.hlines(y=significance, xmin=0, xmax=max(xs)+1, linestyles='dashed', color='darkgray', alpha=0.5)
                                    if significance == 0.5:
                                        verb_pos = -0.5 if 'fast' in dataset else -1.95
                                        ax.vlines(x=[verb_pos, 0.05], ymin=0.35, ymax=0.65, \
                                                  linestyles='dashed', color='darkgray', alpha=0.5)
                                        ax.text(x=verb_pos+0.1, y=0.65, s='verb', va='center', ha='left')
                                        ax.text(x=0.15, y=0.64, s='noun', va='center', ha='left')
                                        ax.scatter(x=12, y=0.36, color='white', edgecolors='black')
                                        ax.text(x=12.1, y=0.36, s='corrected p<=0.05', va='center', ha='left')
                                    ax.legend(ncol=4, fontsize='large', bbox_to_anchor=(0.86, 1.1),\
                                              frameon=False, markerscale=3.)

                                    out_path = os.path.join('plots', mode, dataset, n_folds, spatial, feat)
                                    os.makedirs(out_path, exist_ok=True)
                                    
                                    pyplot.savefig(os.path.join(out_path, '{}_{}.jpg'.format(analysis, data_split)), dpi=300)
                                    pyplot.clf()
                                    pyplot.close()

                                ### Plotting whole trial
                                elif analysis in ['whole_trial', 'flattened_trial']:
                                    labels = list(dataset_results.keys())
                                    data = numpy.array([v for k, v in dataset_results.items()], dtype=numpy.float64)

                                    xs = list(range(1, len(labels)+1))

                                    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(16, 9))
                                    p_values = list()
                                    for x, label, area_data in zip(xs, labels, data):
                                        area_data = numpy.array(area_data)
                                        if len(area_data.shape) == 2:
                                            area_data = area_data.reshape(-1)
                                        ax.violinplot(area_data, positions=[x], showmeans=True)
                                        ### P-values
                                        #p_value = scipy.stats.wilcoxon(\
                                        p_value = scipy.stats.ttest_1samp(
                                                            area_data, \
                                                            #[significance for i in range(area_data.shape[0])], \
                                                            significance,
                                                            alternative='greater',
                                                          
                                                            )[1]
                                        '''
                                        p_value = area_data - significance
                                        '''
                                        p_values.append(p_value)
                                    ### Check statistical significance with FDR
                                    corr_ps = mne.stats.fdr_correction(p_values)[1]
                                    '''
                                    p_values = numpy.array(p_values).T
                                    corr_ps = mne.stats.permutation_t_test(p_values,
                                                   tail=1, n_permutations=10000)[1]
                                    '''
                                    for x, p, area_data in zip(xs, corr_ps, data):
                                        area_data = numpy.array(area_data)
                                        if len(area_data.shape) == 2:
                                            area_data = area_data.reshape(-1)
                                        p = round(p, 4)
                                        ax.text(x=x+.2, y=numpy.average(area_data), \
                                                s='avg: {}\np={}'.format(round(numpy.average(area_data), 3), p), \
                                            ha='left', va='center')
                                    title = 'Classification abstract vs concrete on \'{}\' polysemy cases\n'\
                                            '{} split - {} - {} - {}'.format(dataset, data_split, mode, analysis, feat)
                                    title = title.replace('_', ' ')
                                    ax.set_title(title, fontsize='xx-large', pad=20., fontweight='bold')
                                    ax.set_ylabel('Classification accuracy', fontsize='large')
                                    ax.set_xticks(xs)
                                    ax.set_xticklabels([l.replace('_', '\n') for l in labels], \
                                                       fontsize='x-large', fontweight='bold')
                                    ax.hlines(y=significance, xmin=0, xmax=max(xs)+1, linestyles='dashed', color='darkgray', alpha=0.5)

                                    out_path = os.path.join('plots', mode, dataset, n_folds, spatial, feat)
                                    os.makedirs(out_path, exist_ok=True)
                                    
                                    pyplot.savefig(os.path.join(out_path, '{}_{}.jpg'.format(analysis, data_split)), dpi=300)
                                    pyplot.clf()
                                    pyplot.close()
                                pbar.update(1)
