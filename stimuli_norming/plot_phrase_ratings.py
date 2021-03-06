import itertools
import matplotlib
import mne
import numpy
import os
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats
from sklearn import linear_model

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

out = 'phrase_rating_plots'
os.makedirs(out, exist_ok=True)

for noun_type in ['book', 'lunch']:
    for construction in ['single_', '']:

        f = '{}_fast_stimuli.tsv'.format(noun_type)
        with open(f) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        exp = {tuple(k[0].replace("'", ' ').split()) : k[1] for k in lines}
        exp = {(k[0], k[2]) if len(k)==3 else k : v for k, v in exp.items()}

        f = '../{}_fast_{}stimuli_ratings.tsv'.format(noun_type, construction)
        #f = '../{}_fast_single_stimuli_ratings.tsv'.format(noun_type)
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
        header = {h : numpy.array([d[h_i] for d in data], dtype=numpy.float64) for h_i, h in enumerate(headings) if 'avg' in h or 'std' in h}
        mapper = {d[0] : d[1] for d in data}
        colors = {'dot_concrete' : 'darkorange', 'dot_abstract' : 'navajowhite',
                  'simple_concrete' : 'mediumslateblue', 'simple_abstract' : 'lightsteelblue',
                  'verb_concrete' : 'mediumvioletred', 'verb_abstract' : 'hotpink'}
        markers = {'dot_concrete' : 's', 'dot_abstract' : 's',
                  'simple_concrete' : '8', 'simple_abstract' : '8',
                  'verb_concrete' : 'D', 'verb_abstract' : 'D'}

        ### All ratings
        fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
        for i, val in enumerate(header.values()):
            ax.violinplot(val/7, positions=[i], showmeans=True, showextrema=True)
            if construction == '':
                avg = round(numpy.average(val)/7, 3)
                mdn = round(numpy.median(val)/7, 3)
            else:
                avg = round(numpy.average(val)/5, 3)
                mdn = round(numpy.median(val)/5, 3)
            #avg = round(numpy.average(val)/5, 3)
            #print(val.shape)
            #mdn = round(numpy.median(val)/5, 3)
            ax.text(x=i+0.2, y=avg, s='avg: {}\nmdn: {}'.format(avg, mdn), ha='left', va='center')

        ax.set_xticks(range(len(header)))
        ax.set_xticklabels(header, ha='right', rotation=45)
        ax.set_ylim(ymin=0., ymax=1.1)
        #ax.set_title('Ratings for {}-noun_type polysemy cases'.format(noun_type))

        pyplot.savefig(os.path.join(out, '{}{}_ratings.jpg'.format(construction, noun_type)))
        pyplot.clf()
        pyplot.close()

    ### Concreteness
        header = {h : [d[h_i] for d in data] for h_i, h in enumerate(headings)}
        fig, ax = pyplot.subplots(figsize=(20, 10), constrained_layout=True)
        data = zip(header['phrase'], header['concreteness_avg'], header['concreteness_std'])
        data = sorted(data, key=lambda item : item[1])
        for i, phr_val in enumerate(data):
            if construction == '':
                avg = float(phr_val[1])/7
                std = float(phr_val[2])/7
            else:
                avg = float(phr_val[1])/5
                std = float(phr_val[2])/5
            ax.scatter(x=i, y=avg, color=colors[mapper[phr_val[0]]], 
                       s=200., zorder=2, marker=markers[mapper[phr_val[0]]])
            ax.vlines(x=i, ymin=avg-std, ymax=avg+std, 
                      color=colors[mapper[phr_val[0]]], 
                      linestyle='--',
                      #color='gray', 
                      alpha=0.5, zorder=1)
            ax.hlines(y=avg-std, xmin=i-0.1,xmax=i+0.1, 
                      #color='gray', 
                      color=colors[mapper[phr_val[0]]], 
                      alpha=0.2, zorder=1)
            ax.hlines(y=avg+std, xmin=i-0.1,xmax=i+0.1, 
                      #color='gray', 
                      color=colors[mapper[phr_val[0]]], 
                      alpha=0.2, zorder=1)
            ax.text(x=i, y=(float(phr_val[1])/7)-0.2, s=phr_val[0], 
                    rotation=75, 
                    va='top', ha='right', fontsize=15)

        ax.scatter(x=0., y=.71, color='lightsteelblue', s=200., marker=markers['simple_concrete'])
        ax.scatter(x=1.1, y=.71, color='mediumslateblue', s=200., marker=markers['simple_concrete'])
        ax.text(x=2.2, y=.71, s='Light Verb', va='center', ha='left', fontsize=20)

        ax.scatter(x=0, y=.79, color='hotpink', s=200., marker=markers['verb_concrete'])
        ax.scatter(x=1.1, y=.79, color='mediumvioletred', s=200., marker=markers['verb_concrete'])
        ax.text(x=2.2, y=.79, s='Transparent', va='center', ha='left', fontsize=20)

        ax.scatter(x=0., y=.87, color='navajowhite', s=200., marker=markers['dot_concrete'])
        ax.scatter(x=1.1, y=.87, color='darkorange', s=200., marker=markers['dot_concrete'])
        ax.text(x=2.2, y=.87, s='Coercion', va='center', ha='left', fontsize=20)

        ax.text(x=0., y=.92, s='Information', rotation=45, va='bottom', ha='left', fontsize=20, weight='normal')
        ax.text(x=1.1, y=.92, s='Object', rotation=45, va='bottom', ha='left', fontsize=20, weight='normal')
        ax.tick_params(axis='y', which='major', labelsize=20)

        ax.hlines(xmin=0., xmax=len(data), y=[0.,.2,.4,.6,.8,1.], alpha=0.3, color='darkgray', linestyles='dashed')
        ax.set_ylabel('Average concreteness rating', fontsize=25, fontweight='bold')
        ax.set_ylim(top=1.1, bottom=0.)
        ax.set_xlim(left=-2)
        ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.set_title('Concreteness ratings\nNumber of raters: 38', fontsize=20.)

        #xs = numpy.arange(0, len(header['concreteness_avg']), 1)
        ys = numpy.array(header['concreteness_avg'], dtype=numpy.float32)/7
        ys = numpy.linspace(ys.min(), ys.max(), num=ys.shape[0])
        ax.plot(ys, alpha=0.3, color='gray', zorder=1, linestyle='--')

        pyplot.savefig(os.path.join(out, '{}concreteness_{}_ratings.jpg'.format(construction, noun_type)))
        pyplot.clf()
        pyplot.close()
        conc_data = data.copy()

        ### computing p-values
        print([noun_type, construction])
        cases = ['dot', 'verb', 'simple']
        ps = list()
        names = list()
        for c in cases:
            abs_indices = [i for i, val in enumerate(header['category']) if c in val and 'bst' in val]
            abs_values = numpy.array(header['concreteness_avg'], dtype=numpy.float32)[abs_indices]
            conc_indices = [i for i, val in enumerate(header['category']) if c in val and 'oncr' in val]
            assert len(conc_indices) == 7
            conc_values = numpy.array(header['concreteness_avg'], dtype=numpy.float32)[conc_indices]
            p_val = scipy.stats.ttest_ind(abs_values, conc_values)[1]
            names.append('conc_{}'.format(c))
            ps.append(p_val)
        for c in cases:
            abs_indices = [i for i, val in enumerate(header['category']) if c in val and 'bst' in val]
            abs_values = numpy.array(header['familiarity_avg'], dtype=numpy.float32)[abs_indices]
            conc_indices = [i for i, val in enumerate(header['category']) if c in val and 'oncr' in val]
            assert len(conc_indices) == 7
            conc_values = numpy.array(header['familiarity_avg'], dtype=numpy.float32)[conc_indices]
            p_val = scipy.stats.ttest_ind(abs_values, conc_values)[1]
            names.append('fam_{}'.format(c))
            ps.append(p_val)

        ### Overall familiarity
        abs_indices = [i for i, val in enumerate(header['category']) if 'bst' in val]
        abs_values = numpy.array(header['familiarity_avg'], dtype=numpy.float32)[abs_indices]
        conc_indices = [i for i, val in enumerate(header['category']) if 'oncr' in val]
        assert len(conc_indices) == 21
        conc_values = numpy.array(header['familiarity_avg'], dtype=numpy.float32)[conc_indices]
        p_val = scipy.stats.ttest_ind(abs_values, conc_values)[1]
        names.append('overall_fam_{}'.format(c))
        ps.append(p_val)

        ### Overall concreteness
        abs_indices = [i for i, val in enumerate(header['category']) if 'bst' in val]
        abs_values = numpy.array(header['concreteness_avg'], dtype=numpy.float32)[abs_indices]
        conc_indices = [i for i, val in enumerate(header['category']) if 'oncr' in val]
        assert len(conc_indices) == 21
        conc_values = numpy.array(header['concreteness_avg'], dtype=numpy.float32)[conc_indices]
        p_val = scipy.stats.ttest_ind(abs_values, conc_values)[1]
        names.append('overall_conc_{}'.format(c))
        ps.append(p_val)

        ### Standard deviations
        combs = list(itertools.combinations(cases, 2))
        for comb in combs:
            abs_indices = [i for i, val in enumerate(header['category']) if comb[0] in val]
            abs_values = numpy.array(header['concreteness_std'], dtype=numpy.float32)[abs_indices]
            conc_indices = [i for i, val in enumerate(header['category']) if comb[1] in val]
            assert len(conc_indices) == 14
            conc_values = numpy.array(header['concreteness_std'], dtype=numpy.float32)[conc_indices]
            if construction == '':
                if 'simple' in comb[0]:
                    alternative = 'greater' if comb[0] == 'simple' else 'less'
                else:
                    alternative='two-sided'
            else:
                if 'dot' in comb[0]:
                    alternative = 'greater' if comb[0] == 'dot' else 'less'
                else:
                    alternative='two-sided'
            p_val = scipy.stats.ttest_ind(abs_values, conc_values, alternative=alternative)[1]
            print([comb, p_val])
            #names.append('std_{}'.format('_'.join(comb)))
            #ps.append(p_val)

        ### FDR correction
        corr_ps = mne.stats.fdr_correction(ps)[1]
        with open('{}significance_norming_{}.txt'.format(construction, noun_type), 'w') as o:
            o.write('case\tFDR-corrected p-value\n')
            for n, p in zip(names, corr_ps):
                o.write('{}\t{}\n'.format(n, float(p)))

        '''
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
        #ax.set_title('Imageability ratings for {}-noun_type polysemy cases\navg number of raters per item: 45'.format(noun_type), fontsize=20.)

        pyplot.savefig(os.path.join(out, 'imageability_{}_ratings.jpg'.format(noun_type)))
        pyplot.clf()
        pyplot.close()
        conc_data = sorted(conc_data, key=lambda item:item[0])
        data = sorted(data, key=lambda item:item[0])
        data = zip(header['phrase'], header['concreteness_avg'])
        print(scipy.stats.spearmanr([float(d[1]) for d in data], [float(d[1]) for d in conc_data]))
        '''
