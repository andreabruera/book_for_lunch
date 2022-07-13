import argparse
import itertools
import json
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

def compute_p_values(args, write=False):

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

    ### Comparisons between areas
    for dataset, d_data in whole_collector.items():
        spaces = d_data.keys()
        spaces_combs = list(itertools.combinations(spaces, 2))
        for spatial_analysis, s_data in d_data.items():
            for methodology, m_data in s_data.items():
                for features, f_data in m_data.items():
                    for senses, sense_data in f_data.items():
                        for analysis, a_data in sense_data.items():
                            for computational_model, m_data in a_data.items():
                                if spatial_analysis == 'whole_brain':
                                    for space_one, space_two in spaces_combs:
                                        res_one = whole_collector[dataset][space_one][methodology][features][senses][analysis][computational_model]
                                        res_two = whole_collector[dataset][space_two][methodology][features][senses][analysis][computational_model]
                                        for case, combs in cases.items():
                                            cont_table = numpy.zeros((2, 2))
                                            for s in range(1, 17):
                                                all_subs_one = [res_one['sub-{:02}'.format(s)][c] for c in combs]
                                                all_subs_two = [res_two['sub-{:02}'.format(s)][c] for c in combs]
                                                for o, t in zip(all_subs_one, all_subs_two):
                                                    cont_table[o, t] += 1
                                            p_value = statsmodels.stats.contingency_tables.mcnemar(cont_table)
                                            mods.append([space_one, space_two, computational_model, case]) 
                                            ps.append(vars(p_value)['pvalue'])

    corr_ps = mne.stats.fdr_correction(ps)[1]
    collected_mod = dict()
    for m, p in zip(mods, corr_ps):
        mod = tuple(sorted((m[0], m[1])))
        if mod not in collected_mod.keys():
            collected_mod[mod] = [[m[2:], p]]
        else:
            collected_mod[mod].append([m[2:], p])

    if write:
        with open(os.path.join('results', 'all_classifiers_comparisons.txt'), 'w') as o:

            for k, v in collected_mod.items():
                o.write('{}\n\n'.format(k))
                for val in v:
                    o.write('{}\n'.format(val))
                o.write('\n\n')

    return collected_mod
