import argparse
import itertools
import nibabel
import nilearn
import numpy
import logging
import os
import pandas
import random
import sklearn

from matplotlib import pyplot
from nilearn import datasets, decoding, image, input_data, plotting
from nilearn.glm.first_level import FirstLevelModel
from sklearn import feature_selection
from sklearn.svm import SVC
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, 
                    choices=['book_fast','lunch_fast', \
                    'book_slow', 'lunch_slow'],
                    help='Specify which dataset to use')
parser.add_argument('--n_folds', type=int, default=50, \
                    help = 'Specifies how many folds to test on')
parser.add_argument('--spatial_analysis', choices=['ROI'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--feature_selection', choices=['anova', 'no_reduction'], required=True, \
                    help = 'Specifies how features are to be selected')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.INFO)

os.makedirs('region_maps', exist_ok=True)
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 
                            'dataset', 'neuroscience', \
                            'dot_{}_bids'.format(args.dataset), 
                            'derivatives',
                           )
n_subjects = len(os.listdir(dataset_path))
maps_folder = os.path.join('region_maps', 'maps')   
assert os.path.exists(maps_folder)
map_names = [n for n in os.listdir(maps_folder) if 'hemisphere' not in n]
map_results = dict()

for s in range(1, n_subjects+1):
    ### Loading the image
    sub_path = os.path.join(dataset_path, 'sub-{:02}'.format(s), 'ses-mri', \
                             'func',
                             )
    n_runs = len([k for k in os.listdir(sub_path) if 'nii' in k])

    logging.info('Now loading data for subject {}'.format(s))
    runs = list()
    cat_mapper = dict()

    for r in tqdm(range(1, n_runs+1)):
        #print(r)   
        ### Reading events
        events_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_events.tsv'.format(s, args.dataset.replace('_', ''), r))
        #events = pandas.read_csv(events_path, sep='\t')
        with open(events_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        events = {h : [l[h_i] for l in lines[1:]] for h_i, h in enumerate(lines[0])}
        if 'slow' in args.dataset:
            jump = 4 
            verb_idx = 0
            noun_idx = 1
        elif 'fast' in args.dataset:
            jump = 7
            verb_idx = 1 
            noun_idx = 3
        trial_starts = [float(f) for f in events['onset'][1:][::jump]]
        ### Checking that trial length is between 10 and 12 seconds
        for t_i, t in enumerate(trial_starts):
            if t!=trial_starts[-1]:
                assert trial_starts[t_i+1]-t > 9
                assert trial_starts[t_i+1]-t < 13
        trial_infos = {'onset' : list(), 
                       'trial_type' : list()}
        for t_i, t in enumerate(list(range(len(events['onset'])))[1:][::jump]):
            cat = events['value'][t:t+jump][verb_idx]
            stimulus = events['trial_type'][t:t+jump]
            stimulus = '{} {}'.format(stimulus[verb_idx], stimulus[noun_idx])

            trial_infos['onset'].append(trial_starts[t_i])
            if 'neg neg' not in stimulus:
                cat_mapper[stimulus] = cat
            trial_infos['trial_type'].append(stimulus)
        trial_infos['duration'] = [1. for i in trial_infos['onset']]
        trial_infos = pandas.DataFrame.from_dict(trial_infos)

        file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_bold.nii'.format(s, args.dataset.replace('_', ''), r))

        single_run = nilearn.image.load_img(file_path)

        runs.append((single_run, trial_infos))

    logging.info('Now estimating BOLD responses using GLM '\
                 'for subject {}'.format(s))
    fmri_glm = FirstLevelModel(t_r=1, standardize=True,
                               smoothing_fwhm=8.)
    fmri_glm = fmri_glm.fit([r[0] for r in runs], [r[1] for r in runs])

    betas = {k : fmri_glm.compute_contrast(k) for k in cat_mapper.keys()}
    sub_data = {'all' : dict(), 'simple' : dict(), 
                'verb' : dict(), 'dot' : dict()}
    for stimulus, full_cat in cat_mapper.items():
        if 'con' in full_cat or 'obj' in full_cat:
            cat = 'concrete'
        elif 'abs' in full_cat or 'info' in full_cat:
            cat = 'abstract'
        elif 'catch' in full_cat:
            cat = ''
        ### Not considering cases with no stimulus
        if cat in ['abstract', 'concrete']:
            if cat not in sub_data['all'].keys():
                sub_data['all'][cat] = [betas[stimulus]]
            else:
                sub_data['all'][cat].append(betas[stimulus])
            ### Generic abstract/concrete
            ### Data subdivision
            if 'Coer' in full_cat:
                full_cat = 'dot'
            elif 'Event' in full_cat:
                full_cat = 'verb'
            else:
                full_cat = 'simple'

            if cat not in sub_data[full_cat].keys():
                sub_data[full_cat][cat] = [betas[stimulus]]
            else:
                sub_data[full_cat][cat].append(betas[stimulus])

    if args.spatial_analysis == 'ROI':

        for map_name in map_names:

            map_path = os.path.join(maps_folder, map_name)
            assert os.path.exists(map_path)
            map_id = map_name.split('.')[0]
            logging.info('Masking area: {}'.format(map_id.replace('_', ' ')))
            map_nifti = nilearn.image.load_img(map_path)
            for data_split, current_data in sub_data.items():
                logging.info('Running Searchlight for {}'.format(data_split))

                labels = list()
                samples = list()

                for k, v in current_data.items():
                    n_items = len(v)
                    for item in v:
                        labels.append(k)
                        masked_item = nilearn.masking.apply_mask(item, map_nifti)
                        samples.append(masked_item)

                one_tenth = max(2, int((len(samples)/10)))
                one_label = [i for i in range(0, n_items*2)][::2]
                two_label = [i for i in range(1, n_items*2)][::2]
                if one_tenth > 2:
                    one_label = itertools.combinations(one_label, r=int(one_tenth/2))
                    two_label = itertools.combinations(two_label, r=int(one_tenth/2))
                    splits = list(itertools.product(one_label, two_label))
                    splits = [[k for c_two in c for k in c_two] for c in splits]
                else:
                    splits = list(itertools.product(one_label, two_label))
                n_folds = min(len(splits), args.n_folds)
                if args.n_folds == 1000 and n_folds < 100:
                    print('Does not make sense to run this one')
                    continue

                print('Current split: {} - number of folds: {}'.format(data_split, n_folds)) 
                splits = random.sample(splits, k=n_folds)
                sub_scores = list()
                logging.info('Running cross-validation')
                for split in tqdm(splits):
                    train_samples = [samples[i] for i in range(n_items*2) if i not in split]
                    train_labels = [labels[i] for i in range(n_items*2) if i not in split]
                    test_samples = [samples[i] for i in split]
                    test_labels = [labels[i] for i in split]

                    classifier = SVC(gamma='auto')
                    split_score = list()
                    ### Feature selection
                    ## Anova
                    if args.feature_selection == 'anova':
                        selector = sklearn.feature_selection.SelectPercentile(\
                                            score_func=sklearn.feature_selection.f_classif, \
                                            percentile=50).fit(train_samples, train_labels)
                        train_samples = selector.transform(train_samples)
                        test_samples = selector.transform(test_samples)

                    classifier.fit(train_samples, train_labels)
                    acc = classifier.score(test_samples, test_labels)
                    split_score.append(acc)

                    sub_scores.append(split_score)

                average_scores = numpy.average(sub_scores, axis=0)
                logging.info('Average scores for area {}: {}'.format(map_id, average_scores))

                if map_id not in map_results.keys():
                    map_results[map_id] = dict()

                if data_split not in map_results[map_id].keys():
                    map_results[map_id][data_split] = [average_scores]
                else:
                    map_results[map_id][data_split].append(average_scores)

output_folder = os.path.join('results', 
                             'glm_classification', \
                             args.dataset, 
                             '{}_folds'.format(args.n_folds), \
                             args.spatial_analysis, \
                             args.feature_selection,
                             )
os.makedirs(output_folder, exist_ok=True)
for area, all_results in map_results.items():
    for data_split, v in all_results.items():
        current_output = os.path.join(output_folder, data_split)
        os.makedirs(current_output, exist_ok=True)
        with open(os.path.join(current_output, \
                               '{}.results'.format(area)), 'w') as o:
            for sub_scores in v:
                for acc in sub_scores:
                    o.write('{}\t'.format(acc))
                o.write('\n')
