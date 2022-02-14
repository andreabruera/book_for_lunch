import argparse
import itertools
import nibabel
import nilearn
import numpy
import logging
import os
import random
import scipy
import sklearn

from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting
from scipy import stats
from sklearn import feature_selection
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from tqdm import tqdm

def read_vectors(vectors_folder):

    vectors = dict()
    for f in os.listdir(vectors_folder):
        assert '.vector' in f
        with open(os.path.join(vectors_folder, f)) as i:
            vecs = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float64)
        if vecs.shape[0] == 0:
            print(f)
            continue
        else:
            vecs = numpy.nanmean(vecs, axis=0)
            assert vecs.shape[0] == 768
            vectors[f.replace('_', ' ').split('.')[0]] = vecs
    return vectors

def read_events(events_path):
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
    trial_starts = [int(round(float(f), 0))-1 for f in events['onset'][1:][::jump]]
    ### Checking that trial length is between 10 and 12 seconds
    for t_i, t in enumerate(trial_starts):
        if t!=trial_starts[-1]:
            assert trial_starts[t_i+1]-t > 9
            assert trial_starts[t_i+1]-t < 13
    trial_infos = {'start' : list(), 'stimulus' : list(), 'category' : list()}
    for t_i, t in enumerate(list(range(len(events['onset'])))[1:][::jump]):
        cat = events['value'][t:t+jump][verb_idx]
        stimulus = events['trial_type'][t:t+jump]
        stimulus = '{} {}'.format(stimulus[verb_idx], stimulus[noun_idx])

        trial_infos['start'].append(trial_starts[t_i])
        trial_infos['category'].append(cat)
        trial_infos['stimulus'].append(stimulus)
    return trial_infos

def load_subject_runs(runs, map_nifti=None):

    sub_data = dict()

    for run, infos in tqdm(runs):

        if map_nifti != None:
            masked_run = nilearn.masking.apply_mask(run, map_nifti).T
        else:
            masked_run = run.get_fdata()
            masked_run = masked_run.reshape(numpy.product(masked_run.shape[:3]), -1)

        for t_i, t in enumerate(infos['start']):
            stimulus = infos['stimulus'][t_i]

            if args.analysis == 'time_resolved':
                fmri_timeseries = masked_run[:, t:t+18]
            elif args.analysis == 'whole_trial':
                ### Keeping responses from noun+4 to noun+9
                #fmri_timeseries = numpy.average(masked_run[:, t+6:t+13], \
                #                                axis=1)
                beg = t + 4
                end = t + 13
                if 'slow' in args.dataset:
                    beg += 1
                    end += 1
                fmri_timeseries = masked_run[:, beg:end].flatten()
            if stimulus not in sub_data.keys():
                sub_data[stimulus] = [fmri_timeseries]
            else:
                sub_data[stimulus].append(fmri_timeseries)

        del masked_run
    return sub_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['book_fast','lunch_fast', \
                                                         'book_slow', 'lunch_slow'],
                    help='Specify which dataset to use')
parser.add_argument('--analysis', required=True, \
                    choices=['time_resolved', 'whole_trial'], \
                    help='Average time points, or run classification'
                         'time point by time point?')
parser.add_argument('--spatial_analysis', choices=['ROI', 'all'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--feature_selection', choices=['anova', 'no_reduction'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--cross_validation', choices=['individual_trials', 'average_trials', 'replication'], \
                    required=True, help = 'Specifies how features are to be selected')
parser.add_argument('--n_folds', type=int, default=1000, \
                    help = 'Specifies how many folds to test on')
parser.add_argument('--vectors_folder', type=str, required=True, \
                    help = 'Specifies where the vectors are stored')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

os.makedirs('region_maps', exist_ok=True)
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'neuroscience', \
                        'dot_{}_bids'.format(args.dataset), 'derivatives',
                        )
n_subjects = len(os.listdir(dataset_path))
maps_folder = os.path.join('region_maps', 'maps')   
assert os.path.exists(maps_folder)
map_names = [n for n in os.listdir(maps_folder)]
map_results = dict()

vectors = read_vectors(args.vectors_folder)
decoding_results = list()

for s in range(1, n_subjects+1):
    #print(s)
    ### Loading the image
    sub_path = os.path.join(dataset_path, 'sub-{:02}'.format(s), 'ses-mri', \
                             'func',
                             )
    n_runs = len([k for k in os.listdir(sub_path) if 'nii' in k])

    logging.info('Now loading data for subject {}'.format(s))
    runs = list()

    for r in tqdm(range(1, n_runs+1)):
        #print(r)   
        ### Reading events
        events_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_events.tsv'.format(s, args.dataset.replace('_', ''), r))

        trial_infos = read_events(events_path)

        ### Reading fMRI run
        file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_bold.nii'.format(s, args.dataset.replace('_', ''), r))

        single_run = nilearn.image.load_img(file_path)
        ### Cleaning run file: detrending and standardizing
        single_run = nilearn.image.clean_img(single_run)
        runs.append((single_run, trial_infos))

    if args.spatial_analysis == 'all':

        logging.info('Masking using Fisher feature selection')
        #raise RuntimeError('Part to be implemented')
        ### Left hemisphere
        map_nifti = nilearn.image.load_img('region_maps/maps/left_hemisphere.nii')
        full_sub_data = load_subject_runs(runs, map_nifti)
        #full_sub_data = load_subject_runs(runs)
        ### Averaging, keeping only one response per stimulus
        sub_data = {k : numpy.average(v, axis=0) for k, v in full_sub_data.items()}
        dimensionality = list(set([v.shape[0] for k, v in sub_data.items()]))[0]
        ### Extracting 5k random indices
        #random_indices = random.sample(list(range(dimensionality)), k=10000)
        #sub_data = {k : v[random_indices] for k, v in sub_data.items()}
        with open(os.path.join('fisher_scores', args.dataset, args.analysis, 'sub-{:02}.fisher'.format(s))) as i:
            lines = numpy.array([l.strip().split('\t') for l in i.readlines()][0], dtype=numpy.float64)
        assert len(lines) == dimensionality
        sorted_dims = sorted(list(enumerate(lines)), key=lambda item : item[1], reverse=True)
        n_dims = 10000
        selected_dims = [k[0] for k in sorted_dims[:n_dims]]
        sub_data = {k : v[selected_dims] for k, v in sub_data.items()}

        ### Reduce keys to actually present
        sub_data = {k : v for k, v in sub_data.items() if k in vectors.keys()}
        vectors = {k : vectors[k] for k, v in sub_data.items()}
        combs = list(itertools.combinations(list(vectors.keys()), 2))
        accuracies = list()
        ### Splitting
        for c in tqdm(combs):
            train_inputs = [v for k, v in sub_data.items() if k not in c]
            train_targets = [v for k, v in vectors.items() if k not in c]

            test_inputs = [sub_data[c_i] for c_i in c]
            test_targets = [vectors[c_i] for c_i in c]

            model = Ridge(alpha=1.0)
            model.fit(train_inputs, train_targets)

            predictions = model.predict(test_inputs)
            assert len(predictions) == len(test_targets)
            wrong = 0.
            for idx_one, idx_two in [(0, 1), (1, 0)]:
                wrong += scipy.stats.spearmanr(predictions[idx_one], test_targets[idx_two])[0]
            correct = 0.
            for idx_one, idx_two in [(0, 0), (1, 1)]:
                correct += scipy.stats.spearmanr(predictions[idx_one], test_targets[idx_two])[0]
            if correct > wrong:
                accuracies.append(1)
            else:
                accuracies.append(0)
        accuracy = numpy.average(accuracies)
        print(accuracy)
        decoding_results.append(accuracy)

    if args.spatial_analysis == 'ROI':

        for map_name in map_names:

            map_path = os.path.join(maps_folder, map_name)
            assert os.path.exists(map_path)
            map_id = map_name.split('.')[0]
            logging.info('Masking area: {}'.format(map_id.replace('_', ' ')))
            map_nifti = nilearn.image.load_img(map_path)
            full_sub_data = load_subject_runs(runs, map_nifti=map_nifti)

            for data_split, sub_data in full_sub_data.items():
                ### Averaging presentations of the same stimulus
                if args.cross_validation == 'average_trials':
                    sub_data = {k : [numpy.average(v_two, axis=0) for k_two, v_two in v.items()] for k, v in sub_data.items()}
                else:
                    n_repetitions = list(set([len(v) for k, v in sub_data.items()]))[0]
                    sub_data = {k : [v_three for k_two, v_two in v.items() for v_three in v_two] for k, v in sub_data.items()}
                assert len(set([len(v) for k, v in sub_data.items()])) == 1
                n_items = list(set([len(v) for k, v in sub_data.items()]))[0]

                labels = list()
                samples = list()

                for i in range(n_items):
                    for k, v in sub_data.items():
                        labels.append(k)
                        samples.append(v[i])

                ### Averaged trials - all trials for one exemplar are averaged
                ### Making sure there are at least 2 test items
                if args.cross_validation == 'average_trials':
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
                ### Individual_trials - keeping all trials for one exemplar 
                ### per category out for testing
                elif args.cross_validation == 'individual_trials':
                    raise RuntimeError('To be corrected')
                    one_tenth = max(2, int((len(samples)/(10*n_repetitions))))
                    one_label = [i for i in range(0, n_items*2)][::n_repetitions]
                    two_label = [i for i in range(1, n_items*2)][::n_repetitions]
                    if one_tenth > 2:
                        one_label = itertools.combinations(one_label, r=int(one_tenth/2))
                        two_label = itertools.combinations(two_label, r=int(one_tenth/2))
                        splits = list(itertools.product(one_label, two_label))
                        splits = [[val for c_two in c for k in c_two for val in range(k, k+(n_repetitions*2), 2)] for c in splits]

                    else:
                        splits = list(itertools.product(one_label, two_label))
                        splits = [[val for v in c for val in range(v, v+(n_repetitions*2), 2)] for c in splits]
                ### Replication - leave-two-trials out evaluation
                elif args.cross_validation == 'replication':
                    one_tenth = 2
                    one_label = [i for i in range(0, n_items*2)][::2]
                    two_label = [i for i in range(1, n_items*2)][::2]
                    splits = list(itertools.product(one_label, two_label))

                ### Randomizing and reducing test splits (default 1000)
                n_folds = min(len(splits), args.n_folds)
                if args.n_folds == 1000 and n_folds < 100:
                    print('Does not make sense to run this one')
                    continue
                print('Current split: {} - number of folds: {}'.format(data_split, n_folds)) 
                splits = random.sample(splits, k=n_folds)
                sub_scores = list()
                logging.info('Running cross-validation')
                for split in tqdm(splits):
                    #train_samples = samples[0:split] + samples[split+one_tenth:]
                    #train_labels = labels[0:split] + labels[split+one_tenth:]
                    #test_samples = samples[split:split+one_tenth]
                    #test_labels = labels[split:split+one_tenth]
                    train_samples = [samples[i] for i in range(n_items*2) if i not in split]
                    train_labels = [labels[i] for i in range(n_items*2) if i not in split]
                    test_samples = [samples[i] for i in split]
                    test_labels = [labels[i] for i in split]

                    classifier = SVC(gamma='auto')
                    split_score = list()
                    if args.analysis == 'time_resolved':
                        for t in range(14):
                            t_train = [sample[:, t] for sample in train_samples]
                            t_test = [sample[:, t] for sample in test_samples]
                            classifier.fit(t_train, train_labels)
                            acc = classifier.score(t_test, test_labels)
                            split_score.append(acc)

                    elif args.analysis == 'whole_trial':

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

output_folder = os.path.join('results', args.cross_validation, 
                             'simple_decoding', \
                             args.dataset, args.analysis, 
                             'pairwise', \
                             args.spatial_analysis, \
                             args.feature_selection,
                             )
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, 'accuracies.results'), 'w') as o:
    for acc in decoding_results:
        o.write('{}\n'.format(acc))
