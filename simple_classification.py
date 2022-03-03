import argparse
import itertools
import nibabel
import nilearn
import numpy
import logging
import os
import random
import sklearn

from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting
from sklearn import feature_selection, linear_model
from sklearn.svm import SVC
from tqdm import tqdm

def load_subject_runs(runs, map_nifti=None):

    mapper = {
              'hard_labels' : {'info' : 0, 'absEvent' : 0, 'absCoercion' : 0,
                              'absSimple' : 0, 'absCoer' : 0, 
                              'obj' : 1, 'conEvent' : 1, 'conCoercion' : 1, 
                              'conSimple' : 1, 'conCoer' : 1, 
                              },
              'soft_labels' : {'info' : 2.22, 'absEvent' : 3.32, 'absCoercion' : 3.64,
                              'absSimple' : 3.95, 'absCoer' : 3.58, 
                              'obj' : 3.65, 'conEvent' : 4.41, 'conCoercion' : 4.32, 
                              'conSimple' : 4.46, 'conCoer' : 4.29, 
                              },
              'column_labels' : {'info' : 0, 'absEvent' : 1, 'absCoercion' : 2,
                              'absSimple' : 0, 'absCoer' : 2, 
                              'obj' : 0, 'conEvent' : 1, 'conCoercion' : 2, 
                              'conSimple' : 0, 'conCoer' : 2, 
                              },
              }
    ### Separate mapper for lunch
    lunch_mapper = {'conEvent' : 4.37, 'absEvent' : 3.41}
    sub_data = {'soft' : dict(), 'all' : dict(), 'simple' : dict(), 
                'verb' : dict(), 'dot' : dict(), 'noun_soft' : dict(), 
                'verb_soft' : dict(), 'stimuli' : dict(),
                'columns' : dict()}

    for run, infos in tqdm(runs):

        if map_nifti != None:
            masked_run = nilearn.masking.apply_mask(run, map_nifti).T
        else:
            masked_run = run.get_fdata()
            masked_run = masked_run.reshape(numpy.product(masked_run.shape[:3]), -1)

        for t_i, t in enumerate(infos['start']):
            full_cat = infos['category'][t_i]
            stimulus = infos['stimulus'][t_i]
            if full_cat in mapper['hard_labels'].keys():
                cat = mapper['hard_labels'][full_cat]
            else:
                cat = ''
            #if 'con' in full_cat or 'obj' in full_cat:
            #    cat = 'concrete'
            #elif 'abs' in full_cat or 'info' in full_cat:
            #    cat = 'abstract'
            #elif 'catch' in full_cat:
            #    cat = ''
            ### Not considering cases with no stimulus
            #if 'abstract' in cat or 'concrete' in cat:
            if cat != '':
                ### Columns labels
                col = str(mapper['column_labels'][full_cat])
                if col not in sub_data['columns'].keys():
                    sub_data['columns'][col] = dict()
                if stimulus not in sub_data['columns'][col].keys():
                    sub_data['columns'][col][stimulus] = list()
                ### Soft labels
                soft = str(mapper['soft_labels'][full_cat])
                if 'lunch' in args.dataset:
                    if 'Event' in full_cat:
                        soft = lunch_mapper[full_cat]
                if soft not in sub_data['soft'].keys():
                    sub_data['soft'][soft] = dict()
                if stimulus not in sub_data['soft'][soft].keys():
                    sub_data['soft'][soft][stimulus] = list()
                ### verb-only labels
                verb_soft = stimulus.split()[0]
                if verb_soft not in sub_data['verb_soft'].keys():
                    sub_data['verb_soft'][verb_soft] = dict()
                if stimulus not in sub_data['verb_soft'][verb_soft].keys():
                    sub_data['verb_soft'][verb_soft][stimulus] = list()
                ### noun-only labels
                noun_soft = stimulus.split()[-1]
                if noun_soft not in sub_data['noun_soft'].keys():
                    sub_data['noun_soft'][noun_soft] = dict()
                if stimulus not in sub_data['noun_soft'][noun_soft].keys():
                    sub_data['noun_soft'][noun_soft][stimulus] = list()
                ### stimuli labels
                if stimulus not in sub_data['stimuli'].keys():
                    sub_data['stimuli'][stimulus] = dict()
                if stimulus not in sub_data['stimuli'][stimulus].keys():
                    sub_data['stimuli'][stimulus][stimulus] = list()
                ### Generic abstract/concrete
                if cat not in sub_data['all'].keys():
                    sub_data['all'][cat] = dict()
                if stimulus not in sub_data['all'][cat].keys():
                    sub_data['all'][cat][stimulus] = list()
                ### Data subdivision
                if 'Coer' in full_cat:
                    full_cat = 'dot'
                elif 'Event' in full_cat:
                    full_cat = 'verb'
                else:
                    full_cat = 'simple'
                if cat not in sub_data[full_cat].keys():
                    sub_data[full_cat][cat] = dict()
                if stimulus not in sub_data[full_cat][cat].keys():
                    sub_data[full_cat][cat][stimulus] = list()

                if args.analysis == 'time_resolved':
                    fmri_timeseries = masked_run[:, t:t+18]
                elif args.analysis in ['whole_trial', 'flattened_trial']:
                    if 'slow' in args.dataset:
                        stim = 2
                        stim = 2
                    else:
                        stim = 1
                        stim = 1
                    ## Keeping responses from noun+4 to noun+4+4
                    beg = t + stim + 4
                    end = t + stim + 10
                    if args.analysis == 'whole_trial':
                        fmri_timeseries = numpy.average(masked_run[:, beg:end], \
                                                        axis=1)
                    elif args.analysis == 'flattened_trial':
                        fmri_timeseries = masked_run[:, beg:end].flatten()
                sub_data['soft'][soft][stimulus].append(fmri_timeseries)
                sub_data['columns'][col][stimulus].append(fmri_timeseries)
                sub_data['verb_soft'][verb_soft][stimulus].append(fmri_timeseries)
                sub_data['noun_soft'][noun_soft][stimulus].append(fmri_timeseries)
                sub_data['stimuli'][stimulus][stimulus].append(fmri_timeseries)
                sub_data['all'][cat][stimulus].append(fmri_timeseries)
                sub_data[full_cat][cat][stimulus].append(fmri_timeseries)

        del masked_run
    return sub_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['book_fast','lunch_fast', \
                                                         'book_slow', 'lunch_slow'],
                    help='Specify which dataset to use')
parser.add_argument('--analysis', required=True, \
                    choices=['time_resolved', 'whole_trial', 'flattened_trial'], \
                    help='Average time points, or run classification'
                         'time point by time point?')
parser.add_argument('--spatial_analysis', choices=['ROI'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--feature_selection', choices=['anova', 'no_reduction'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--cross_validation', choices=['individual_trials', 'average_trials', 'replication'], \
                    required=True, help = 'Specifies how features are to be selected')
parser.add_argument('--n_folds', type=int, default=50, \
                    help = 'Specifies how many folds to test on')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

os.makedirs('region_maps', exist_ok=True)
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'neuroscience', \
                        'dot_{}_bids'.format(args.dataset), 'derivatives',
                        )
n_subjects = len(os.listdir(dataset_path))
maps_folder = os.path.join('region_maps', 'maps')   
assert os.path.exists(maps_folder)
map_names = [n for n in os.listdir(maps_folder) 
             if 'hemisphere' not in n
             ]
map_results = dict()
os.makedirs('prova', exist_ok=True)

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

        file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_bold.nii'.format(s, args.dataset.replace('_', ''), r))

        single_run = nilearn.image.load_img(file_path)
               
        ### Cleaning run file: detrending and standardizing
        single_run = nilearn.image.clean_img(single_run)
        ### Smoothing 2*2*2mm
        #single_run = nilearn.image.smooth_img(single_run, 2)
        runs.append((single_run, trial_infos))

    if args.spatial_analysis == 'ROI':

        for map_name in map_names:

            map_path = os.path.join(maps_folder, map_name)
            assert os.path.exists(map_path)
            map_id = map_name.split('.')[0]
            logging.info('Masking area: {}'.format(map_id.replace('_', ' ')))
            map_nifti = nilearn.image.load_img(map_path)
            full_sub_data = load_subject_runs(runs, map_nifti=map_nifti)
            ### CHECK BEFORE USE!
            full_sub_data = {k : v for k, v in full_sub_data.items() 
                                  #if k in 
                                  if k not in 
                                  #['stimuli']
                                  [
                                  'stimuli', 
                                  #'soft', 
                                  'noun_soft', 
                                  #'verb_soft',
                                  #    'columns']
                                  #['columns'
                                  ]
                                  }

            for data_split, sub_data in full_sub_data.items():
                ### Averaging  presentations of the same stimulus
                print([len(v_two) for k, v in sub_data.items() for k_two, v_two in v.items()])
                if args.cross_validation == 'average_trials': 
                    jump = 6
                    sub_data = {k : [numpy.average(v_two[beg:beg+jump], axis=0) for k_two, v_two in v.items() for beg in range(0, len(v_two), jump)] for k, v in sub_data.items()}
                else:
                    n_repetitions = list(set([len(v) for k, v in sub_data.items()]))[0]
                    sub_data = {k : [v_three for k_two, v_two in v.items() for v_three in v_two] for k, v in sub_data.items()}
                counter = [len(v) for k, v in sub_data.items()]
                ### Shuffling
                sub_data = {k : random.sample(v, k=len(v)) for k, v in sub_data.items()}
                ### Matching number of images
                sub_data = {k : v[:min(counter)] for k, v in sub_data.items()}
                print([len(v) for k, v in sub_data.items()])

                if data_split in ['verb_soft', 'noun_soft']: 
                    labels = list()
                    samples = list()

                    for k, v in sub_data.items():
                        for vec in v:
                            labels.append(k)
                            samples.append(vec)
                else:
                    assert len(set([len(v) for k, v in sub_data.items()])) == 1

                    n_items = list(set([len(v) for k, v in sub_data.items()]))[0]

                    labels = list()
                    samples = list()

                    for i in range(n_items):
                        for k, v in sub_data.items():
                            if data_split == 'soft':
                                #k = float(k)
                                pass
                            labels.append(k)
                            samples.append(v[i])

                n_folds = 50
                #splits = list()
                #for _ in range(n_folds):
                #    splits.append(random.sample(set(range(len(samples))), k=int(len(samples)/10)))
                if 2 == 1:
                    pass
                else:
                    ### Averaged trials - all trials for one exemplar are averaged
                    ### Making sure there are at least 2 test items
                    if args.cross_validation == 'average_trials':
                        splits = list()
                        one_tenth = max(1, int((len(samples)/10)/len(set(labels))))
                        possible_labels = {k : [i_i for i_i, i in enumerate(labels) if i==k] for k in set(labels)}
                        #one_label = [i for i in range(0, n_items*2)][::2]
                        #two_label = [i for i in range(1, n_items*2)][::2]
                        for _ in range(n_folds):
                            split = list()
                            for current_labels in possible_labels.values(): 
                                split.extend(random.sample(current_labels, k=one_tenth))
                            if len(split) > (len(samples) / 10):
                                split = random.sample(split, 
                                                      k=int(len(samples)/10))
                            splits.append(split)
                        '''
                        if one_tenth > 2:
                            one_label = itertools.combinations(one_label, r=int(one_tenth/2))
                            two_label = itertools.combinations(two_label, r=int(one_tenth/2))
                            splits = itertools.product(one_label, two_label)
                            splits = [[k for c_two in c for k in c_two] for c in splits]
                        else:
                            splits = list(itertools.product(one_label, two_label))
                        '''
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
                #n_folds = min(len(splits), args.n_folds)
                if args.n_folds == 1000 and n_folds < 100:
                    print('Does not make sense to run this one')
                    continue
                splits = random.sample(splits, k=n_folds)
                print('Current split: {} - number of folds: {}'.format(data_split, n_folds)) 
                sub_scores = list()
                logging.info('Running cross-validation')
                for split in tqdm(splits):
                    #train_samples = samples[0:split] + samples[split+one_tenth:]
                    #train_labels = labels[0:split] + labels[split+one_tenth:]
                    #test_samples = samples[split:split+one_tenth]
                    #test_labels = labels[split:split+one_tenth]
                    #train_samples = [samples[i] for i in range(n_items*2) if i not in split]
                    train_samples = [samples[i] for i in range(len(samples)) if i not in split]
                    #train_labels = [labels[i] for i in range(n_items*2) if i not in split]
                    train_labels = [labels[i] for i in range(len(samples)) if i not in split]
                    test_samples = [samples[i] for i in split]
                    test_labels = [labels[i] for i in split]

                    model = 'ridge'
                    if model == 'svc':
                        classifier = SVC(gamma='auto')
                    elif model == 'ridge':
                        classifier = linear_model.RidgeClassifier()
                    split_score = list()
                    if args.analysis == 'time_resolved':
                        for t in range(18):
                            t_train = [sample[:, t] for sample in train_samples]
                            t_test = [sample[:, t] for sample in test_samples]
                            try:
                                classifier.fit(t_train, train_labels)
                            except ValueError:
                                import pdb; pdb.set_trace()
                            acc = classifier.score(t_test, test_labels)
                            split_score.append(acc)

                    elif args.analysis in ['whole_trial', 'flattened_trial']:

                        ### Feature selection
                        ## Anova
                        if args.feature_selection == 'anova':
                            selector = sklearn.feature_selection.SelectPercentile(\
                                                score_func=sklearn.feature_selection.f_classif, \
                                                percentile=50).fit(train_samples, train_labels)
                            train_samples = selector.transform(train_samples)
                            test_samples = selector.transform(test_samples)

                        try:
                            classifier.fit(train_samples, train_labels)
                        except ValueError:
                            import pdb; pdb.set_trace()
                        try:
                            acc = classifier.score(test_samples, test_labels)
                            split_score.append(acc)
                        except ValueError:
                            import pdb; pdb.set_trace()

                    sub_scores.append(split_score)

                average_scores = numpy.average(sub_scores, axis=0)
                logging.info('Average scores for area {}: {}'.format(map_id, average_scores))

                if map_id not in map_results.keys():
                    map_results[map_id] = dict()

                if data_split not in map_results[map_id].keys():
                    map_results[map_id][data_split] = [average_scores]
                else:
                    map_results[map_id][data_split].append(average_scores)

output_folder = os.path.join('results', '{}_{}_{}'.format(args.cross_validation, jump, model), 
                             'simple_classification', \
                             args.dataset, args.analysis, 
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
