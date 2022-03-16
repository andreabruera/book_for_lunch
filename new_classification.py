import argparse
import collections
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
from scipy import spatial, stats
from sklearn import feature_selection
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVC
from tqdm import tqdm

def read_vectors(args):

    vectors = dict()
    for f in os.listdir(args.vectors_folder):
        assert '.vector' in f
        with open(os.path.join(args.vectors_folder, f)) as i:
            if 'selected' in args.vectors_folder:
                vecs = numpy.array([l.strip() for l in i.readlines()], dtype=numpy.float64)
            else:
                vecs = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float64)
        if vecs.shape[0] == 0:
            print(f)
            continue
        else:
            ### Randomizing vector order
            numpy.random.shuffle(vecs)
            ### Limiting to 32 mentions
            vecs = vecs[:32, :]
            if vecs.shape[0] not in [768, 300]:
                if args.vector_averaging == 'avg':
                    ### Average
                    vecs = numpy.nanmean(vecs, axis=0)
                else:
                    ### Maxpool
                    vecs = numpy.array([max([v[i] for v in vecs]) for i in range(vecs.shape[-1])], dtype=numpy.float64)
            assert vecs.shape[0] in [768, 300]
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
        if cat in ['obj', 'info']:
            if cat == 'obj':
                cat = 'simple_concrete'
            else:
                cat = 'simple_abstract'
        elif 'Event' in cat:
            if 'con' in cat:
                cat = 'verb_concrete'
            else:
                cat = 'verb_abstract'
        elif 'Coercion' in cat:
            if 'con' in cat:
                cat = 'dot_concrete'
            else:
                cat = 'dot_abstract'
        stimulus = events['trial_type'][t:t+jump]
        stimulus = '{} {}'.format(stimulus[verb_idx], stimulus[noun_idx])

        trial_infos['start'].append(trial_starts[t_i])
        trial_infos['category'].append(cat)
        trial_infos['stimulus'].append(stimulus)
    return trial_infos

def load_subject_runs(runs, map_nifti=None):

    sub_data = dict()

    for run, infos in tqdm(runs):

        if map_nifti == None:
            map_nifti = nilearn.masking.compute_brain_mask(run)
        #if map_nifti != None:
        #else:
        #    masked_run = run.get_fdata()
        #    masked_run = masked_run.reshape(numpy.product(masked_run.shape[:3]), -1)
        masked_run = nilearn.masking.apply_mask(run, map_nifti).T

        for t_i, t in enumerate(infos['start']):
            stimulus = infos['stimulus'][t_i]

            if args.analysis == 'time_resolved':
                fmri_timeseries = masked_run[:, t:t+18]
            elif 'whole_trial' in args.analysis:
                beg = 4
                end = 11
                t_one = t + beg
                t_two = t + end
                if 'slow' in args.dataset:
                    t_one += 1
                    t_two += 1
                if 'flattened' in args.analysis:
                    fmri_timeseries = masked_run[:, t_one:t_two].flatten()
                ### Keeping responses from noun+4 to noun+9
                else:
                    fmri_timeseries = numpy.average(
                                       masked_run[:, t_one:t_two],
                                       axis=1)
            if stimulus not in sub_data.keys():
                sub_data[stimulus] = [fmri_timeseries]
            else:
                sub_data[stimulus].append(fmri_timeseries)

        del masked_run
    return sub_data, beg, end

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['book_fast','lunch_fast', \
                                                         'book_slow', 'lunch_slow'],
                    help='Specify which dataset to use')
parser.add_argument('--analysis', required=True, \
                    choices=['time_resolved', 'whole_trial', 
                             'whole_trial_flattened'], \
                    help='Average time points, or run classification'
                         'time point by time point?')
parser.add_argument('--spatial_analysis', choices=['ROI', 'all', 'language_areas'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--feature_selection', choices=['anova', 'no_reduction'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--cross_validation', choices=['individual_trials', 'average_trials', 'replication'], \
                    required=True, help = 'Specifies how features are to be selected')
parser.add_argument('--target', 
                    choices=['class', 'continuous'], \
                    required=True, help = 'What to predict?')
parser.add_argument('--n_folds', type=int, default=1000, \
                    help = 'Specifies how many folds to test on')
parser.add_argument('--rsa', action='store_true', default=False, \
                    help = 'Specifies whether to run the RSA analysis')
parser.add_argument('--only_nouns', action='store_true', default=False, \
                    help = 'Using as targets only the vectors for the nouns?')
parser.add_argument('--vectors_folder', type=str, required=True, \
                    help = 'Specifies where the vectors are stored')
parser.add_argument('--n_brain_features', type=int, required=True, \
                    help = 'How many brain features to use?')
parser.add_argument('--vector_averaging', choices=['avg', 'maxpool'], 
                    required=True, help='How to aggregate across mentions?')

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

vectors = read_vectors(args)
decoding_results = list()

all_results = collections.defaultdict(list)
trials = dict()
for s in range(1, n_subjects+1):
#for s in range(1, 3):
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
        with open('{}_stimuli.tsv'.format(args.dataset), 'w') as o:
            for v, k in zip(trial_infos['category'], trial_infos['stimulus']):
                ke = k.replace("'", ' ')
                ke = ke.split()
                new_k = '{} {}'.format(ke[0], ke[2]) if len(ke)==3 else ' '.join(ke)
                if new_k not in trials.keys():
                    trials[new_k] = v
                if 'neg neg' not in k:
                    o.write('{}\t{}\n'.format(k, v))

        ### Reading fMRI run
        file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_bold.nii'.format(s, args.dataset.replace('_', ''), r))

        single_run = nilearn.image.load_img(file_path)
        ### Cleaning run file: detrending and standardizing
        single_run = nilearn.image.clean_img(single_run)
        runs.append((single_run, trial_infos))

    sub_results = collections.defaultdict(list)
    logging.info('Masking using Fisher feature selection')
    #raise RuntimeError('Part to be implemented')
    ### Left hemisphere
    #map_nifti = nilearn.image.load_img('region_maps/maps/left_hemisphere.nii')
    if args.spatial_analysis == 'language_areas':
        map_path = os.path.join(maps_folder, 'language_areas.nii')
        assert os.path.exists(map_path)
        logging.info('Masking language areas...')
        map_nifti = nilearn.image.load_img(map_path)
    else:
        map_nifti = None

    full_sub_data, beg, end = load_subject_runs(runs, map_nifti)
    ### Averaging, keeping only one response per stimulus
    sub_data = {k : numpy.average(v, axis=0) for k, v in full_sub_data.items()}
    dimensionality = list(set([v.shape[0] for k, v in sub_data.items()]))[0]
    ### Extracting 5k random indices
    #random_indices = random.sample(list(range(dimensionality)), k=10000)
    #sub_data = {k : v[random_indices] for k, v in sub_data.items()}
    with open(os.path.join('voxel_selection',
                           'fisher_scores', 
                           '{}_to_{}'.format(beg, end),
                           args.dataset, args.analysis, args.spatial_analysis, 
                           'sub-{:02}.fisher'.format(s))) as i:
        lines = numpy.array([l.strip().split('\t') for l in i.readlines()][0], dtype=numpy.float64)
    assert len(lines) == dimensionality
    sorted_dims = sorted(list(enumerate(lines)), key=lambda item : item[1], reverse=True)
    n_dims = args.n_brain_features
    selected_dims = [k[0] for k in sorted_dims[:n_dims]]
    sub_data = {k : v[selected_dims] for k, v in sub_data.items()}

    ### Reduce keys to actually present
    ### Correcting vectors and brain data keys
    sub_data_keys = {tuple(k.replace("'", ' ').split()) : k  for k in sub_data.keys()}
    sub_data_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in sub_data_keys.items()}
    sub_data = {k : sub_data[v] for k, v in sub_data_keys.items()}
    vectors_keys = {tuple(k.replace("_", ' ').split()) : k  for k in vectors.keys()}
    vectors_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in vectors_keys.items()}
    vectors = {k : vectors[v] for k, v in vectors_keys.items()}
    ### Limiting
    ### Taking away 'unclear' samples
    '''
    exclude = ['avere tavolo', 'cambiare storia', 
               'regalare fiore', 'presentare programma', 
               'raccogliere catalogo', 'consultare libro']
    vectors = {k : v for k, v in vectors.items() if k not in exclude}
    if args.only_nouns:
        ### Collapsing to word
        word_vec = {k.split()[-1] : list() for k in vectors.keys()}
        for k, v in vectors.items():
            word_vec[k.split()[-1]].append(v)
        word_vec = {k : numpy.average(v, axis=0) for k, v in word_vec.items()}
        vectors = {k : word_vec[k.split()[-1]] for k in vectors.keys()}
    '''
    if args.only_nouns:
        ### Collapsing to word
        word_vec = {k.split()[0] : list() for k in vectors.keys()}
        for k, v in vectors.items():
            word_vec[k.split()[0]].append(v)
        word_vec = {k : numpy.average(v, axis=0) for k, v in word_vec.items()}
        vectors = {k : word_vec[k.split()[0]] for k in vectors.keys()}

    sub_data = {k : v for k, v in sub_data.items() if k in vectors.keys()}
    vectors = {k : vectors[k] for k, v in sub_data.items()}
    combs = list(itertools.combinations(list(vectors.keys()), 2))

    ### Prepare trial info data
    if args.target == 'continuous':
        with open('book_fast_stimuli_ratings.tsv') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        cl_con = {l[0] : float(l[4]) for l in lines[1:]}
        combs = [[c] for c in cl_con.keys()]
        '''
        conc = [abs(cl_con[c[0]]-cl_con[c[1]]) for c in combs]
        data = [1-scipy.stats.spearmanr(sub_data[c[0]], sub_data[c[1]])[0] for c in combs]
        vecs = [1-scipy.stats.spearmanr(vectors[c[0]], vectors[c[1]])[0] for c in combs]
        print('Correlation brain-vectors: {}'.format(scipy.stats.spearmanr(vecs, data)))
        accuracy = scipy.stats.spearmanr(conc, data)[0]
        '''

    elif args.target == 'class':
        cl_con = {tuple(n.replace("'", ' ').split()) : c.split('_')[1] for n, c in zip(trial_infos['stimulus'], trial_infos['category']) if 'neg' not in n}
        cl_con = {' '.join([n[0], n[-1]]) : c for n, c in cl_con.items()}
        '''
        dot = {tuple(n.replace("'", ' ').split()) : c.split('_')[0] for n, c in zip(trial_infos['stimulus'], trial_infos['category']) if 'neg' not in n}
        dot = [' '.join([n[0], n[-1]]) for n, c in dot.items() if c=='dot']
        '''
        combs = [c for c in combs if c[0]!=c[1]\
                 #and (c[0] in dot and c[1] in dot)
                 ]

    ### RSA decoding
    if args.rsa:
        logging.info('Computing RSA vectors...')
        sub_data = {k : [scipy.stats.spearmanr(v, v_two)[0] for k_two, v_two in sub_data.items() if k!=k_two] for k, v in sub_data.items()}
        vectors = {k : [scipy.stats.spearmanr(v, v_two)[0] for k_two, v_two in vectors.items() if k!=k_two] for k, v in vectors.items()}
    accuracies = list()
    ### Splitting
    for c in tqdm(combs):
        test_targets = [cl_con[c_i] for c_i in c]
        train_inputs = [v for k, v in sub_data.items() if k not in c]
        #train_targets = [v for k, v in vectors.items() if k not in c]
        train_targets = [cl_con[k] for k, v in vectors.items() if k not in c]

        test_inputs = [sub_data[c_i] for c_i in c]
        #test_targets = [vectors[c_i] for c_i in c]

        if args.target == 'class':
            model = RidgeClassifier(alpha=1.0)
            #model = SVC()
            model.fit(train_inputs, train_targets)
            score = model.score(test_inputs, test_targets)
        elif args.target == 'continuous':
            model = Ridge(alpha=1.0)
            model.fit(train_inputs, train_targets)
            #score = model.score(test_inputs, test_targets)
            score = model.predict(test_inputs)
        accuracies.append(score)

                #assert len(predictions) == len(test_targets)
    if args.target == 'class':
        accuracy = numpy.average(accuracies)
    elif args.target == 'continuous':
        original_values = list(cl_con.values())
        accuracy = scipy.stats.spearmanr(original_values, accuracies)[0]
    decoding_results.append(accuracy)
    print(accuracy)
    if args.target == 'class':
        for c, a in zip(combs, accuracies):
            sub_results[c].append(a)
        ### Preparing per-category results
        clean_results = collections.defaultdict(list)
        for k, v in sub_results.items():
            if k[0] not in trials.keys() or k[1] not in trials.keys():
                print(k)
            else:
                if trials[k[0]] == trials[k[1]]:
                    clean_results[trials[k[0]]].extend(v)
                else:
                    new_k = '_'.join(sorted([trials[k[0]], trials[k[1]]]))
                    clean_results[new_k].extend(v)
                ### Abstract / concrete
                abs_con_one = trials[k[0]].split('_')[-1]
                abs_con_two = trials[k[1]].split('_')[-1]
                abs_con = '_'.join(sorted(list(set([abs_con_one, abs_con_two]))))
                clean_results[abs_con].extend(v)
                ### First
                abs_con_one = trials[k[0]].split('_')[0]
                abs_con_two = trials[k[1]].split('_')[0]
                abs_con = '_'.join(sorted(list(set([abs_con_one, abs_con_two]))))
                clean_results[abs_con].extend(v)

        for k, v in clean_results.items():
            all_results[k].append(numpy.average(v))


output_folder = os.path.join('results', args.cross_validation, 
        #'{}_{}_simple_decoding_rsa_{}_only_nouns_{}'.format(
        '{}_{}_new_classification_rsa_{}_target_{}'.format(
                             n_dims, args.vectors_folder.split('/')[1], args.rsa, args.target), \
                             args.dataset, args.analysis, 
                             'pairwise', 
                             args.spatial_analysis,
                             args.feature_selection, args.vector_averaging,
                             )
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, 'accuracies.results'), 'w') as o:
    o.write('overall_accuracy\t')
    for k, v in all_results.items():
        o.write('{}\t'.format(k))
    o.write('\n')
    for acc_i, acc in enumerate(decoding_results):
        o.write('{}\t'.format(acc))
        for k, v in all_results.items():
            o.write('{}\t'.format(v[acc_i]))
        o.write('\n')
