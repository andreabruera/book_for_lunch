import argparse
import collections
import itertools
import nibabel
import nilearn
import numpy
import logging
import os
import pandas
import random
import scipy
import sklearn

from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting
from nilearn.glm.first_level import FirstLevelModel
from scipy import spatial, stats
from sklearn import decomposition, feature_selection
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from tqdm import tqdm

from utils import read_vectors

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
    trial_infos = {'onset' : list(), 
                   'trial_type' : list(), 
                   'duration' : list(),
                   'category' : list()}
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
        elif 'Coer' in cat:
            if 'con' in cat:
                cat = 'dot_concrete'
            else:
                cat = 'dot_abstract'
        elif 'Simple' in cat:
            if 'con' in cat:
                cat = 'simple_concrete'
            else:
                cat = 'simple_abstract'
        stimulus = events['trial_type'][t:t+jump]
        stimulus = '{} {}'.format(stimulus[verb_idx], stimulus[noun_idx])

        trial_infos['onset'].append(trial_starts[t_i])
        trial_infos['duration'].append(1.)
        trial_infos['category'].append(cat)
        trial_infos['trial_type'].append(stimulus)
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

        for t_i, t in enumerate(infos['onset']):
            stimulus = infos['trial_type'][t_i]

            if args.analysis == 'time_resolved':
                fmri_timeseries = masked_run[:, t:t+18]
            elif 'whole_trial' in args.analysis or args.analysis == 'glm':
                beg = 4
                end = 11
                t_one = t + beg
                t_two = t + end
                if 'slow' in args.dataset:
                    t_one += 1
                    t_two += 1
                ### Reducing number of fmri images when using senses, to avoid imbalance
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
                             'glm', 'whole_trial_flattened'], \
                    help='Average time points, or run classification'
                         'time point by time point?')
parser.add_argument('--spatial_analysis', choices=[ 
                                                    'whole_brain', 
                                                    'fedorenko_language', 
                                                    'control_semantics', 
                                                    'general_semantics'], 
                                                    required=True, 
                    help = 'Specifies how features are to be selected')
parser.add_argument('--feature_selection', choices=['fisher', 'stability'], required=True, \
                    help = 'Specifies how features are to be selected')
parser.add_argument('--n_folds', type=int, default=1000, \
                    help = 'Specifies how many folds to test on')
parser.add_argument('--method', choices=['encoding', 'decoding', 'rsa_encoding', 'rsa_decoding',
                    required=True,
                    help = 'Encoding instead of decoding?')
parser.add_argument('--ceiling', action='store_true', default=False, \
                    help = 'Ceiling instead of word vectors?')
parser.add_argument('--computational_model', type=str, required=True, \
                                         choices=[
                                         'fasttext', 'gpt2',
                                         'familiarity', 
                                         #'vector_familiarity', 
                                         'concreteness', 
                                         #'vector_concreteness', 
                                         'frequency', 
                                         #'vector_frequency',
                                         'imageability', 
                                         #'vector_imageability'
                                         ], 
                    help = 'Specifies where the vectors are stored')
parser.add_argument('--n_brain_features', type=int, required=True, \
                    help = 'How many brain features to use?')
parser.add_argument('--senses', action='store_true', default=False, \
                    help = 'Averaging different senses together?')
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

if args.computational_model in ['gpt2', 'fasttext']:
    vectors = read_vectors(args)
else:
    ### Concreteness or other variables
    with open('{}_stimuli_ratings.tsv'.format(args.dataset)) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    if 'concreteness' in args.computational_model:
        model_data = {l[0] : float(l[4]) for l in lines[1:]}
    elif 'familiarity' in args.computational_model:
        model_data = {l[0] : float(l[2]) for l in lines[1:]}
    elif 'imageability' in args.computational_model:
        model_data = {l[0] : float(l[6]) for l in lines[1:]}
    elif 'frequency' in args.computational_model:
        model_data = dict()
        for l in lines[1:]:
            with open(os.path.join('resources', 'book_for_lunch_sentences',
                '{}.vector'.format(l[0].replace(' ', '_')))) as i:
                length = len(list(i.readlines()))
            model_data[l[0]] = numpy.log(length)

### Ceiling
if args.ceiling:
    ceiling_data = dict()
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

            ### Reading fMRI run
            file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dot{}_run-{:02}_bold.nii'.format(s, args.dataset.replace('_', ''), r))

            single_run = nilearn.image.load_img(file_path)
            ### Cleaning run file: detrending and standardizing
            single_run = nilearn.image.clean_img(single_run)
            runs.append((single_run, trial_infos))

        ### GLM
        if args.analysis == 'glm':
            raise RuntimeError('Still to implement')

        else:
            full_sub_data, beg, end = load_subject_runs(runs, None)
            ### Averaging, keeping only one response per stimulus
            if args.senses:
                mapper = {tuple(s.replace("'", ' ').replace(' ', '_').split('_')) : '_'.join([s.split(' ')[-1], trial_infos['category'][s_i]]) for s_i, s in enumerate(trial_infos['trial_type'])}
                senses = {v : [k for k, v_two in mapper.items() if v_two==v] for v in mapper.values()}
                senses = {k : [' '.join([v_two[i] for i in [0, -1]]) for v_two in v] for k, v in senses.items()}
                senses = {k : v for k, v in senses.items() if 'neg' not in k}
                for sense, stimuli in senses.items():
                    if len(stimuli) > 1:
                        k = 2 if len(stimuli)==3 else 3
                        random_indices = random.choices(range(5), k=k)
                        for sense_stim in stimuli:
                            full_sub_data[sense_stim] = numpy.array(full_sub_data[sense_stim])[random_indices]
            sub_data = {k : numpy.average(v, axis=0) for k, v in full_sub_data.items()}
        ceiling_data[s] = sub_data


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
            for v, k in zip(trial_infos['category'], trial_infos['trial_type']):
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
        raise RuntimeError('ROI removed!')
        map_path = os.path.join(maps_folder, 'language_areas.nii')
        assert os.path.exists(map_path)
        logging.info('Masking language areas...')
        map_nifti = nilearn.image.load_img(map_path)
    elif args.spatial_analysis == 'general_semantics':
        map_path = os.path.join(maps_folder, 'General_semantic_cognition_ALE_result.nii')
        assert os.path.exists(map_path)
        logging.info('Masking general semantics areas...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        map_nifti = nilearn.image.resample_to_img(map_nifti, single_run, interpolation='nearest')
    elif args.spatial_analysis == 'control_semantics':
        map_path = os.path.join(maps_folder, 'semantic_control_ALE_result.nii')
        assert os.path.exists(map_path)
        logging.info('Masking control semantics areas...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        map_nifti = nilearn.image.resample_to_img(map_nifti, single_run, interpolation='nearest')
    elif args.spatial_analysis == 'fedorenko_language':
        map_path = os.path.join(maps_folder, 'allParcels_language_SN220.nii')
        assert os.path.exists(map_path)
        logging.info('Masking Fedorenko lab\'s language areas...')
        map_nifti = nilearn.image.load_img(map_path)
        map_nifti = nilearn.image.binarize_img(map_nifti, threshold=0.)
        map_nifti = nilearn.image.resample_to_img(map_nifti, single_run, interpolation='nearest')
    else:
        #map_nifti = None
        map_nifti = nilearn.masking.compute_brain_mask(runs[0][0])
    ### GLM
    if args.analysis == 'glm':
        logging.info('Now estimating BOLD responses using GLM '\
                     'for subject {}'.format(s))

        fmri_glm = FirstLevelModel(t_r=1, 
                                   standardize=False, 
                                   signal_scaling=False,
                                   smoothing_fwhm=None, 
                                   mask_img=map_nifti)

        glm_events = [pandas.DataFrame.from_dict(r[1]) for r in runs]

        fmri_glm = fmri_glm.fit([r[0] for r in runs], glm_events)

        full_sub_data = {k : fmri_glm.compute_contrast(k) for k in set(glm_events[0]['trial_type']) if 'neg' not in k}
        full_sub_data = {k : nilearn.masking.apply_mask(v, map_nifti).T for k, v in full_sub_data.items()}
        beg = 4
        end = 11

    else:

        full_sub_data, beg, end = load_subject_runs(runs, map_nifti)

    ### Reduce keys to actually present
    ### Correcting vectors and brain data keys
    sub_data_keys = {tuple(k.replace("'", ' ').split()) : k  for k in full_sub_data.keys()}
    sub_data_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in sub_data_keys.items()}
    full_sub_data = {k : full_sub_data[v] for k, v in sub_data_keys.items() if 'neg' not in k}
    ### Pairwise similarities for concreteness etc
    if args.computational_model not in ['fasttext', 'gpt2']:
        vectors = {k : [abs(model_data[k]-model_data[k_two]) for k_two in full_sub_data.keys() if k_two!=k] for k in full_sub_data.keys()}
    ### Ceiling
    elif args.computational_model == 'ceiling':
        current_ceiling = {k : [ceiling_data[sub][k] for sub in range(1, len(ceiling_data.keys())+1) if sub!=s] for k in ceiling_data[s].keys()}
        current_ceiling = {k : numpy.average(v, axis=0) for k, v in current_ceiling.items()}
        current_ceiling = {k : v[selected_dims] for k, v in current_ceiling.items()}
        ceiling_keys = {tuple(k.replace("'", ' ').split()) : k  for k in current_ceiling.keys()}
        ceiling_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in ceiling_keys.items()}
        vectors = {k : current_ceiling[v] for k, v in ceiling_keys.items() if 'neg' not in k}

    vectors_keys = {tuple(k.replace("_", ' ').split()) : k  for k in vectors.keys()}
    vectors_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in vectors_keys.items()}
    vectors = {k : vectors[v] for k, v in vectors_keys.items()}
    full_sub_data = {k : v for k, v in full_sub_data.items() if k in vectors.keys()}
    vectors = {k : vectors[k] for k, v in full_sub_data.items()}

    ### Balancing number of trials in the case of senses
    sub_data = dict()
    if args.senses:
        for sense, stimuli in senses.items():
            if len(stimuli) > 1:
                k = 2 if len(stimuli)==3 else 3
                random_indices = random.choices(range(5), k=k)
                for sense_stim in stimuli:
                    sub_data[sense_stim] = numpy.array(full_sub_data[sense_stim])[random_indices]
    sub_data = {k : numpy.average(v, axis=0) if len(v)<10 else v for k, v in full_sub_data.items()}
    print([len(v) for v in sub_data.values()])
    dimensionality = list(set([v.shape[0] for k, v in sub_data.items()]))[0]
    ### Extracting 5k random indices
    #random_indices = random.sample(list(range(dimensionality)), k=10000)
    #sub_data = {k : v[random_indices] for k, v in sub_data.items()}
    feature_analysis = 'whole_trial' if args.analysis == 'glm' else args.analysis
    feature_folder = os.path.join('voxel_selection',
                           '{}_scores'.format(args.feature_selection), 
                           '{}_to_{}'.format(beg, end),
                           args.dataset, feature_analysis, args.spatial_analysis, 
                           'sub-{:02}.{}'.format(s, args.feature_selection))
    with open(feature_folder) as i:
        lines = numpy.array([l.strip().split('\t') for l in i.readlines()][0], dtype=numpy.float64)
    assert len(lines) == dimensionality
    sorted_dims = sorted(list(enumerate(lines)), key=lambda item : item[1], reverse=True)
    n_dims = args.n_brain_features
    selected_dims = [k[0] for k in sorted_dims[:n_dims]]
    sub_data = {k : v[selected_dims] for k, v in sub_data.items()}

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
    if args.only_nouns:
        ### Collapsing to word
        word_vec = {k.split()[0] : list() for k in vectors.keys()}
        for k, v in vectors.items():
            word_vec[k.split()[0]].append(v)
        word_vec = {k : numpy.average(v, axis=0) for k, v in word_vec.items()}
        vectors = {k : word_vec[k.split()[0]] for k in vectors.keys()}

    ### TODO: implement word vector to norm decoding
    ### Use trial infos
    if 'vector' in args.computational_model:
        sub_data = vectors.copy()
        vectors = {k : v for k, v in model_vecs.items()}
    '''

    accuracies = list()

    if args.senses:
        ### Collapsing senses for words if they are abstract/concrete
        
        actual_vectors = dict()
        actual_brain = dict()

        for sense, stimuli in senses.items():
            if len(stimuli) == 1:
                actual_brain[stimuli[0]] = sub_data[stimuli[0]]
                actual_vectors[stimuli[0]] = vectors[stimuli[0]]
            else:
                ### Averaging brain responses
                data_avg = numpy.average([sub_data[s] for s in stimuli], axis=0)
                actual_brain[stimuli[0]] = data_avg

                vec_avg = numpy.average([vectors[s] for s in stimuli], axis=0)
                actual_vectors[stimuli[0]] = vec_avg
    else:
        actual_brain = sub_data.copy()
        actual_vectors = vectors.copy()
    ### Splitting
    #sub_data = {k : v for k, v in sub_data.items() if k in vectors.keys()}
    #actual_brain = {k : v for k, v in actual_brain.items() if k in actual_vectors.keys()}
    #actual_vectors = {k : actual_vectors[k] for k, v in actual_brain.items()}
    ### RSA decoding
    if args.rsa:
        logging.info('Computing RSA vectors...')
        actual_brain = {k : [scipy.stats.pearsonr(v, v_two)[0] for k_two, v_two in actual_brain.items() if k!=k_two] for k, v in actual_brain.items()}
        actual_vectors = {k : [scipy.stats.pearsonr(v, v_two)[0] for k_two, v_two in actual_vectors.items() if k!=k_two] for k, v in actual_vectors.items()}
    ### TODO: implement pairwise word vectors
    '''
    if args.computational_model == 'pairwise_word_vectors':
        actual_vectors = {k : [scipy.stats.spearmanr(v, v_two)[0] for k_two, v_two in actual_vectors.items() if k!=k_two] for k, v in actual_vectors.items()}
    '''
    #vectors = {k : vectors[k] for k, v in sub_data.items()}
    ### Standardization
    #standardized_input = sklearn.preprocessing.StandardScaler().fit_transform(numpy.array(list(actual_brain.values())))
    numpy_target = numpy.array(list(actual_vectors.values()))
    if numpy_target.shape == (42, ):
        numpy_target = numpy_target.reshape(-1, 1)
        standardized_target = sklearn.preprocessing.StandardScaler().fit_transform(numpy_target)
        actual_vectors = {k : v for k, v in zip(actual_vectors.keys(), standardized_target.reshape((42,)))}
    else:
        standardized_target = sklearn.preprocessing.StandardScaler().fit_transform(numpy_target)
        actual_vectors = {k : v for k, v in zip(actual_vectors.keys(), standardized_target)}
    #actual_brain = {k : v for k, v in zip(actual_brain.keys(), standardized_input)}
    #    current_ceiling = {k : v for k, v in current_ceiling.items() if k in vectors.keys()}
    #    #current_ceiling = {k : v for k, v in current_ceiling.items() if k in actual_vectors.keys()}
    '''
    ### PCA reduction of word vectors
    if args.encoding:
        vecs = sklearn.decomposition.PCA(n_components=.7).fit_transform(list(vectors.values()))
        vectors = {k : v for k, v in zip(vectors.keys(), vecs)}
    '''
    combs = list(itertools.combinations(list(actual_vectors.keys()), 2))
    for c in tqdm(combs):
        ### Encoding
        if 'encoding' in args.method:
            ### PCA reduction of vectors
            train_inputs = [v for k, v in actual_vectors.items() if k not in c]
            train_targets = [v for k, v in actual_brain.items() if k not in c]

            test_inputs = [actual_vectors[c_i] for c_i in c]
            test_targets = [actual_brain[c_i] for c_i in c]
        ### Decoding
        else:
            '''
            if args.ceiling:
                train_inputs = [v for k, v in actual_brain.items() if k not in c]
                train_targets = [v for k, v in current_ceiling.items() if k not in c]

                test_inputs = [actual_brain[c_i] for c_i in c]
                test_targets = [current_ceiling[c_i] for c_i in c]
            else:
            '''
            train_inputs = [v for k, v in actual_brain.items() if k not in c]
            train_targets = [v for k, v in actual_vectors.items() if k not in c]

            test_inputs = [actual_brain[c_i] for c_i in c]
            test_targets = [actual_vectors[c_i] for c_i in c]

            ### RSA
            if 'rsa' in args.method:
                wrong = 0.
                for idx_one, idx_two in [(0, 1), (1, 0)]:
                    wrong += scipy.stats.pearsonr(test_inputs[idx_one], test_targets[idx_two])[0]
                correct = 0.
                for idx_one, idx_two in [(0, 0), (1, 1)]:
                    correct += scipy.stats.pearsonr(test_inputs[idx_one], test_targets[idx_two])[0]
                if correct > wrong:
                    accuracies.append(1)
                else:
                    accuracies.append(0)
            else:
                ### Ridge
                model = Ridge(alpha=1.0)
                model.fit(train_inputs, train_targets)

                predictions = model.predict(test_inputs)
                assert len(predictions) == len(test_targets)
                wrong = 0.
                for idx_one, idx_two in [(0, 1), (1, 0)]:
                    wrong += scipy.stats.pearsonr(predictions[idx_one], test_targets[idx_two])[0]
                    '''
                    if args.computational_model == 'word_vectors':
                        wrong += scipy.stats.spearmanr(predictions[idx_one], test_targets[idx_two])[0]
                    else:
                        wrong -= abs(predictions[idx_one] - test_targets[idx_two])
                    '''
                correct = 0.
                for idx_one, idx_two in [(0, 0), (1, 1)]:
                    correct += scipy.stats.pearsonr(predictions[idx_one], test_targets[idx_two])[0]
                    '''
                    if args.computational_model == 'word_vectors':
                        correct += scipy.stats.spearmanr(predictions[idx_one], test_targets[idx_two])[0]
                    else:
                        correct -= abs(predictions[idx_one] - test_targets[idx_two])
                    '''
                #import pdb; pdb.set_trace()
                if correct > wrong:
                    accuracies.append(1)
                else:
                    accuracies.append(0)

    accuracy = numpy.average(accuracies)
    print(accuracy)
    for c, a in zip(combs, accuracies):
        sub_results[c].append(a)
    ### Preparing per-category results
    clean_results = collections.defaultdict(list)
    for k, v in sub_results.items():
        ### k is a tuple containing the stimuli couple
        if k[0] not in trials.keys() or k[1] not in trials.keys():
            print(k)
        else:
            if trials[k[0]] == trials[k[1]]:
                clean_results[trials[k[0]]].extend(v)
            else:
                new_k = '_'.join(sorted([trials[k[0]], trials[k[1]]]))
                clean_results[new_k].extend(v)
                ### Book-type
                if 'libro' in k[0] and 'libro' in k[1]:
                    clean_results['book'].extend(v)
                if 'catalogo' in k[0] and 'catalogo' in k[1]:
                    clean_results['catalogue'].extend(v)
                if 'rivista' in k[0] and 'rivista' in k[1]:
                    clean_results['magazine'].extend(v)
                if 'disegno' in k[0] and 'disegno' in k[1]:
                    clean_results['drawing'].extend(v)

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
    #print(clean_results.keys())

    for k, v in clean_results.items():
        all_results[k].append(numpy.average(v))
    decoding_results.append(accuracy)

output_folder = os.path.join(
                             'results',
                             'vector_{}'.format(args.method),
                             args.analysis, 
                             'senses_{}'.format(args.senses)
                             '{}_{}'.format(args.feature_selection, n_dims), 
                             args.computational_model, 
                             args.spatial_analysis,
                             args.dataset, 
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
