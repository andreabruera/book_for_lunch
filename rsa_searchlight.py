import argparse
import collections
import itertools
import multiprocessing
import nibabel
import nilearn
import numpy
import logging
import os
import random
import scipy
import sklearn

from scipy import stats
from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from scipy import spatial, stats
from tqdm import tqdm

from utils import read_vectors

def compute_corr(all_args):
    indices = all_args[0]
    sub_data = all_args[1]
    vec_sims = all_args[2]
    voxel_idx = all_args[3]
    #indices = adj_matrix[voxel_idx].nonzero()[1]
    current_sub_data = {k : v[indices] for k, v in sub_data.items()}
    #brain_sims = [scipy.stats.spearmanr(current_sub_data[c[0]], current_sub_data[c[1]], nan_policy='omit')[0] for c in combs]
    brain_sims = [1-scipy.stats.spearmanr(current_sub_data[c[0]], current_sub_data[c[1]], nan_policy='omit')[0] for c in combs]
    corr = scipy.stats.spearmanr(brain_sims, vec_sims, nan_policy='omit')[0]
    if numpy.isnan(corr):
        #logging.info('corrected nan...')
        corr = 0.
    #results.append(corr)
    return (corr, voxel_idx)

'''
def read_vectors(vectors_folder):

    vectors = dict()
    for f in os.listdir(vectors_folder):
        assert '.vector' in f
        with open(os.path.join(vectors_folder, f)) as i:
            if 'selected' in vectors_folder:
                vecs = numpy.array([l.strip() for l in i.readlines()], dtype=numpy.float64)
            else:
                vecs = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float64)
        if vecs.shape[0] == 0:
            print(f)
            continue
        else:
            if vecs.shape[0] != 768:
                vecs = numpy.nanmean(vecs, axis=0)
            assert vecs.shape[0] == 768
            vectors[f.replace('_', ' ').split('.')[0]] = vecs

    return vectors
'''

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
            cat = 'simple'
        elif 'Event' in cat:
            cat = 'verb'
        elif 'Coercion' in cat:
            cat = 'dot'
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

            beg = 4
            end = 11
            t_one = t + beg
            t_two = t + end
            if 'slow' in args.dataset:
                t_one += 1
                t_two += 1
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
parser.add_argument('--target', choices=['familiarity', 'concreteness', 
                    'imageability', 'frequency', 'word_vectors'], required=True, \
                    help = 'Which model to look for?')
parser.add_argument('--data_split', choices=['all', 'dot', 
                    'verb', 'simple'], required=True, \
                    help = 'Which data split to use?')
parser.add_argument('--vectors_folder', type=str, required=True, \
                    help = 'Specifies where the vectors are stored')
parser.add_argument('--senses', action='store_true', default=False, \
                    help = 'Averaging different senses together?')
parser.add_argument('--ceiling', action='store_true', default=False, \
                    help = 'Ceiling instead of word vectors?')
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
template = nilearn.datasets.load_mni152_template()

vectors = read_vectors(args)
output_folder = os.path.join('results', 'rsa_searchlight_{}_{}'.format(args.target, args.data_split), 
                             args.dataset, args.vectors_folder.split('_')[-1].split('_')[0])
if args.senses:
    output_folder = output_folder.replace('rsa_searchlight', 'rsa_searchlight_senses')
os.makedirs(output_folder, exist_ok=True)
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

    sub_results = collections.defaultdict(list)
    logging.info('Masking using Fisher feature selection')
    #raise RuntimeError('Part to be implemented')
    ### Left hemisphere
    #map_nifti = nilearn.image.load_img('region_maps/maps/left_hemisphere.nii')
    full_sub_data, beg, end = load_subject_runs(runs)
    ### Correcting vectors and brain data keys
    sub_data_keys = {tuple(k.replace("'", ' ').split()) : k  for k in full_sub_data.keys()}
    sub_data_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in sub_data_keys.items()}
    full_sub_data = {k : full_sub_data[v] for k, v in sub_data_keys.items()}
    ### Balancing number of trials in the case of senses
    if args.senses:
        mapper = {tuple(s.replace("'", ' ').replace(' ', '_').split('_')) : '_'.join([s.split(' ')[-1], trial_infos['category'][s_i]]) for s_i, s in enumerate(trial_infos['stimulus'])}
        senses = {v : [k for k, v_two in mapper.items() if v_two==v] for v in mapper.values()}
        senses = {k : [' '.join([v_two[i] for i in [0, -1]]) for v_two in v] for k, v in senses.items()}
        senses = {k : v for k, v in senses.items() if 'neg' not in k}

        random_indices = random.choices(range(5), k=2)
        for sense, stimuli in senses.items():
            if len(stimuli) > 1:
                for sense_stim in stimuli:
                    full_sub_data[sense_stim] = numpy.array(full_sub_data[sense_stim])[random_indices]
    #full_sub_data = load_subject_runs(runs)
    ### Averaging, keeping only one response per stimulus
    sub_data = {k : numpy.average(v, axis=0) for k, v in full_sub_data.items()}
    dimensionality = list(set([v.shape[0] for k, v in sub_data.items()]))[0]

    '''
    ### Reduce keys to actually present
    ### Correcting vectors and brain data keys
    sub_data_keys = {tuple(k.replace("'", ' ').split()) : k  for k in sub_data.keys()}
    sub_data_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in sub_data_keys.items()}
    sub_data = {k : sub_data[v] for k, v in sub_data_keys.items()}
    '''
    with open('{}_stimuli_ratings.tsv'.format(args.dataset)) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]

    if args.target == 'concreteness':
        model_data = {l[0] : float(l[4]) for l in lines[1:]}
    elif args.target == 'familiarity':
        model_data = {l[0] : float(l[2]) for l in lines[1:]}
    elif args.target == 'imageability':
        model_data = {l[0] : float(l[6]) for l in lines[1:]}
    elif args.target == 'frequency':
        model_data = dict()
        for l in lines[1:]:
            with open(os.path.join('resources', 'book_for_lunch_sentences',
                '{}.vector'.format(l[0].replace(' ', '_')))) as i:
                length = len(list(i.readlines()))
            model_data[l[0]] = numpy.log(length)
    elif args.target == 'word_vectors':
        vectors_keys = {tuple(k.replace("_", ' ').split()) : k  for k in vectors.keys()}
        vectors_keys = {'{} {}'.format(k[0], k[2]) if len(k)==3 else ' '.join(k) : v for k, v in vectors_keys.items()}
        model_data = {k : vectors[v] for k, v in vectors_keys.items()}
    ### Limiting
    if args.data_split != 'all':
        stimuli = [l[0] for l in lines[1:] if args.data_split in l[1]]
    else:
        stimuli = list(model_data.keys())
    model_data = {s : model_data[s] for s in stimuli}
    sub_data = {k : v for k, v in sub_data.items() if k in model_data.keys()}
    if args.senses:
        ### Collapsing senses for words if they are abstract/concrete
        
        actual_vectors = dict()
        actual_brain = dict()
        for sense, stimuli in senses.items():
            if len(stimuli) == 1:
                actual_brain[stimuli[0]] = sub_data[stimuli[0]]

                if args.ceiling:
                    actual_vectors[stimuli[0]] = current_ceiling[stimuli[0]]
                else:
                    actual_vectors[stimuli[0]] = model_data[stimuli[0]]
            else:
                data_avg = numpy.average([sub_data[s] for s in stimuli], axis=0)
                actual_brain[stimuli[0]] = data_avg

                if args.ceiling:
                    vec_avg = numpy.average([current_ceiling[s] for s in stimuli], axis=0)
                else:
                    vec_avg = numpy.average([model_data[s] for s in stimuli], axis=0)
                actual_vectors[stimuli[0]] = vec_avg
        sub_data = actual_brain.copy()
        model_data = actual_brain.copy()
    combs = list(itertools.combinations(list(sub_data.keys()), 2))
    if args.target in ['concreteness', 'familiarity', 'imageability', 'frequency']:
        model_sims = [abs(model_data[c[0]]-model_data[c[1]]) for c in combs]
    elif args.target == 'word_vectors':
        model_sims = [scipy.stats.spearmanr(vectors[c[0]], vectors[c[1]])[0] for c in combs]
    #vec_sims = [scipy.stats.spearmanr(vectors[c[0]], vectors[c[1]])[0] for c in combs]

    logging.info('Computing adjacency matrix...')
    ### Computing adjacency matrix
    sample_img = nilearn.image.index_img(single_run, 15)
    whole_brain = nilearn.masking.compute_brain_mask(sample_img)
    '''
    seeds_gen = whole_brain.get_fdata()
    seeds = list()
    for x in range(whole_brain.shape[0]):
        #if x > whole_brain.shape[0]/2:
        if 1 == 1:
            for y in range(whole_brain.shape[1]):
                for z in range(whole_brain.shape[2]):
                    if seeds_gen[x, y, z] == 1.0:
                        seeds.append((x, y, z))
        ### No half-brain analyses
        #else:
            for y in range(whole_brain.shape[1]):
                for z in range(whole_brain.shape[2]):
                    if seeds_gen[x, y, z] == 1.0:
                        seeds.append((x, y, z))
                        #half_brain[x,y,z] = 0.
    '''

    ### World coordinates of the seeds
    process_mask_coords = numpy.where(whole_brain.get_fdata()!=0)
    process_mask_coords = nilearn.image.resampling.coord_transform(
                        process_mask_coords[0], process_mask_coords[1],
                        process_mask_coords[2], whole_brain.affine)
    process_mask_coords = numpy.asarray(process_mask_coords).T
    #half_brain = nilearn.image.new_img_like(ref_niimg=nii_img, data=half_brain)
    _, adj_matrix = _apply_mask_and_get_affinity(
                                                 #seeds, sample_img, 
                                                 process_mask_coords, sample_img, 
                                                 radius=6., 
                                                 allow_overlap=True, 
                                                 mask_img=whole_brain)
                                                 #mask_img=half_brain)
    logging.info('Now collecting correlations...')
    results = list()
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as mp:
        results = mp.map(compute_corr, [[adj_matrix[voxel_idx].nonzero()[1], sub_data, model_sims, voxel_idx] for voxel_idx in range(adj_matrix.shape[0])])
        #for voxel_idx in tqdm(range(adj_matrix.shape[-1])):
        mp.terminate()
        mp.join()
    logging.info('Done collecting correlations!')
    results = [v for v, i in sorted(results, key=lambda item : item[1])]
    
    empty_brain = numpy.zeros(whole_brain.shape)
    empty_brain[whole_brain.get_fdata().astype(bool)] = results
    current_brain = nilearn.image.new_img_like(ref_niimg=whole_brain, 
                                           data=empty_brain)
    #current_brain = nilearn.image.resample_to_img(current_brain, template)
    current_brain.to_filename(os.path.join(output_folder, 'sub-{:02}.nii'.format(s)))
