import argparse
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

from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting
from scipy import stats
from sklearn import feature_selection
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from tqdm import tqdm

def fisher(all_args):
    current_data = all_args[0]
    dimensionality = all_args[1]
    subject = all_args[2]
    sub_values = numpy.zeros(dimensionality)
    for sub in current_data:
        values = numpy.zeros(dimensionality)
        for idx in tqdm(range(dimensionality)):
            dim_recs = [[trial[idx] for trial in stim_data] for stim, stim_data in sub.items()]
            dim_recs = numpy.array(dim_recs)
            overall_mean = numpy.average(dim_recs)
            means = numpy.average(dim_recs, axis=1)
            variances = numpy.var(dim_recs, axis=1)
            fisher = (numpy.sum((means*overall_mean)**2))/numpy.sum(variances**2)
            if numpy.isnan(fisher):
                fisher = 0.0
            values[idx] = fisher
        sub_values += values
    return [subject, sub_values]

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

        if map_nifti == None:
            map_nifti = nilearn.masking.compute_brain_mask(run)
            #masked_run = run.get_fdata()
            #masked_run = masked_run.reshape(numpy.product(masked_run.shape[:3]), -1)
        masked_run = nilearn.masking.apply_mask(run, map_nifti).T

        for t_i, t in enumerate(infos['start']):
            stimulus = infos['stimulus'][t_i]

            if args.analysis == 'time_resolved':
                fmri_timeseries = masked_run[:, t:t+18]
            elif 'whole_trial' in args.analysis:
                ### Keeping responses from noun+4 to noun+9
                beg = 4
                end = 11
                t_one = t + beg
                t_two = t + end
                if 'slow' in args.dataset:
                    t_one += 1
                    t_two += 1
                if 'flattened' in args.analysis:
                    fmri_timeseries = masked_run[:, t_one:t_two].flatten()
                else:
                    fmri_timeseries = numpy.average(\
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

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

os.makedirs('region_maps', exist_ok=True)
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 
                            'dataset', 'neuroscience',
                            'dot_{}_bids'.format(args.dataset), 
                            'derivatives',
                        )

n_subjects = len(os.listdir(dataset_path))
overall_sub_data = list()

for s in range(1, n_subjects+1):
    #print(s)
    ### Loading the image
    sub_path = os.path.join(dataset_path, 'sub-{:02}'.format(s), 
                             'ses-mri', 'func',
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
    ### Left hemisphere
    #map_nifti = nilearn.image.load_img('region_maps/maps/left_hemisphere.nii')
    full_sub_data, beg, end = load_subject_runs(runs)
    dimensionality = list(set([v[0].shape[0] for k, v in full_sub_data.items()]))[0]
    full_sub_data = {k : v for k, v in full_sub_data.items() if k != 'neg neg'}
    overall_sub_data.append(full_sub_data)

subjects = list(range(len(overall_sub_data)))

all_args = [[[overall_sub_data[s] for s in subjects if s!=curr], dimensionality, curr+1] for curr in subjects]
with multiprocessing.Pool() as mp:
    results = mp.map(fisher, all_args)
    mp.terminate()
    mp.join()

output_folder = os.path.join('voxel_selection', 'fisher_scores', 
                             '{}_to_{}'.format(beg, end),
                             args.dataset, args.analysis, 
                             )
os.makedirs(output_folder, exist_ok=True)

for s in results:
    with open(os.path.join(output_folder, 'sub-{:02}.fisher'.format(s[0])), 'w') as o:
        for v in s[1]: 
            o.write('{}\t'.format(float(v)))
