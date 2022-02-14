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
from nilearn import datasets, decoding, image, input_data, plotting
from sklearn import feature_selection
from sklearn.svm import SVC
from tqdm import tqdm

def load_subject_runs(runs, start, end, map_nifti=None):

    sub_data = {'all' : dict(), 'simple' : dict(), 'verb' : dict(), 'dot' : dict()}

    for run, infos in tqdm(runs):

        masked_run = run.get_fdata()

        for t_i, t in enumerate(infos['start']):
            full_cat = infos['category'][t_i]
            if 'con' in full_cat or 'obj' in full_cat:
                cat = 'concrete'
            elif 'abs' in full_cat or 'info' in full_cat:
                cat = 'abstract'
            elif 'catch' in full_cat:
                cat = ''
            ### Not considering cases with no stimulus
            if 'abstract' in cat or 'concrete' in cat:
                stimulus = infos['stimulus'][t_i]
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
                    fmri_timeseries = masked_run[:, :, :, t:t+18]
                elif args.analysis in ['whole_trial', 'flattened_trial']:
                    if 'slow' in args.dataset:
                        stim = 2
                        stim = 2
                    else:
                        stim = 1
                        stim = 1
                    ## Keeping responses from noun+4 to noun+4+4
                    beg = t + stim + start
                    end = t + stim + end
                    if args.analysis == 'whole_trial':
                        fmri_timeseries = numpy.average(\
                                              masked_run[:, :, :, beg:end], \
                                                        axis=3)
                sub_data['all'][cat][stimulus].append(fmri_timeseries)
                sub_data[full_cat][cat][stimulus].append(fmri_timeseries)

        del masked_run
    return sub_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['book_fast','lunch_fast', \
                                                         'book_slow', 'lunch_slow'],
                    help='Specify which dataset to use')
parser.add_argument('--analysis', required=True, \
                    choices=['time_resolved', 'whole_trial'],
                    help='Average time points, or run classification'
                         'time point by time point?')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

os.makedirs('region_maps', exist_ok=True)
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'neuroscience', \
                        'dot_{}_bids'.format(args.dataset), 'derivatives',
                        )
n_subjects = len(os.listdir(dataset_path))
map_results = dict()

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

        runs.append((single_run, trial_infos))

    start = 3
    end = 9
    full_sub_data = load_subject_runs(runs, start, end)

    for data_split, sub_data in full_sub_data.items():
        sub_data = {k : [numpy.average(v_two, axis=0) for k_two, v_two in v.items()] for k, v in sub_data.items()}
        assert len(set([len(v) for k, v in sub_data.items()])) == 1
        n_items = list(set([len(v) for k, v in sub_data.items()]))[0]

        labels = list()
        samples = list()

        for i in range(n_items):
            for k, v in sub_data.items():
                labels.append(k)
                samples.append(v[i])
        ### Randomizing
        samples = random.sample(samples, k=len(samples))
        if args.analysis == 'time_resolved':
            samples = numpy.array(samples)
            t_scores = list()
            for time_point in range(samples.shape[-1]):
                current_samples = [s[:,:,:,time_point] for s in samples]
                ### Stacking
                current_samples = numpy.stack(tuple(current_samples), axis=-1)
                assert current_samples.shape[:3] == single_run.shape[:3]

                ### Create 4D image from samples
                img_4d = nilearn.image.new_img_like(ref_niimg=single_run,\
                              data=current_samples)
                whole_brain = nilearn.masking.compute_brain_mask(img_4d)
                cv = sklearn.model_selection.KFold(n_splits=10)
                searchlight = nilearn.decoding.SearchLight(whole_brain,
                                process_mask_img=whole_brain, radius=12., 
                                n_jobs=48, verbose=0, cv=cv)
                searchlight.fit(img_4d, labels)
                t_scores.append(searchlight.scores_)
            t_scores = numpy.stack(tuple(t_scores), axis=-1)
            assert t_scores.shape[:3] == single_run.shape[:3]
            out_img = nilearn.image.new_img_like(ref_niimg=single_run,\
                          data=t_scores)

        if args.analysis == 'whole_trial':
            ### Stacking
            samples = numpy.stack(tuple(samples), axis=-1)
            assert samples.shape[:3] == single_run.shape[:3]

            ### Create 4D image from samples
            img_4d = nilearn.image.new_img_like(ref_niimg=single_run,\
                          data=samples)
            whole_brain = nilearn.masking.compute_brain_mask(img_4d)
            cv = sklearn.model_selection.KFold(n_splits=10)
            searchlight = nilearn.decoding.SearchLight(whole_brain,
                            process_mask_img=whole_brain, radius=12., 
                            n_jobs=48, verbose=0, cv=cv)
            searchlight.fit(img_4d, labels)
            out_img = nilearn.image.new_img_like(ref_niimg=single_run,\
                          data=searchlight.scores_)

        output_folder = os.path.join('results', 'searchlight', 
                                     args.dataset, args.analysis, 
                                     data_split)
        if args.analysis == 'whole_trial':
            current_folder = '{}_{}_whole_trial'.format(start, end)
            output_folder = output_folder.replace(args.analysis, current_folder)
        os.makedirs(output_folder, exist_ok=True)
        out_img.to_filename(os.path.join(output_folder, 
                               'sub-{:02}_searchlight_scores.nii'.format(s)))
