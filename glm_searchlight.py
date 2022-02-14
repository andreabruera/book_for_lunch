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

    betas = {k : fmri_glm.compute_contrast(k).get_fdata() for k in cat_mapper.keys()}
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

    for data_split, current_data in sub_data.items():
        logging.info('Running Searchlight for {}'.format(data_split))

        labels = list()
        samples = list()

        for k, v in current_data.items():
            for item in v:
                labels.append(k)
                samples.append(item)
        ### Randomizing
        samples = random.sample(samples, k=len(samples))

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
                                     args.dataset, 'glm', 
                                     data_split)
        os.makedirs(output_folder, exist_ok=True)
        out_img.to_filename(os.path.join(output_folder, 
                               'sub-{:02}_searchlight_scores.nii'.format(s)))
