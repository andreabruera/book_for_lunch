import argparse
import mne
import nilearn
import numpy
import os
import scipy

from nilearn import datasets, image, plotting
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from scipy import stats
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--target', choices=['concreteness', 'familiarity', 
                    'frequency', 'word_vectors', 'imageability'],
                    required=True, help='What model to consider?')
parser.add_argument('--data_split', choices=['all', 'dot', 
                    'verb', 'simple'], required=True, \
                    help = 'Which data split to use?')
parser.add_argument('--dataset', choices=['book_fast', 'lunch_fast'],
                    required=True, help='Which dataset?')
args = parser.parse_args()

template = nilearn.datasets.load_mni152_template()

folder = os.path.join('results', 
                      'rsa_searchlight_{}_{}'.format(args.target, args.data_split),
                      args.dataset)
#folder = os.path.join('results', 'searchlight')
search_results = dict()

print('Loading images')
with tqdm() as pbar:
    for root, direc, filez in os.walk(folder):
        for f in filez:
            #print(f)
            split_root = root.split('/')
            split = split_root[-1]
            analysis = split_root[-2]
            dataset = split_root[-3]
            key = (split, analysis, dataset)

            nii_img = nilearn.image.load_img(os.path.join(root, f))
            mask = nilearn.masking.compute_brain_mask(nii_img)
            sub_data = nilearn.masking.apply_mask(nii_img, mask)
            if key not in search_results.keys():
                search_results[key] = [sub_data]
            else:
                search_results[key].append(sub_data)
            pbar.update(1)

print('Loading adjacency matrix')
### Computing adjacency matrix
whole_brain = nilearn.masking.compute_brain_mask(nii_img)
'''
half_brain = whole_brain.get_fdata()
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

#half_brain = nilearn.image.new_img_like(ref_niimg=nii_img, data=half_brain)
_, adj_matrix = _apply_mask_and_get_affinity(seeds, nii_img, radius=12., 
                                             allow_overlap=True, 
                                             mask_img=whole_brain)
                                             #mask_img=half_brain)
'''
### World coordinates of the seeds
bool_mask = whole_brain.get_fdata().astype(bool)
process_mask_coords = numpy.where(whole_brain.get_fdata()!=0)
process_mask_coords = nilearn.image.resampling.coord_transform(
                    process_mask_coords[0], process_mask_coords[1],
                    process_mask_coords[2], whole_brain.affine)
process_mask_coords = numpy.asarray(process_mask_coords).T
#half_brain = nilearn.image.new_img_like(ref_niimg=nii_img, data=half_brain)
_, adj_matrix = _apply_mask_and_get_affinity(
                                             #seeds, sample_img, 
                                             process_mask_coords, nii_img, 
                                             radius=6., 
                                             allow_overlap=True, 
                                             mask_img=whole_brain
                                             #mask_img=half_brain
                                             )
print('Loaded!')
#empty_brain = numpy.zeros(seeds_gen.shape)
empty_brain = numpy.zeros(whole_brain.shape)
output_folder = os.path.join('results', 'group_rsa_searchlight_{}'.format(args.target))
os.makedirs(output_folder, exist_ok=True)

for k, subs in search_results.items():
    print(k)
    '''
    subs = list()
    for sub_data in v:
        sub_data = sub_data.get_fdata()
        sub_coll = list()
        for s in seeds:
            sub_coll.append(sub_data[s])
        sub_coll = numpy.array(sub_coll)
        subs.append(sub_coll)
    '''

    '''
    if 'time_resolved' in k[-2]:
        continue
        times = list()
        for t in range(sub_coll.shape[-1]):
            current_subs = [s[:,t] for s in subs]
            current_subs = numpy.array(current_subs)
            ### Difference between actual scores and baseline
            #ps = scipy.stats.ttest_1samp(current_subs, 0.5, alternative='greater')[1]
            #ps = mne.stats.fdr_correction(ps)[1]
            #import pdb; pdb.set_trace()
            current_subs = current_subs-0.5
            ts,one,ps,two = mne.stats.permutation_cluster_1samp_test(current_subs, tail=1,
                                                               threshold=dict(start=0, step=0.2),
                                                               adjacency=adj_matrix, n_jobs=48)
            current_brain = empty_brain.copy()
            sig_places = list()
            print([k, numpy.amin(ps)])
            for s, p in zip(seeds, ps):
                current_brain[s] = 1-p
                if p <= 0.05:
                    sig_places.append((s, p))
            times.append(current_brain)
        current_brain = numpy.stack(times, axis=-1)
    '''

    #continue
    subs = numpy.array(subs)
    ### Difference between actual scores and baseline
    '''
    ps = scipy.stats.ttest_1samp(subs, 0.5, alternative='greater')[1]
    ps = mne.stats.fdr_correction(ps)[1]
    import pdb; pdb.set_trace()
    '''
    #subs = subs-0.5
    ts,one,ps,two = mne.stats.permutation_cluster_1samp_test(subs, tail=1,
                                                       threshold=dict(start=0, step=0.2),
                                                       adjacency=adj_matrix, n_jobs=48)
    current_brain = empty_brain.copy()
    current_brain[bool_mask] = 1-ps
    sig_places = list()
    print([k, numpy.amin(ps)])
    '''
    for s, p in zip(seeds, ps):
        current_brain[s] = 1-p
        if p <= 0.05:
            sig_places.append((s, p))
    '''

    current_brain = nilearn.image.new_img_like(ref_niimg=whole_brain, 
                                           data=current_brain)
    #current_brain = nilearn.image.resample_to_img(current_brain, template)
    current_brain.to_filename(os.path.join(output_folder, '{}.nii'.format('_'.join(k))))
