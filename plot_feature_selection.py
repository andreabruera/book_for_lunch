import argparse
import nilearn
import numpy
import os

from nilearn import datasets, image, plotting

beg = 4
end = 11

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
args = parser.parse_args()

folder = os.path.join('voxel_selection',
                       'fisher_scores', 
                       '{}_to_{}'.format(beg, end),
                       args.dataset, args.analysis,
                       args.spatial_analysis,
                       )

dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'neuroscience', \
                        'dot_{}_bids'.format(args.dataset), 'derivatives',
                        )
sub_path = os.path.join(dataset_path, 'sub-01', 'ses-mri', \
                         'func',
                         )
file_path = os.path.join(sub_path, 'sub-01_ses-mri_task-dot{}_run-01_bold.nii'.format(args.dataset.replace('_', '')))
single_run = nilearn.image.load_img(file_path)
if args.spatial_analysis == 'all':
    map_nifti = nilearn.masking.compute_brain_mask(single_run)
elif args.spatial_analysis == 'language_areas':
    maps_folder = os.path.join('region_maps', 'maps')   
    assert os.path.exists(maps_folder)
    map_path = os.path.join(maps_folder, 'language_areas.nii')
    assert os.path.exists(map_path)
    map_nifti = nilearn.image.load_img(map_path)
masked_run = nilearn.masking.apply_mask(single_run, map_nifti).T[:, beg:end]
#sample_img = nilearn.image.index_img(single_run, 15)
final_img = numpy.zeros(masked_run.flatten().shape)

for root, direc, filez in os.walk(folder):
    for f in filez:
        with open(os.path.join(root, f)) as i:

            lines = numpy.array([l.strip().split('\t') for l in i.readlines()][0], dtype=numpy.float64)
        sorted_dims = sorted(list(enumerate(lines)), key=lambda item : item[1], reverse=True)
        assert final_img.shape[0] == len(sorted_dims)
        #blank = numpy.zeros(len(sorted_dims))
        n_dims = 5000
        selected_dims = [k[0] for k in sorted_dims[:n_dims]]
        for dim in selected_dims:
            final_img[dim] += 1.


out = os.path.join('plots', 
       'feature_selection_location')
os.makedirs(out, exist_ok=True)
final_img = numpy.average(final_img.reshape(masked_run.shape), axis=1)
blank = numpy.zeros(map_nifti.shape)
blank[map_nifti.get_fdata().astype(bool)] = final_img
plot_img = nilearn.image.new_img_like(map_nifti, blank)
maps_folder = os.path.join('region_maps', 'maps')   
assert os.path.exists(maps_folder)
map_path = os.path.join(maps_folder, 'language_areas.nii')
assert os.path.exists(map_path)
map_nifti = nilearn.image.load_img(map_path)
display = nilearn.plotting.plot_stat_map(
                               #plot_img, 
                               map_nifti,
                               #bg_img=map_nifti,
                               threshold = 0., 
                               display_mode='mosaic', 
                               #cut_coords=(-50, 28, 8), 
                               )
display.add_overlay(plot_img, threshold=0., cmap=plotting.cm.purple_green)
output_file = os.path.join(out, '{}_{}.jpg'.format(args.spatial_analysis, n_dims))
display.savefig(output_file)

dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
maps = dataset['maps']
labels = dataset['labels']
collector = {l : 0. for l in labels}
maps = nilearn.image.resample_to_img(maps, map_nifti, interpolation='linear')
for idx, val in zip(maps.get_fdata().flatten(), plot_img.get_fdata().flatten()):
    collector[labels[int(idx)]] += 1


