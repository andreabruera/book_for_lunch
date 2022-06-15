import argparse
import nilearn
import numpy
import os

from nilearn import datasets, image, plotting, surface
### Surface
fsaverage = datasets.fetch_surf_fsaverage()

dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 
                            'dataset', 'neuroscience', 
                            'dot_book_fast_bids', 'derivatives',
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
maps_data = maps.get_fdata()
labels = dataset['labels']
collector = {l : numpy.array([], dtype=numpy.float64) for l in labels}
#maps = nilearn.image.resample_to_img(map_nifti, maps, interpolation='nearest')
interpr_nifti = nilearn.image.resample_to_img(plot_img, maps, interpolation='nearest')
for l_i, label in enumerate(labels):
    if l_i > 0:
        msk = maps_data == l_i
        '''
        for index in indices:
            region_map = numpy.logical_or(region_map, \
                                      maps_data==index)
        region_map = numpy.where(region_map==True, 1, 0)
        '''
        mskd = nilearn.masking.apply_mask(interpr_nifti, nilearn.image.new_img_like(maps, msk))
        collector[label] = mskd[mskd>0.]
collector = sorted({k : v.shape[0]/16 for k, v in collector.items()}.items(), key=lambda items : items[1],
                    reverse=True)
total = sum(list([k[1] for k in collector]))
percentages = {k[0] : k[1]/total if k[1] != 0. else 0. for k in collector}
output_perc = os.path.join(out, '{}_{}.txt'.format(args.spatial_analysis, n_dims))
with open(output_perc, 'w') as o:
    for k, v in percentages.items():
        o.write('{}\t{}%\n'.format(k, round(v*100, 2)))

### Right
texture = surface.vol_to_surf(plot_img, fsaverage.pial_right)
r = plotting.plot_surf_stat_map(
            fsaverage.pial_right, texture, hemi='right',
            title='Surface right hemisphere', colorbar=True,
            threshold=.95, 
            bg_map=fsaverage.sulc_right,
            darkness=0.6,
            cmap='BuPu',
            #view='medial',
            alpha=0.4,
            dpi=600,
            #cmap='Spectral_R'
            )
r.savefig(os.path.join(out, \
            'surface_right_{}_{}.jpg'.format(args.spatial_analysis, n_dims)),
            dpi=600)
### Left
texture = surface.vol_to_surf(plot_img, fsaverage.pial_left)
l = plotting.plot_surf_stat_map(
            fsaverage.pial_left, texture, hemi='left',
            title='Surface left hemisphere', colorbar=True,
            threshold=.95, 
            bg_map=fsaverage.sulc_left,
            #cmap='Spectral_R', 
            #cmap='Wistia',
            cmap='BuPu',
            #view='medial',
            darkness=0.6,
            alpha=0.4,
            dpi=600,
            )
l.savefig(os.path.join(out, \
            'surface_left_{}_{}.jpg'.format(args.spatial_analysis, n_dims)),
            dpi=600)
