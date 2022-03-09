import nibabel
import nilearn
import numpy
import os

from matplotlib import pyplot
from nilearn import datasets, image, input_data, plotting

os.makedirs('region_maps', exist_ok=True)

### Extracting masks for regions
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
maps = dataset['maps']
labels = dataset['labels']
relevant_regions = dict()
center = dict()
# Anterior Inferior Frontal Gyrus
aifg = ['Inferior Frontal Gyrus, pars triangularis', \
        'Frontal Operculum Cortex']
relevant_regions['anterior_inferior_frontal_gyrus'] = aifg
center['anterior_inferior_frontal_gyrus'] = (-50, 28, 8)
# Angular Gyrus
ag = ['Angular Gyrus', \
      'Lateral Occipital Cortex, superior division']
relevant_regions['angular_gyrus'] = ag
center['angular_gyrus'] = (-48, -52, -38)
# Precuneus and Posterior Cingulate Gyrus
pc = ['Precuneous Cortex', \
      'Cingulate Gyrus, posterior division']
relevant_regions['precuneus'] = pc
center['precuneus'] = (0, -64, 38)
# Posterior Inferior Temporal Lobe
pitl = ['Occipital Fusiform Gyrus', \
        'Temporal Occipital Fusiform Cortex', \
        'Inferior Temporal Gyrus, temporooccipital part']
relevant_regions['posterior_inferior_temporal_lobe'] = pitl
center['posterior_inferior_temporal_lobe'] = (-34, -52, -14)
# Posterior Lateral Temporal Cortices
pltc = ['Middle Temporal Gyrus, temporooccipital part', \
        'Superior Temporal Gyrus, posterior division', \
        'Planum Temporale']
relevant_regions['posterior_lateral_temporal_cortices'] = pltc
center['posterior_lateral_temporal_cortices'] = (-60, -22, -2)
# Ventral Anterior Temporal Lobe
vatl = ['Temporal Fusiform Cortex, anterior division', \
        'Parahippocampal Gyrus, anterior division']
relevant_regions['ventral_anterior_temporal_lobe'] = vatl
center['ventral_anterior_temporal_lobe'] = (-24, -2, -36)


region_indices = dict()
for k, v in relevant_regions.items():
    for n in v:
        assert n in labels
    region_indices[k] = [labels.index(n) for n in v]
### All regions
region_indices['language_areas'] = [labels.index(n) for k, v in relevant_regions.items() for n in v]
center['language_areas'] = (-50, 28, 8)

masker = nilearn.input_data.NiftiLabelsMasker(labels_img=maps, standardize=True, \
                                              memory='nilearn_cache', resampling_target='labels')
#template_path = 'resources/spm12/canonical/avg152T1.nii'
#template = nilearn.image.load_img(template_path)
template = nilearn.datasets.load_mni152_template()
dataset_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'neuroscience', \
                        'dot_book_fast_bids', 'derivatives', 
                        )
#'sub-01', 'ses-mri', \
n_subjects = len(os.listdir(dataset_path))

for s in range(1, n_subjects+1):
    if s > 1:
        break
    print(s)
    ### Loading the image
    sub_path = os.path.join(dataset_path, 'sub-{:02}'.format(s), 'ses-mri', \
                             'func', 
                             #'sub-01_ses-mri_task-dotbookfast_run-01_bold.nii'
                             )
    n_runs = len([k for k in os.listdir(sub_path) if 'nii' in k])
    for r in range(1, n_runs+1):
        if r > 1:
            break
        print(r)
        ### Reading events
        events_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dotbookfast_run-{:02}_events.tsv'.format(s, r))
        with open(events_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        events = {h : [l[h_i] for l in lines[1:]] for h_i, h in enumerate(lines[0])}
        trial_starts = [int(round(float(f), 0))-1 for f in events['onset'][1:][::7]]
        ### Checking that trial length is between 10 and 12 seconds
        for t_i, t in enumerate(trial_starts):
            if t!=trial_starts[-1]:
                assert trial_starts[t_i+1]-t > 9
                assert trial_starts[t_i+1]-t < 13
        trial_infos = {'start' : list(), 'stimulus' : list(), 'category' : list()}
        for t_i, t in enumerate(list(range(len(events['onset'])))[1:][::7]):
            cat = events['value'][t:t+7][1]
            stimulus = events['trial_type'][t:t+7]
            stimulus = '{} {}'.format(stimulus[1], stimulus[3])

            trial_infos['start'].append(trial_starts[t_i])
            trial_infos['category'].append(cat)
            trial_infos['stimulus'].append(stimulus)

        file_path = os.path.join(sub_path, 'sub-{:02}_ses-mri_task-dotbookfast_run-{:02}_bold.nii'.format(s, r))

        single_run = nilearn.image.load_img(file_path)
        single_run = nilearn.image.mean_img(single_run)
        #adapted_maps = nilearn.image.resample_to_img(maps, nilearn.image.index_img(single_run, 6))
        ### Just plotting to check whether everything went fine
        #new_img = nilearn.image.resample_to_img(nilearn.image.index_img(single_run, 6), template)
        #old_img = nilearn.image.index_img(single_run, 6)

        plotting.plot_stat_map(
                #adapted_maps,
                #new_img,
                single_run,
                bg_img=template,
                cut_coords=(0, 0, 0),
                threshold=3,
                title="Resampled t-map")

        pyplot.savefig(os.path.join('region_maps', 'map_check.jpg'), dpi=300)
        pyplot.clf()
        pyplot.close()

        #maps_data = adapted_maps.get_fdata()
        maps = nilearn.image.resample_to_img(maps, single_run, interpolation='nearest')
        maps_data = maps.get_fdata()
        region_maps = dict()
        for region, indices in region_indices.items():
            ### Creating an array full of False, just to fill it later
            region_map = maps_data == 99
            for index in indices:
                region_map = numpy.logical_or(region_map, \
                                          maps_data==index)
            region_map = numpy.where(region_map==True, 1, 0)
            ### Considering only the left hemisphere
            '''
            for x in range(region_map.shape[0]):
                for y in range(region_map.shape[1]):
                    for z in range(region_map.shape[2]):
                        if x < region_map.shape[0]/2:
                            region_map[x, y, z] = 0
            '''
            new_mask = nilearn.image.new_img_like(
                          #ref_niimg=adapted_maps,\
                          ref_niimg=maps,
                          data=region_map)
            cut_coords = center[region]
            plotting.plot_stat_map(new_mask,
                    bg_img=template,
                    cut_coords=cut_coords,
                    threshold=0.5,
                    title="{} map".format(region))

            out_folder = os.path.join('region_maps', 'plots')
            os.makedirs(out_folder, exist_ok=True)
            pyplot.savefig(os.path.join(out_folder, \
                           '{}.jpg'.format(region)), dpi=300)
            pyplot.clf()
            pyplot.close()
            out_folder = os.path.join('region_maps', 'maps')
            os.makedirs(out_folder, exist_ok=True)
            nibabel.save(new_mask, os.path.join(out_folder, '{}.nii'.format(region)))
            region_maps[region] = region_map

        '''
        #temp_resamp = nilearn.image.resample_to_img(template, nilearn.image.index_img(single_run, 6)).get_fdata()
        resamp = nilearn.masking.compute_brain_mask(nilearn.image.index_img(single_run, 6)).get_fdata()
        ### Right hemisphere mask
        for x in range(region_map.shape[0]):
            for y in range(region_map.shape[1]):
                for z in range(region_map.shape[2]):
                    if x > region_map.shape[0]/2:
                        region_map[x, y, z] = 0
                    else:
                        if resamp[x, y, z] > 0.:
                            region_map[x, y, z] = 1 
        new_mask = nilearn.image.new_img_like(ref_niimg=adapted_maps,\
                      data=region_map)
        cut_coords = center[region]
        plotting.plot_stat_map(new_mask,
                bg_img=template,
                cut_coords=cut_coords,
                threshold=0.5,
                title="{} map".format(region))

        out_folder = os.path.join('region_maps', 'plots')
        os.makedirs(out_folder, exist_ok=True)
        pyplot.savefig(os.path.join(out_folder, \
                       'right_hemisphere.jpg'), dpi=300)
        pyplot.clf()
        pyplot.close()
        out_folder = os.path.join('region_maps', 'maps')
        os.makedirs(out_folder, exist_ok=True)
        nibabel.save(new_mask, os.path.join(out_folder, 'right_hemisphere.nii'))
        region_maps[region] = region_map

        ### Left hemisphere mask
        for x in range(region_map.shape[0]):
            for y in range(region_map.shape[1]):
                for z in range(region_map.shape[2]):
                    if x < region_map.shape[0]/2:
                        region_map[x, y, z] = 0
                    else:
                        if resamp[x, y, z] > 0.:
                            region_map[x, y, z] = 1 
        new_mask = nilearn.image.new_img_like(ref_niimg=adapted_maps,\
                      data=region_map)
        cut_coords = center[region]
        plotting.plot_stat_map(new_mask,
                bg_img=template,
                cut_coords=cut_coords,
                threshold=0.5,
                title="{} map".format(region))

        out_folder = os.path.join('region_maps', 'plots')
        os.makedirs(out_folder, exist_ok=True)
        pyplot.savefig(os.path.join(out_folder, \
                       'left_hemisphere.jpg'), dpi=300)
        pyplot.clf()
        pyplot.close()
        out_folder = os.path.join('region_maps', 'maps')
        os.makedirs(out_folder, exist_ok=True)
        nibabel.save(new_mask, os.path.join(out_folder, 'left_hemisphere.nii'))
        region_maps[region] = region_map
        '''
