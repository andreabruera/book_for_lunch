import mne
import nilearn
import os

from nilearn import image, plotting

folder = os.path.join('results', 'group_searchlight')
out_folder = os.path.join('plots', 'group_searchlight')
os.makedirs(out_folder, exist_ok=True)
for f in os.listdir(folder):
    if 'nii' in f:
        img = nilearn.image.load_img(os.path.join(folder, f))
        ### Whole trial
        if len(img.shape) == 3:
            nilearn.plotting.plot_stat_map(img, 
                      threshold=0.4, vmax=1.0,
                      display_mode='mosaic',
                      output_file=os.path.join(out_folder, \
                                    f.replace('nii','jpg')))

        ### Time resolved
        elif len(img.shape) == 4:
            for t in range(img.shape[-1]):
                current_img = nilearn.image.index_img(img, t)
                file_name = os.path.join(out_folder, \
                                '{}_{}'.format(t, f.replace('nii','jpg')))
                nilearn.plotting.plot_stat_map(current_img, \
                                       threshold=0.4, vmax=1.0,
                                       display_mode='mosaic',
                                       output_file=file_name)
