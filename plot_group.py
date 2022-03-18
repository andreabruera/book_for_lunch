import argparse
import mne
import nilearn
import os

from nilearn import datasets, image, plotting, surface

parser = argparse.ArgumentParser()
parser.add_argument('--target', choices=['concreteness', 'familiarity', 
                    'frequency', 'word_vectors', 'imageability'],
                    required=True, help='What model to consider?')
parser.add_argument('--data_split', choices=['all', 'dot', 
                    'verb', 'simple'], required=True, \
                    help = 'Which data split to use?')
args = parser.parse_args()
### Surface
fsaverage = datasets.fetch_surf_fsaverage()

folder = os.path.join('results', 'group_rsa_searchlight_{}'.format(args.target))
out_folder = os.path.join('plots', 'group_rsa_searchlight_{}_{}'.format(args.target, args.data_split))
os.makedirs(out_folder, exist_ok=True)
for f in os.listdir(folder):
    if 'nii' in f:
        img = nilearn.image.load_img(os.path.join(folder, f))
        ### Whole trial
        if len(img.shape) == 3:
            ### Mosaic
            nilearn.plotting.plot_stat_map(img, 
                      threshold=0.95, 
                      cmap='prism',
                      vmax=1.,
                      display_mode='mosaic',
                      output_file=os.path.join(out_folder, \
                                    f.replace('nii','jpg')))
            ### Right
            texture = surface.vol_to_surf(os.path.join(folder, f), fsaverage.pial_right)
            r = plotting.plot_surf_stat_map(
                        fsaverage.pial_right, texture, hemi='right',
                        title='Surface right hemisphere', colorbar=True,
                        threshold=.95, 
                        bg_map=fsaverage.sulc_right,
                        darkness=0.6,
                        cmap='prism_r',
                        #view='medial',
                        alpha=0.4,
                        dpi=600,
                        #cmap='Spectral_R'
                        )
            r.savefig(os.path.join(out_folder, \
                        'surface_right_{}'.format(f.replace('nii','jpg'))),
                        dpi=600)
            ### Left
            texture = surface.vol_to_surf(os.path.join(folder, f), fsaverage.pial_left)
            l = plotting.plot_surf_stat_map(
                        fsaverage.pial_left, texture, hemi='left',
                        title='Surface left hemisphere', colorbar=True,
                        threshold=.95, 
                        bg_map=fsaverage.sulc_left,
                        #cmap='Spectral_R', 
                        cmap='prism_r',
                        #view='medial',
                        darkness=0.6,
                        alpha=0.4,
                        dpi=600,
                        )
            l.savefig(os.path.join(out_folder, \
                        'surface_left_{}'.format(f.replace('nii','jpg'))),
                        dpi=600)
                        

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
