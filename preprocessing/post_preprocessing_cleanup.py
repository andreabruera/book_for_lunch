import os
import re

dataset_folder = os.path.join('/', 'import', 'cogsci', 'andrea', \
                      #'dataset', 'neuroscience', 'dot_book_slow_bids',
                      'dataset', 'neuroscience', 'dot_lunch_slow_bids',
                      )

### Checking that cleanup has to take place
files = [f for root, direc, filez in os.walk(dataset_folder) for f in filez if 'iy' in f or 'mean' in f]
assert len(files) > 0

### Removing useless files
for root, direc, filez in os.walk(dataset_folder):
    for f in filez:
        if 'derivatives' in root:

            if 'func' in root:
                if 'final_' not in f:
                    os.system('rm {}'.format(os.path.join(root, f)))
            elif 'anat' in root:
                if 'y_rsub' not in f or 'iy_rsub' in f:
                    os.system('rm {}'.format(os.path.join(root, f)))

### Renaming important files
for root, direc, filez in os.walk(dataset_folder):
    for f in filez:
        if 'derivatives' in root:

            if 'func' in root:
                if 'final_' in f:
                    os.system('mv {} {}'.format(os.path.join(root, f), os.path.join(root, f.replace('final_au', ''))))
            elif 'anat' in root:
                if 'y_rsub' in f:
                    os.system('mv {} {}'.format(os.path.join(root, f), os.path.join(root, f.replace('y_rsub', 'sub'))))

### Copying event files
for root, direc, filez in os.walk(dataset_folder):
    for f in filez:
        if 'derivatives' not in root:
            if 'tsv' in f:
                os.system('cp {} {}'.format(os.path.join(root, f), os.path.join(root.replace(dataset_folder, '{}/derivatives'.format(dataset_folder)), f))) 
