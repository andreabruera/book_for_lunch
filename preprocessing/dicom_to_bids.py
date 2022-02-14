import argparse
import json
import os
import pandas
import re
import scipy

from scipy import io

#base_folder = os.path.join('.')
#folders = [o for o in os.listdir(base_folder) if 'MRI' in o]

#base_folders = [os.path.join(base_folder, f) for f in folders]
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True, type=str,
                    help='Indicates where the files are')
args = parser.parse_args()

assert 'MRI' in args.dataset_folder

number_files = 430 if 'lunch' in args.dataset_folder else 500
number_mprage = 177 if 'book' in args.dataset_folder else 179

subjects = sorted([f for f in os.listdir(args.dataset_folder) if '_' not in f])
task = args.dataset_folder.lower().replace('mri_', '').replace('_', '')
for s in range(len(subjects)):
#for s in range(1):
    sub = s + 1
    subject_folder = os.path.join(args.dataset_folder, subjects[s])
    assert os.path.exists(subject_folder)
    runs_folders = list()
    mprage_folders = list()
    mat_files = list()
    subject_id = subject_folder.split('/')[-1]
    for root, direc, filez in os.walk(subject_folder):
        for f in filez:
            if 'epi' in root and 'DiCo' in root:
                alternative_root = root.split('/')
                number = int(alternative_root[-1].split('_')[0])
                alternative_root[-1] = alternative_root[-1]\
                            .replace('{}_'.format(number), '{}_'.format(number-1))\
                            .replace('_DiCo', '')
                alternative_root = '/'.join(alternative_root)
                if root not in runs_folders and alternative_root not in runs_folders:
                    try:
                        assert len(os.listdir(root)) == number_files+1
                        runs_folders.append(root)
                    except AssertionError:
                        try:
                            assert len(os.listdir(alternative_root)) == number_files+1
                            runs_folders.append(alternative_root)
                            #print([alternative_root, len(os.listdir(alternative_root))])
                        except AssertionError:
                            #print([root, len(os.listdir(root))])
                            pass
                    except AssertionError:
                        #print([root, len(os.listdir(root))])
                        pass
            ### Anatomical image
            elif 'mprage' in root:
                if root not in mprage_folders:
                    assert len(os.listdir(root)) > 170
                    mprage_folders.append(root)

            elif 'Exp' in f and '.mat' in f:
                mat_file = os.path.join(root, f)
                mat_files.append(mat_file)

    mat_files = sorted(mat_files)

    ### Collecting stimuli and condition files
    condition_files = list()
    stimuli_files = list()

    mat_blueprint = mat_files[0]
    for trial in range(1, 7):
        trial_blueprint = mat_blueprint.replace('s1.', 's{}.'.format(trial))
        stimuli_file = trial_blueprint.replace('mat', 'trd.targetConcept')
        if os.path.exists(stimuli_file):
            stimuli_files.append(stimuli_file)
        condition_file = trial_blueprint.replace('mat', 'trd.cond')
        if os.path.exists(condition_file):
            condition_files.append(condition_file)

    ### Sort runs
    runs_folders = sorted(runs_folders, key = lambda item : int(item.split('/')[-1].split('_')[0]))
    ### Sort stimuli
    stimuli_files = sorted(stimuli_files, key = lambda item : int(item.split('/')[-1].split('.')[0][-1]))
    ### Sort conditions
    condition_files = sorted(condition_files, key = lambda item : int(item.split('/')[-1].split('.')[0][-1]))
    ### Sort times
    mat_files = sorted(mat_files, key = lambda item : int(item.split('/')[-1].split('.')[0][-1]))
    ### Correcting missing time files
    if len(condition_files) == 6 and len(mat_files) == 5:
        ### Find missing file
        m_f = [int(n.split('/')[-1].split('.')[0][-1]) for n in mat_files]
        missing_number = [i for i in range(1, 7) if i not in m_f][0]
        mat_files.append(mat_files[missing_number-2])
        mat_files = sorted(mat_files, key = lambda item : int(item.split('/')[-1].split('.')[0][-1]))
    ### Checking
    print(subject_id)
    print('Runs: {}, Stimuli: {}, Conditions: {}, Times: {}'.format(\
             len(runs_folders), len(stimuli_files), len(condition_files), len(mat_files)))
    for f in runs_folders:
        print(f)
    for f in stimuli_files:
        print(f)
    for f in condition_files:
        print(f)
    for f in mat_files:
        print(f)


    out_folder = os.path.join('polysemy_bids', '{}_bids'.format(args.dataset_folder.replace('MRI_', '')), \
            'sub-{:02}'.format(sub), 'ses-mri')

    try:
        assert len(list(set([len(mat_files), len(stimuli_files), len(runs_folders)]))) == 1
    except AssertionError:
        print(subject_folder)
        print([len(mat_files), len(stimuli_files), len(runs_folders)])
        #raise RuntimeError('Check again, there is a problem')
        print('Check again, there is a problem with {}'.format(subject_folder))
        continue

    ### Create functional runs and anatomical structures

    print(sub, subject_folder)
    for r_i, r in enumerate(runs_folders):
        r_folder = os.path.join(out_folder, 'func')
        os.makedirs(r_folder, exist_ok=True)
        os.system('dcm2niix -9 -z y -b y -o {} -f sub-{:02}_ses-mri_task-{}_run-{:02}_bold {}'.format(r_folder, sub, task, r_i+1, r))

    if len(mprage_folders) > 1:
        mprage_folders = [sorted(mprage_folders, key=lambda item : int(item.split('/')[-1].split('_')[0]))[-1]]
    for r_i, r in enumerate(sorted(mprage_folders)):
        anat_folder = os.path.join(out_folder, 'anat')
        os.makedirs(anat_folder, exist_ok=True)
        os.system('dcm2niix -z y -9 -b y -o {} -f sub-{:02}_ses-mri_acq-mprage_T1w {}'.format(anat_folder, sub, r))

    ### Writing file the events
    for trial_i in range(len(mat_files)):
        mat_file = mat_files[trial_i]
        stimuli_file = stimuli_files[trial_i]
        condition_file = condition_files[trial_i]

        tsv_dict = {'onset' : list(), 'duration' : list(), 'trial_type' : list(), 'value' : list()}

        trial_infos = scipy.io.loadmat(mat_file)['ExpInfo'][0][0][3][0]
        times = [float(t[2]) for t in trial_infos]
        with open(stimuli_file) as i:
            stimuli = [l.strip()[:-2] for l in i.readlines()]
        with open(condition_file) as i:
            conditions = [l.strip()[:-2] for l in i.readlines()]
        ### Adding fixation cross at the beginning
        ### Setting trial structure
        ### Event, duration
        if 'fast' in task:
            trial = [('fixation', 0.5), ('w1', 0.45), ('empty', 0.1), ('w2', 0.45), \
                     ('fixation', 1.5), ('question_mark', 1.), ('empty', 6.)]
            times[0] += 15.
        else:
            trial = [('w1', 2.), ('w2', 2.), \
                     ('question_mark', 1.), ('empty', 5.)]
            del times[0]

        assert len(stimuli) == len(times)
        assert len(stimuli) == len(conditions)
        for e_i, starting_point in enumerate(times):
            word = stimuli[e_i]
            condition = conditions[e_i]
            for t_structure in trial:
                tsv_dict['onset'].append(starting_point)
                tsv_dict['duration'].append(t_structure[1])
                if t_structure[0] in ['w1', 'w2']:
                    if t_structure[0] == 'w1':
                        current_word = word.split(' ')[0]
                    else:
                        current_word = ' '.join(word.split(' ')[1:])
                    if word == 'neg':
                        current_word = 'neg'
                    tsv_dict['trial_type'].append(current_word)
                    tsv_dict['value'].append(condition)

                else:
                    tsv_dict['trial_type'].append(t_structure[0])
                    tsv_dict['value'].append(t_structure[0])
                starting_point += t_structure[1]

        assert len(list(set([len(v) for k, v in tsv_dict.items()]))) == 1
        ### Adding fixation cross at the beginning
        tsv_dict['onset'].insert(0, 0.)
        tsv_dict['duration'].insert(0, 15.)
        tsv_dict['trial_type'].insert(0, 'fixation')
        tsv_dict['value'].insert(0, 'fixation')
        assert len(list(set([len(v) for k, v in tsv_dict.items()]))) == 1

        out_path = os.path.join(r_folder, 'sub-{:02}_ses-mri_task-{}_run-{:02}_events.tsv'.format(sub, task, trial_i+1))
        pandas.DataFrame.from_dict(tsv_dict).to_csv(out_path,index=False, sep='\t')

    for root, direc, filez in os.walk(out_folder):
        for f in filez:
            if 'json' in f:
                print(f)
                with open(os.path.join(root, f)) as i:
                    lines = json.load(i)
                lines['TaskName'] = task

                with open(os.path.join(root, f), 'w') as i:
                    i.write(json.dumps(lines))
