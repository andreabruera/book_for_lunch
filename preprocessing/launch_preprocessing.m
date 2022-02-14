addpath ('.')
%SPM folder should be in book_for_lunch/preprocessing
addpath ('spm12')
%basedir = '/import/cogsci/andrea/dataset/neuroscience/dot_lunch_fast_bids';
%basedir = '/import/cogsci/andrea/dataset/neuroscience/dot_book_fast_bids';
basedir = '/import/cogsci/andrea/dataset/neuroscience/dot_book_slow_bids';
%basedir = '/import/cogsci/andrea/dataset/neuroscience/dot_lunch_slow_bids';
if contains(basedir, 'lunch') == 1
    if contains(basedir, 'slow') == 1
        n_subjects = 11
    else
        n_subjects = 15
    end
else
    if contains(basedir, 'slow') == 1
        n_subjects = 12
    else
        n_subjects = 16
    end
end

for s = 1:n_subjects;
    % specify the number of functional runs acquired
    if contains(basedir, 'lunch') == 1
        if s == 1 
            nruns=5;                
        else
            nruns=6;
        end
    else
        nruns = 6
    end
    try
        preprocessing_BIDS(s, nruns, basedir)
    catch
        warning('Problem with a subject')
    end
end
