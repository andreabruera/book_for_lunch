import itertools
import pdb
import os

datasets = ['book_fast', 'lunch_fast',
            #'book_slow', 'lunch_slow'
            ]
analyses = ['time_resolved', 
            'whole_trial',
            'flattened_trial'
            ]
n_folds = [50, \
           #10, 100, 1000
           ]

cvs = ['average_trials', 
       #'replication'
       ]
features = ['no_reduction', 
            'anova'
            ]
spatial = ['ROI']

args = list(itertools.product(datasets, analyses, n_folds, cvs, features, spatial))

for dataset, analysis, n_folds, cv, feature, spatial in args:
    print('--dataset {} --analysis {} --n_folds {} --cross_validation {} --feature_selection {} --spatial_analysis {}'.format(dataset, analysis, n_folds, cv, feature, spatial)) 
    os.system('python3 simple_classification.py --dataset {} --analysis {} --n_folds {} --cross_validation {} --feature_selection {} --spatial_analysis {}'.format(dataset, analysis, n_folds, cv, feature, spatial)) 
