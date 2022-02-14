import os

datasets = ['book_fast', 'lunch_fast',
            'book_slow', 'lunch_slow'
            ]

for d in datasets:
    os.system('python3 fisher_scores.py --dataset {} --analysis whole_trial'.format(d))
