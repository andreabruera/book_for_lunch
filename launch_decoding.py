import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--spatial_analysis', choices=['whole_brain', 'general_semantics',
                                                   'fedorenko_language', 'control_semantics'],
                                          required=True)
parser.add_argument('--senses', action='store_true', default=False)
args = parser.parse_args()

models = [
          'gpt2', 'concreteness', 'fasttext', 
          'ceiling',
          #'gpt2_concreteness', 'fasttext_concreteness'
          ]

for model in models:

    if args.senses:
        os.system('python3 simple_decoding.py --dataset book_fast --analysis whole_trial_flattened --spatial_analysis {} --feature_selection fisher --computational_model {} --n_brain_features 5000 --methodology decoding --senses'.format(args.spatial_analysis, model))
    else:
        os.system('python3 simple_decoding.py --dataset book_fast --analysis whole_trial_flattened --spatial_analysis {} --feature_selection fisher --computational_model {} --n_brain_features 5000 --methodology decoding'.format(args.spatial_analysis, model))
