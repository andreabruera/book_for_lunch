import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--spatial_analysis', choices=['whole_brain', 'general_semantics',
                                                   'fedorenko_language', 'control_semantics'],
                                          required=True)
args = parser.parse_args()

models = ['ceiling', 'gpt2', 'concreteness', 'fasttext', 'gpt2_concreteness', 'gpt2_fasttext']

for model in models:

    os.system('python3 simple_decoding.py --dataset book_fast --analysis whole_trial_flattened --spatial_analysis {} --feature_selection fisher --computational_model {} --n_brain_features 5000 --methodology decoding'.format(args.spatial_analysis, model))
