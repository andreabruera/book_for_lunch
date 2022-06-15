import argparse

import fasttext
import numpy
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['concatenated', 'averaged'], required=True, 
                    help='How to finalize vectors?')
args = parser.parse_args()

ft = fasttext.load_model(os.path.join('resources', 'cc.it.300.bin'))

folder = os.path.join('resources', 'ITGPT2medium_top_four_span_average')
for f in tqdm(os.listdir(folder)):
    if 'vector' in f:
        words = f.split('.')[0].split('_')
        if args.mode == 'concatenated':
            vec = numpy.concatenate([ft.get_word_vector(w) for w in words])
            assert vec.shape == (600, )
        else:
            vec = numpy.average([ft.get_word_vector(w) for w in words], axis=0)
            assert vec.shape == (300, )
        new_folder = os.path.join('resources', 'fasttext_{}'.format(args.mode))
        os.makedirs(new_folder, exist_ok=True)
        with open(os.path.join(new_folder, f), 'w') as o:
            for d in vec:
                o.write('{}\t'.format(d))
