import numpy
import os

def read_vectors(args):

    if args.computational_model == 'gpt2':
        vec_folder = os.path.join('resources', 'ITGPT2medium_top_four_span_average')
    if args.computational_model == 'fasttext':
        vec_folder = os.path.join('resources', 'fasttext_concatenated')
    vectors = dict()
    for f in os.listdir(vec_folder):
        assert '.vector' in f
        with open(os.path.join(vec_folder, f)) as i:
            vecs = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float64)
        if vecs.shape[0] == 0:
            print(f)
            continue
        else:
            #vecs = vecs[0]
            ### First vector is for mention
            #first_vec = vecs[0, :].reshape(1, -1)
            ### Randomizing vector order
            #vecs = numpy.random.shuffle(vecs[1:])
            ### Limiting to 32 mentions
            #vecs = numpy.concatenate((first_vec, vecs[:32, :]), axis=0)
            if vecs.shape[0] not in [1024, 768, 300, 600]:
                vecs = numpy.nanmean(vecs, axis=0)
            assert vecs.shape[0] in [1024, 768, 300, 600]
            vectors[f.replace('_', ' ').split('.')[0]] = vecs

    return vectors
