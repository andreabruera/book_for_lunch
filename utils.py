import numpy
import os

def read_vectors(args):

    vectors = dict()
    for f in os.listdir(args.vectors_folder):
        assert '.vector' in f
        with open(os.path.join(args.vectors_folder, f)) as i:
            if 'selected' in args.vectors_folder:
                vecs = numpy.array([l.strip() for l in i.readlines()], dtype=numpy.float64)
            else:
                vecs = numpy.array([l.strip().split('\t') for l in i.readlines()], dtype=numpy.float64)
        if vecs.shape[0] == 0:
            print(f)
            continue
        else:
            ### Randomizing vector order
            numpy.random.shuffle(vecs)
            ### Limiting to 32 mentions
            vecs = vecs[:32, :]
            if vecs.shape[0] not in [768, 300]:
                vecs = numpy.nanmean(vecs, axis=0)
                '''
                if args.vector_averaging == 'avg':
                    ### Average
                    vecs = numpy.nanmean(vecs, axis=0)
                else:
                    ### Maxpool
                    vecs = numpy.array([max([v[i] for v in vecs]) for i in range(vecs.shape[-1])], dtype=numpy.float64)
                '''
            assert vecs.shape[0] in [768, 300]
            vectors[f.replace('_', ' ').split('.')[0]] = vecs

    return vectors
