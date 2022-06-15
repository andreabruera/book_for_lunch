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
            #vecs = vecs[0]
            ### First vector is for mention
            #first_vec = vecs[0, :].reshape(1, -1)
            ### Randomizing vector order
            #vecs = numpy.random.shuffle(vecs[1:])
            ### Limiting to 32 mentions
            #vecs = numpy.concatenate((first_vec, vecs[:32, :]), axis=0)
            #vecs = vecs[:, :]
            if vecs.shape[0] not in [1024, 768, 300, 600]:
                vecs = numpy.nanmean(vecs, axis=0)
                '''
                if args.vector_averaging == 'avg':
                    ### Average
                    vecs = numpy.nanmean(vecs, axis=0)
                else:
                    ### Maxpool
                    vecs = numpy.array([max([v[i] for v in vecs]) for i in range(vecs.shape[-1])], dtype=numpy.float64)
                '''
            assert vecs.shape[0] in [1024, 768, 300, 600]
            vectors[f.replace('_', ' ').split('.')[0]] = vecs

    return vectors
