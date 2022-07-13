import itertools
import numpy
import os


folder = 'new_voxel_selection/fisher_scores/4_to_11/book_fast/whole_trial_flattened'

cases = dict()

for f in os.listdir(folder):
    cases[f] = list()
    for root, direc, filez in os.walk(os.path.join(folder, f)):
        for fil in filez:
            with open(os.path.join(root, fil)) as i:
                lines = numpy.array([l.strip().split('\t')[1:] for l in i.readlines()], dtype=numpy.float32)
            lines = numpy.average(lines, axis=0)
            if f == 'whole_brain':
                dimensionality = lines.shape[0]
            cases[f].append(lines)
    cases[f] = numpy.average(numpy.vstack(cases[f]), axis=0)
    cases[f] = [f_i for f_i, f in sorted([(f_i, f) for f_i, f in enumerate(cases[f])], 
                                          key=lambda item : item[1], reverse=True)][:5000]

for c_one, c_two in itertools.combinations(cases.keys(), 2):
    total = list(set(cases[c_one] + cases[c_two]))
    print([c_one, c_two, ((10000-len(total))/10000)*100])

del cases['whole_brain']
counts = dict()
for k, v in cases.items():
    for val in v:
        if val not in counts.keys():
            counts[val] = 1
        else:
            counts[val] += 1

best_feats = [v for v in counts.values() if v==3]
print(len(best_feats))
with open('best_features.txt', 'w') as o:
    for l in range(dimensionality):
        if l in best_feats:
            val = 0.1
        else:
            val = 0.
        o.write('{}\t'.format(float(val)))
