import collections
import numpy
import os
import pandas

files = list()
folder = 'norming_data'

for root, direc, filez in os.walk(folder):
    for f in filez:
        if 'xls' in f:
            if f not in files:
                files.append(os.path.join(root, f))

collector = collections.defaultdict(list)
for f in files:
    try:
        f = pandas.read_excel(f)
    except ValueError:
        print(f)
        continue
    length = f.shape[0]
    data = {k : list(v.values()) for k, v in f.to_dict().items()}
    data = [[v[i] for v in data.values()] for i in range(length)]
    for d in data:
        collector[d[0]].append(d[1:])

### Clean stimuli mention
collector = {tuple(k.replace("'", ' ').split()) : v for k, v in collector.items() if isinstance(k, str)}
collector = {' '.join([k[0], k[-1]]) : v for k, v in collector.items() if len(k)>2}
size = {k : len(v) for k, v in collector.items()}
collector_mean = {k : numpy.nanmean(v, axis=0) for k, v in collector.items()}
collector_std = {k : numpy.nanstd(v, axis=0) for k, v in collector.items()}
collector_mean['annullare aperitivo'] = collector_mean['annullare apertivo']
collector_std['annullare aperitivo'] = collector_std['annullare apertivo']
size['annullare aperitivo'] = size['annullare apertivo']

### Read stimuli
with open('lunch_fast_stimuli.tsv') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
stimuli = [k[0].replace("'", ' ').split() for k in lines]
stimuli = [' '.join([k[0], k[-1]]) for k in stimuli]
for s in stimuli:
    if s not in collector.keys():
        print(s)
columns = f.columns[1:4].to_list()
columns = ['concreteness' if 'conc' in w else w for w in columns]
columns = ['familiarity' if 'fam' in w else w for w in columns]
columns = ['imageability' if 'im' in w else w for w in columns]

with open('lunch_fast_stimuli_ratings.tsv', 'w') as o:
    o.write('phrase\tcategory\t{}_avg\t{}_std\t{}_avg\t{}_std\t{}_avg\t{}_std\n'.format(
                   columns[0], columns[0], columns[1], columns[1], columns[2], columns[2]))
    for s_i, s in enumerate(stimuli):
        print(size[s])
        o.write('{}\t'.format(s))
        o.write('{}\t'.format(lines[s_i][1]))
        o.write('{}\t'.format(collector_mean[s][0]))
        o.write('{}\t'.format(collector_std[s][0]))
        o.write('{}\t'.format(collector_mean[s][1]))
        o.write('{}\t'.format(collector_std[s][1]))
        o.write('{}\t'.format(collector_mean[s][2]))
        o.write('{}\t'.format(collector_std[s][2]))
        o.write('\n')
