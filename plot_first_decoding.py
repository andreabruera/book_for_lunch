import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats

folder = 'first_decoding'

collector = dict()
for root, direc, filez in os.walk(folder):
    for fil in filez:
        if 'results' in fil:
            with open(os.path.join(root, fil)) as i:
                try:
                    lines = numpy.array([l.strip() for l in i.readlines()], dtype = numpy.float64)
                except ValueError:
                    continue

            fold = root.split('/')[-6]
            collector[fold] = lines

fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
positions = list(range(len(collector.keys())))
for pos, data in zip(positions, collector.items()):
    ax.violinplot(data[1], positions=[pos], showmeans=True, showextrema=True)
    avg = round(numpy.average(data[1]), 3)
    mdn = round(numpy.median(data[1]), 3)
    p = round(scipy.stats.wilcoxon(data[1]-.5, alternative='greater')[1], 4)
    ax.text(x=pos+0.2, y=avg, s='avg: {}\nmdn: {}\np: {}'.format(avg, mdn, p), ha='left', va= 'center')
ax.set_ylim(ymin=0.0, ymax=1.0)
ax.hlines(xmin=-0.5, xmax=max(positions)+0.5, y=0.5, alpha=0.5, color='darkgray', linestyle='dashdot')
ax.set_xticks(positions)
ax.set_xticklabels(list(collector.keys()), rotation=45, ha='right')
ax.set_title('Word vector decoding for book-type polysemic samples')

pyplot.savefig('first_decoding.jpg')

