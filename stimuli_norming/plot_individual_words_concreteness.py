import matplotlib
import numpy

from matplotlib import pyplot

f = 'concreteness_single_words_en.txt'
with open(f) as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
headings = lines[0]
data = lines[1:]
#data = list()
missing = 0

words = [l[0] for l in lines]
concr = [float(l[1]) for l in lines]
cat = [l[2] for l in lines]
meaning = [l[3] for l in lines]

#header = {h : numpy.array([d[h_i] for d in data], dtype=numpy.float64) for h_i, h in enumerate(headings) if 'avg' in h}
#mapper = {d[0] : d[1] for d in data}
colors = {'dot_polysemic' : 'orange',
          'simple_object' : 'black', 'simple_information' : 'darkgray',
          'coercing+simple_object' : 'blue', 'coercing+simple_information' : 'lightblue',
          'coercing_information' : 'darkseagreen', 'coercing_object': 'palegreen', 'simple_simple': 'limegreen'}

'''
### All ratings
fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
#for i, val in enumerate(header.values()):
    ax.violinplot(val, positions=[i], showmeans=True, showextrema=True)
    avg = round(numpy.average(val), 3)
    print(val.shape)
    mdn = round(numpy.median(val), 3)
    ax.text(x=i+0.2, y=avg, s='avg: {}\nmdn: {}'.format(avg, mdn), ha='left', va='center')

ax.set_xticks(range(len(header)))
ax.set_xticklabels(header, ha='right', rotation=45)
ax.set_ylim(ymin=0., ymax=7.)
ax.set_title('Ratings for book-type polysemy cases')

pyplot.savefig('book_ratings.jpg')
pyplot.clf()
pyplot.close()
'''

### Concreteness
#header = {h : [d[h_i] for d in data] for h_i, h in enumerate(headings) if 'std' not in h}
fig, ax = pyplot.subplots(figsize=(16, 9), constrained_layout=True)
#data = zip(header['phrase'], header['concreteness_avg'])
#data = sorted(data, key=lambda item : item[1])
#for i, phr_val in enumerate(data):
for i in range(len(words)):
    w = words[i]
    c = concr[i]
    ca = cat[i]
    m = meaning[i]
    ax.scatter(x=i, y=float(c)/5, color=colors['{}_{}'.format(ca, m)], s=100.)
    ax.text(x=i, y=(c/5)-0.1, s=w, rotation=45, va='center', ha='right', fontsize=15)
'''
ax.scatter(x=0., y=5., color='gray', s=100.)
ax.scatter(x=1., y=5., color='black', s=100.)
ax.text(x=2., y=5., s='simple V + simple N', va='center', ha='left', fontsize=15)
ax.scatter(x=0, y=5.25, color='lightblue', s=100.)
ax.scatter(x=1., y=5.25, color='blue', s=100.)
ax.text(x=2., y=5.25, s='coercing V + simple N', va='center', ha='left', fontsize=15)
ax.scatter(x=0., y=5.5, color='yellow', s=100.)
ax.scatter(x=1., y=5.5, color='orange', s=100.)
ax.text(x=2., y=5.5, s='coercing V + dot N', va='center', ha='left', fontsize=15)
ax.text(x=0., y=5.7, s='information', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
ax.text(x=1., y=5.7, s='object', rotation=90, va='bottom', ha='center', fontsize=15, weight='normal')
ax.tick_params(axis='y', which='major', labelsize=15)
'''

ax.hlines(xmin=0., xmax=len(data), y=[0.,.2,.4,.6,.8,1.], alpha=0.3, color='darkgray', linestyles='dashed')
ax.set_ylabel('Average concreteness rating', fontsize=20)
ax.set_ylim(top=1.1, bottom=0.)
ax.get_xaxis().set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('Concreteness ratings for individual words\nElicited in English by Brysbaert et al. 2014', fontsize=20.)

pyplot.savefig('concreteness_individual_ratings.jpg')
pyplot.clf()
pyplot.close()

