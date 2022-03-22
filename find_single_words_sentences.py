import os
import resources.lemmatize_italian_data.pymorphit_cls
import re

from tqdm import tqdm

book_f = '/import/cogsci/andrea/dataset/neuroscience/dot_book_fast_bids/derivatives/sub-01/ses-mri/func/sub-01_ses-mri_task-dotbookfast_run-01_events.tsv'
lunch_f = '/import/cogsci/andrea/dataset/neuroscience/dot_lunch_fast_bids/derivatives/sub-01/ses-mri/func/sub-01_ses-mri_task-dotlunchfast_run-01_events.tsv'
stimuli = list()
for events_path in [book_f, lunch_f]:
    with open(events_path) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    events = {h : [l[h_i] for l in lines[1:]] for h_i, h in enumerate(lines[0])}
    if 'slow' in events_path:
        jump = 4 
        verb_idx = 0
        noun_idx = 1
    elif 'fast' in events_path:
        jump = 7
        verb_idx = 1 
        noun_idx = 3
    trial_starts = [int(round(float(f), 0))-1 for f in events['onset'][1:][::jump]]
    ### Checking that trial length is between 10 and 12 seconds
    for t_i, t in enumerate(trial_starts):
        if t!=trial_starts[-1]:
            assert trial_starts[t_i+1]-t > 9
            assert trial_starts[t_i+1]-t < 13
    trial_infos = {'start' : list(), 'stimulus' : list(), 'category' : list()}
    for t_i, t in enumerate(list(range(len(events['onset'])))[1:][::jump]):
        stimulus = events['trial_type'][t:t+jump]
        stimulus = '{} {}'.format(stimulus[verb_idx], stimulus[noun_idx])
        stimuli.append(stimulus)

sentences = {tuple(k.replace("'", ' ').split(' ')) : k for k in stimuli}
sentences = {(k[0], k[-1]) if len(k)==3 else k : ['[SEP] {} [SEP]'.format(v)] for k, v in sentences.items()}
del sentences[('neg', 'neg')]
print('Itwac')

### ItWac sentences
counter = 0
folders = ['/import/cogsci/andrea/dataset/corpora/itwac'] 
with tqdm() as pbar:
    for folder in folders:
        for root, direc, filez in os.walk(folder):
            for f in filez:
                f_sentences = list()
                f_lemmas = list()

                sentence = list()
                lemma = list()
                #with open(os.path.join(root, f), errors='ignore') as i:
                with open(os.path.join(root, f), encoding='latin-1') as i:
                    for l in i.readlines():
                        l = l.encode('utf-8').decode().strip()
                        if l == '<s>':
                            pass
                        elif l == '</s>':
                            f_sentences.append(sentence)
                            sentence = list()
                            f_lemmas.append(lemma)
                            lemma = list()
                        elif '<' not in l:
                            l = l.split('\t')
                            lemma.append(l[-1])
                            sentence.append(l[0])

                ### Making sentences shorter
                short_sentences = list()
                short_lemmas = list()
                for split_s, split_lem in zip(f_sentences, f_lemmas):
                    if len(split_s) < 128:
                        short_sentences.append(split_s)
                        short_lemmas.append(split_lem)
                    else:
                        sent = list()
                        sent_lem = list()
                        for t, l_t in zip(split_s, split_lem):
                            sent.append(t)
                            sent_lem.append(l_t)
                            if '.' in t or ';' in t:
                                if len(sent) > 128:
                                    short_sentences.append(sent)
                                    short_lemmas.append(sent_lem)
                                    sent = list()
                                    sent_lem = list()
                        short_sentences.append(sent)
                        short_lemmas.append(sent_lem)

                for lemmaed, l in zip(short_lemmas, short_sentences):
                    assert isinstance(lemmaed, list)
                    assert isinstance(l, list)
                    #for lemmaed, l in zip(f_lemmas, f_sentences):
                    for one, two in sentences.keys():
                        indices_one = [i for i, t in enumerate(lemmaed) if t==one]
                        indices_two = [i for i, t in enumerate(lemmaed) if t==two]
                        if len(indices_one)>1 and len(indices_two)>1:
                            for idx_one in indices_one:
                                for idx_two in indices_two:
                                    if idx_two-idx_one in [1, 2, 3]:
                                        try:
                                            l[idx_one] = '[SEP] {}'.format(l[idx_one])
                                            l[idx_two] = '{} [SEP]'.format(l[idx_two])
                                        except TypeError:
                                            pass
                            l = ' '.join(l)
                            if '[SEP]' in l:
                                print(l)
                                sentences[(one, two)].append(l)
                                pbar.update(1)

print('other')
lemmatizer = pymorphit_cls.PyMorphITCLS()

### To be lemmatized
folders = ['/import/cogsci/andrea/dataset/corpora/opensubs_it_ready', 
            '/import/cogsci/andrea/dataset/corpora/wikipedia_italian/it_wiki_article_by_article/']

counter = 0
with tqdm() as pbar:
    for folder in folders:
        for root, direc, filez in os.walk(folder):
            for f in filez:
                with open(os.path.join(root, f), errors='ignore') as i:
                    for l in i.readlines():
                        split_s = l.strip()
                        split_s = re.sub(r'\s+[\']\s+', r"' ", split_s).split()
                        split_lem = lemmatizer.lemmatize_line(l, mode='Q').split()
                        if len(split_s) == len(split_lem):
                            ### Making sentences shorter
                            if len(split_s) < 128:
                                short_sentences = [split_s]
                                short_lemmas = [split_lem]
                            else:
                                short_sentences = list()
                                short_lemmas = list()
                                sent = list()
                                sent_lem = list()
                                for t, l_t in zip(split_s, split_lem):
                                    sent.append(t)
                                    sent_lem.append(l_t)
                                    if '.' in t:
                                        if len(sent) > 128:
                                            short_sentences.append(sent)
                                            short_lemmas.append(sent_lem)
                                            sent = list()
                                            sent_lem = list()
                                short_sentences.append(sent)
                                short_lemmas.append(sent_lem)

                            for l, lemmaed in zip(short_sentences, short_lemmas):
                                assert isinstance(lemmaed, list)
                                assert isinstance(l, list)
                                for one, two in sentences.keys():
                                    indices_one = [i for i, t in enumerate(lemmaed) if t==one]
                                    indices_two = [i for i, t in enumerate(lemmaed) if t==two]
                                    if len(indices_one)>1 and len(indices_two)>1:
                                        for idx_one in indices_one:
                                            for idx_two in indices_two:
                                                if idx_two-idx_one in [1, 2, 3]:
                                                    try:
                                                        l[idx_one] = '[SEP] {}'.format(l[idx_one])
                                                        l[idx_two] = '{} [SEP]'.format(l[idx_two])
                                                    except TypeError:
                                                        pass
                                        l = ' '.join(l)
                                        if '[SEP]' in l:
                                            print(l)
                                            sentences[(one, two)].append(l)
                                            pbar.update(1)



out = 'book_for_lunch_sentences'
os.makedirs(out, exist_ok=True)
for k, v in sentences.items():
    with open(os.path.join(out, '{}.vector'.format('_'.join(k))), 'w') as o:
        for sent in v:
            sent = sent.replace('\n', ' ')
            o.write('{}\n'.format(sent))
