import itertools
import scipy
import torch
import transformers

from scipy import stats
from transformers import AutoModel,AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

model_name = 'GroNLP/gpt2-medium-italian-embeddings'
model_name = "GroNLP/gpt2-small-italian"
model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda:1')
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

with open('book_fast_stimuli_ratings.tsv') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]
stimuli = [(l[0], l[1]) for l in lines]

cases = dict()

with open('book_fast_stimuli_perplexity_gpt2.tsv', 'w') as o:
    for s, cat in stimuli:
        cat = cat.replace('dot', 'coercion').replace('simple', 'light').replace('verb', 'transparent')
        cat = cat.replace('concrete', 'object').replace('abstract', 'information')
        if cat.split('_')[0] not in cases.keys():
            cases[cat.split('_')[0]] = list()
        input_ids = tokenizer(s, return_tensors='pt')['input_ids'].to('cuda:1')
        #input_ids = input_ids[:, :-1]
        targets = input_ids.clone()
        #targets[:, :] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=targets, return_dict=True)
            neg_log_likelihood = outputs.loss.cpu().numpy() * 2
        o.write('{}\t{}\t{}\n'.format(s, neg_log_likelihood, cat))
        cases[cat.split('_')[0]].append(neg_log_likelihood)

for c_one, c_two in itertools.combinations(cases.keys(), 2):
    one = cases[c_one]
    two = cases[c_two]
    alternative = 'two-sided'
    if 'light' in [c_one, c_two]:
        alternative = 'greater' if c_one=='light' else 'less'
    diff = scipy.stats.ttest_ind(one, two, alternative=alternative)
    print([c_one, c_two, diff])
