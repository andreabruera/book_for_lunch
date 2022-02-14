import numpy
import os
import re
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = 'dbmdz/bert-base-italian-xxl-cased'
#mode = 'cls'
#mode = 'average'
mode = 'span'

if 'large' in model_name:
    required_shape = (1024,)
else:
    required_shape = (768, )

tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[SEP]')
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda:1')

entity_vectors = dict()

sentences_folder = 'sentences'

with tqdm() as pbar:
    for f in os.listdir(sentences_folder):
        stimulus = f.split('.')[0]
        entity_vectors[stimulus] = list()
        with open(os.path.join(sentences_folder, f)) as i:
            print(f)
            for l in i:
                l = l.strip()
                inputs = tokenizer(l, return_tensors="pt")

                if mode == 'span':
                    spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if i == 103][:-1]
                    if len(spans) > 1:
                        try:
                            assert len(spans) % 2 == 0
                        except AssertionError:
                            print(l)
                            continue
                        l = re.sub(r'\[SEP\]', '', l)
                        ### Correcting spans
                        correction = list(range(len(spans)))
                        spans = [s-c for s,c in zip(spans, correction)]
                        split_spans = list()
                        for i in list(range(len(spans)))[::2]:
                            current_span = (spans[i], spans[i+1])
                            split_spans.append(current_span)
                        inputs = tokenizer(l, return_tensors="pt").to('cuda:1')

                        try:
                            outputs = model(**inputs, output_attentions=False, \
                                            output_hidden_states=True, return_dict=True)
                        except RuntimeError:
                            print(l)
                            continue

                        last_hidden_states = numpy.array([k.cpu().detach().numpy() for k in \
                                                  outputs['hidden_states']])[-4:, 0, :]
                        #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                        for beg, end in split_spans:
                            mention = last_hidden_states[:, beg:end, :]
                            mention = numpy.average(mention, axis=1)
                            mention = numpy.average(mention, axis=0)
                            assert mention.shape == required_shape
                            entity_vectors[stimulus].append(mention)
                        pbar.update(1)

                if mode == 'cls':
                    outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)

                    last_hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])[-4:, 0, 0, :]
                    mention = numpy.average(last_hidden_states, axis=0)
                    assert mention.shape == required_shape
                    entity_vectors[stimulus].append(mention)
                    pbar.update(1)

                if mode == 'average':
                    outputs = model(**inputs, output_attentions=False, \
                                    output_hidden_states=True, return_dict=True)

                    last_hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])[-4:, 0, 1:]
                    mention = numpy.average(last_hidden_states, axis=1)
                    mention = numpy.average(mention, axis=0)
                    assert mention.shape == required_shape
                    entity_vectors[stimulus].append(mention)
                    pbar.update(1)

out_folder = os.path.join('vectors', model_name.replace('/', '_'), mode)
os.makedirs(out_folder, exist_ok=True)
for k, v in entity_vectors.items():
    with open(os.path.join(out_folder, '{}.vector'.format(k)), 'w') as o:
        for vec in v:
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
