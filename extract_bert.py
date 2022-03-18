import argparse
import numpy
import os
import re
import torch

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument('--layer', choices=['middle_four', 'top_four'],
                    required=True, help='Which layer?')
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium'],
                    required=True, help='Which model?')
args = parser.parse_args()

if args.model == 'ITBERT':
    model_name = 'dbmdz/bert-base-italian-xxl-cased'
if args.model == 'GILBERTO':
    model_name = 'idb-ita/gilberto-uncased-from-camembert'
if args.model == 'ITGPT2small':
    model_name = 'GroNLP/gpt2-small-italian'
if args.model == 'ITGPT2medium':
    model_name = 'GroNLP/gpt2-medium-italian-embeddings'
if args.model == 'MBERT':
    model_name = 'bert-base-multilingual-cased'
#mode = 'cls'
#mode = 'average'
mode = 'span'

if 'large' in model_name or 'medium' in model_name:
    required_shape = (1024,)
else:
    required_shape = (768, )

tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[SEP]')
if 'gpt' in model_name:
    model = AutoModel.from_pretrained(model_name).to('cuda:1')
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda:1')


entity_vectors = dict()

sentences_folder = os.path.join('resources', 'book_for_lunch_sentences')

with tqdm() as pbar:
    for f in os.listdir(sentences_folder):
        stimulus = f.split('.')[0]
        entity_vectors[stimulus] = list()
        with open(os.path.join(sentences_folder, f)) as i:
            #print(f)
            for l in i:
                l = l.strip()
                inputs = tokenizer(l, return_tensors="pt")

                if mode == 'span':
                    spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                            i==tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
                    if 'bert' in model_name and len(spans)%2==1:
                        spans = spans[:-1]
                        #i==102
                        #i == 103 
                        #or 
                        #or 
                        #i==30000
                            
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

                        max_len = 512 if 'bert' in model_name else 1024
                        if len(tokenizer.tokenize(l)) > max_len:
                            continue
                        #outputs = model(**inputs, output_attentions=False, \
                        #                output_hidden_states=True, return_dict=True)
                        try:
                            inputs = tokenizer(l, return_tensors="pt").to('cuda:1')
                        except RuntimeError:
                            import pdb; pdb.set_trace()
                            continue
                        try:
                            outputs = model(**inputs, output_attentions=False, \
                                            output_hidden_states=True, return_dict=True)
                        except IndexError:
                            import pdb; pdb.set_trace()
                            print(l)
                            continue

                        hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                        #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                        for beg, end in split_spans:
                            mention = hidden_states[:, beg:end, :]
                            mention = numpy.average(mention, axis=1)
                            if args.layer == 'middle_four':
                                layer_start = -8
                                layer_end = -4
                            if args.layer == 'top_four':
                                layer_start = -4
                                layer_end = mention.shape[0]
                            mention = mention[layer_start:layer_end, :]

                            mention = numpy.average(mention, axis=0)
                            assert mention.shape == required_shape
                            entity_vectors[stimulus].append(mention)
                        pbar.update(1)

                if mode == 'cls':
                    outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)

                    last_hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])[-4:, 0, 0, :]
                    mention = numpy.average(hidden_states, axis=0)
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

out_folder = os.path.join('resources', '{}_{}_big'.format(args.model, args.layer))
os.makedirs(out_folder, exist_ok=True)
for k, v in entity_vectors.items():
    with open(os.path.join(out_folder, '{}.vector'.format(k)), 'w') as o:
        for vec in v:
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
