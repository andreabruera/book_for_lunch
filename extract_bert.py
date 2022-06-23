import argparse
import random
import numpy
import os
import re
import torch

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead

parser = argparse.ArgumentParser()
parser.add_argument('--layer', choices=['middle_four', 'top_four',
                                        'high_four'],
                    required=True, help='Which layer?')
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium',
                                        'geppetto'],
                    required=True, help='Which model?')
parser.add_argument('--tokens', choices=['single_words', 'span_average', 
                    'sentence_average'],
                    required=True, help='How to use words?')
args = parser.parse_args()

if args.model == 'ITBERT':
    model_name = 'dbmdz/bert-base-italian-xxl-cased'
if args.model == 'GILBERTO':
    model_name = 'idb-ita/gilberto-uncased-from-camembert'
if args.model == 'ITGPT2small':
    model_name = 'GroNLP/gpt2-small-italian'
if args.model == 'ITGPT2medium':
    model_name = 'GroNLP/gpt2-medium-italian-embeddings'
if args.model == 'geppetto':
    model_name = 'LorenzoDeMattei/GePpeTto'
if args.model == 'MBERT':
    model_name = 'bert-base-multilingual-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[SEP]')
if 'GeP' in model_name:
    model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto").to('cuda:1')
    required_shape = model.config.n_embd
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
elif 'gpt' in model_name or 'GPT' in model_name:
    model = AutoModel.from_pretrained(model_name).to('cuda:1')
    required_shape = model.embed_dim
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda:1')
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers

entity_vectors = dict()

sentences_folder = os.path.join('resources', 'book_for_lunch_sentences')
#sentences_folder = os.path.join('book_for_lunch_sentences')
#sentences_folder = os.path.join('resources', 'single_words_sentences')

with tqdm() as pbar:
    for f in os.listdir(sentences_folder):
        stimulus = f.split('.')[0]
        entity_vectors[stimulus] = list()
        with open(os.path.join(sentences_folder, f)) as i:
            #print(f)
            lines = [l.strip() for l in i]
        #lines = random.sample(lines, k=min(len(lines), 100000))
        lines = random.sample(lines, k=min(len(lines), 32))
        for l in lines:

            inputs = tokenizer(l, return_tensors="pt")

            if args.tokens == 'span_average':
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
                    correction = list(range(1, len(spans)+1))
                    spans = [max(0, s-c) for s,c in zip(spans, correction)]
                    split_spans = list()
                    for i in list(range(len(spans)))[::2]:
                        current_span = (spans[i], spans[i+1])
                        split_spans.append(current_span)

                    if len(tokenizer.tokenize(l)) > max_len:
                        continue
                    #outputs = model(**inputs, output_attentions=False, \
                    #                output_hidden_states=True, return_dict=True)
                    try:
                        inputs = tokenizer(l, return_tensors="pt").to('cuda:1')
                    except RuntimeError:
                        continue
                    try:
                        outputs = model(**inputs, output_attentions=False, \
                                        output_hidden_states=True, return_dict=True)
                    except RuntimeError:
                        print(l)
                        continue

                    hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                    #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                    for beg, end in split_spans:
                        print(tokenizer.tokenize(l)[beg:end])
                        if len(tokenizer.tokenize(l)[beg:end]) == 0:
                            print(l)
                            continue
                        mention = hidden_states[:, beg:end, :]
                        mention = numpy.average(mention, axis=1)
                        if args.layer == 'middle_four':
                            layer_start = int(n_layers / 2)-2
                            layer_end = int(n_layers/2)+3
                        if args.layer == 'top_four':
                            layer_start = -4
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers+1
                        if args.layer == 'high_four':
                            layer_start = -5
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers-2
                        mention = mention[layer_start:layer_end, :]

                        mention = numpy.average(mention, axis=0)
                        assert mention.shape == (required_shape, )
                        entity_vectors[stimulus].append(mention)
                    pbar.update(1)

            '''
            if mode == 'cls':
                outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)

                last_hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])[-4:, 0, 0, :]
                mention = numpy.average(hidden_states, axis=0)
                assert mention.shape == required_shape
                entity_vectors[stimulus].append(mention)
                pbar.update(1)
            '''

            if args.tokens == 'sentence_average':
                inputs = tokenizer(l, return_tensors="pt").to('cuda:1')
                if len(tokenizer.tokenize(l)) > max_len:
                    continue
                outputs = model(**inputs, output_attentions=False, \
                                output_hidden_states=True, return_dict=True)

                hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])
                ### We leave out the CLS
                hidden_states = hidden_states[:, 0, 1:-1, :]
                if args.layer == 'middle_four':
                    layer_start = int(n_layers / 2)-2
                    layer_end = int(n_layers/2)+2
                if args.layer == 'top_four':
                    layer_start = -4
                    ### outputs has at dimension 0 the final output
                    layer_end = n_layers +1
                if args.layer == 'high_four':
                    layer_start = -5
                    ### outputs has at dimension 0 the final output
                    layer_end = n_layers-2
                ### Averaging tokens
                mention = numpy.average(hidden_states, axis=1)
                assert mention.shape == (n_layers+1, required_shape)
                ### Averaging layers
                mention = numpy.average(mention[layer_start:layer_end, :], axis=0)
                assert mention.shape == (required_shape, )
                entity_vectors[stimulus].append(mention)
                pbar.update(1)

out_folder = os.path.join('resources', '{}_{}_{}'.format(args.model, args.layer, args.tokens))
os.makedirs(out_folder, exist_ok=True)
for k, v in entity_vectors.items():
    with open(os.path.join(out_folder, '{}.vector'.format(k)), 'w') as o:
        for vec in v:
            for dim in vec:
                assert not numpy.isnan(dim)
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
