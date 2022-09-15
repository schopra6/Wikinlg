import json, os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats.stats import pearsonr
import numpy as np
path = 'textgen/parentoutput.jsonl'
from collections import defaultdict
triplesetbleu = defaultdict(list)
triplesetSimilarity = defaultdict(list)
bleubefore = defaultdict(int)
bleuafter = defaultdict(int)
gleubefore = defaultdict(int)
gleuafter = defaultdict(int)
similaritybefore = defaultdict(int)
similarityafter = defaultdict(int)
parentbefore = defaultdict(int)
parentafter = defaultdict(int)
bleuStats=[]
def percentage(part, whole):
  return 100 * float(part)/float(whole)
with jsonlines.open(path) as reader:

        for d in reader:
              #score = sacrebleu.corpus_bleu([d['serialized_triples']],[[d['text']]]).score
              bleubefore[len(d['triples'])]= bleubefore[len(d['triples'])] + 1
              if d['bleu'] >=8.53:
                 bleuafter[len(d['triples'])]= bleuafter[len(d['triples'])] + 1
              gleubefore[len(d['triples'])]= gleubefore[len(d['triples'])] + 1
              if d['gleu'] >=0.15:
                  gleuafter[len(d['triples'])]= gleuafter[len(d['triples'])] + 1
              similaritybefore[len(d['triples'])]= similaritybefore[len(d['triples'])] + 1
              if d['similarity'] >=0.:
                  similarityafter[len(d['triples'])]= similarityafter[len(d['triples'])] + 1
              parentbefore[len(d['triples'])]= parentbefore[len(d['triples'])] + 1
              if d['f1'] >=0.49:
                  parentafter[len(d['triples'])]= parentafter[len(d['triples'])] + 1
#print(bleubefore)
#print(bleuafter)

for i in range(1,6):
 #print(f'bleu before {k} : {bleubefore[k]}')
 print(percentage(bleuafter[i],bleubefore[i]),end =' & ')
  #print(f'bleu after {i} : {bleuafter[i]}')
 #print(f'bleu for size {k} : {np.mean(triplesetbleu[k])}')
 #print(f'bleu for size {k} : {np.mean(triplesetSimilarity[k])}')

for i in range(1,6):
    #print(f'bleu before {k} : {bleubefore[k]}')
    print(percentage(gleuafter[i],gleubefore[i]),end =' & ')
for i in range(1,6):
    #print(f'bleu before {k} : {bleubefore[k]}')
    print(percentage(similaritybefore[i],similarityafter[i]),end =' & ')
for i in range(1,6):
    #print(f'bleu before {k} : {bleubefore[k]}')
    print(percentage(parentbefore[i],parentafter[i]),end =' & ')
