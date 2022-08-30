import json, os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats.stats import pearsonr
import numpy as np
path = '../Bleugen/output.jsonl'
from collections import defaultdict
triplesetbleu = defaultdict(list)
triplesetSimilarity = defaultdict(list)
bleubefore = defaultdict(int)
bleuafter = defaultdict(int)
bleuStats=[]
def percentage(part, whole):
  return 100 * float(part)/float(whole)
with jsonlines.open(path) as reader:

        for d in reader:
              #score = sacrebleu.corpus_bleu([d['serialized_triples']],[[d['text']]]).score
              bleubefore[len(d['triples'])]= bleubefore[len(d['triples'])] + 1
              if d['Bleu'] >4.04:
                 bleuafter[len(d['triples'])]= bleuafter[len(d['triples'])] + 1
              triplesetbleu[len(d['triples'])].append(d['Bleu'])
              triplesetSimilarity[len(d['triples'])].append(d['similarity'])
#print(bleubefore)
#print(bleuafter)

for i in range(1,11):
 #print(f'bleu before {k} : {bleubefore[k]}')
 print(percentage(bleuafter[i],bleubefore[i]),end =' & ')
  #print(f'bleu after {i} : {bleuafter[i]}')
 #print(f'bleu for size {k} : {np.mean(triplesetbleu[k])}')
 #print(f'bleu for size {k} : {np.mean(triplesetSimilarity[k])}')

