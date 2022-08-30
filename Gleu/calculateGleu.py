import json, os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import evaluate
path = 'similarity/data.jsonl'
import numpy as np
from collections import defaultdict
triplesetgleu = defaultdict(list)
triplesetSimilarity = defaultdict(list)
gleuStats=[]
google_bleu = evaluate.load("google_bleu")
outfile1= open('Gleugen/output6-7.jsonl', 'w')
with jsonlines.open(path) as reader:
    with open('Gleugen/output.jsonl', 'w') as outfile:
        for d in reader:
              if len(d['triples']) < 6:
                  score = google_bleu.compute([d['serialized_triples']],[[d['text']]])
                  score= round(score["google_bleu"], 2))
                  triplesetgleu[len(d['triples'])].append(score)
                  d['Gleu'] = score
                  gleuStats.append(score)
                  json.dump(d, outfile)
                  outfile.write('\n')
             elif len(d['triples']) < 8:

                               json.dump(d, outfile1)
                               outfile1.write('\n')
outfile1.close()
print(pd.DataFrame(bleuStats).describe())
fig, ax = plt.subplots()
ax.boxplot(gleuStats,patch_artist=True, meanline=True, showmeans=True)
plt.xticks([1], ["Gleu Score"])
