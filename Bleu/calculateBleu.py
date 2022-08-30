import json, os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sacrebleu
from scipy.stats.stats import pearsonr
path = 'similarity/data.jsonl'
from collections import defaultdict
triplesetbleu = defaultdict(list)
triplesetSimilarity = defaultdict(list)
bleuStats=[]
with jsonlines.open(path) as reader:
    with open('Bleugen/output.jsonl', 'w') as outfile:
        for d in reader:
            score = sacrebleu.corpus_bleu([d['serialized_triples']],[[d['text']]]).score
            triplesetbleu[len(d['triples'])].append(score)
            triplesetSimilarity[len(d['triples'])].append(d['similarity'])
            d['Bleu'] = score
            bleuStats.append(score)
            json.dump(d, outfile)
            outfile.write('\n')

print(pd.DataFrame(bleuStats).describe())
fig, ax = plt.subplots()
ax.boxplot(bleuStats,patch_artist=True, meanline=True, showmeans=True)
plt.xticks([1], ["Bleu Score"])
for i in range(1,8):
    print(f'correlation for size {i} {pearsonr(triplesetbleu[i] ,triplesetSimilarity[i])[0]}')