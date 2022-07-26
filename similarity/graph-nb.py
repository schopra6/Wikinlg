import json, os
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
path = 'data.jsonl'
n_bins = [1,2,3,4,5,6,7,8]
with jsonlines.open(path) as reader:
    number_of_triples = [len(d['triples']) for d in reader]
    fig, ax = plt.subplots(figsize=(8, 4))
number_of_triples=[1,2,4]
# plot the cumulative histogram
n, bins, patches = ax.hist(number_of_triples, n_bins, histtype='barstacked')
print(n)
ax.grid(True)
ax.set_title(' histograms')
ax.set_ylabel('total occurance')

fig.savefig('nb_triple_distribution.jpg')
