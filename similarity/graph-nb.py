import json, os
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
path = 'data.jsonl'
n_bins = [1,2,3,4,5,6,7,8]
with jsonlines.open(path) as reader:
    number_of_triples = [len(d['triples']) for d in reader]
    fig, ax = plt.subplots(figsize=(8, 4))
# plot the cumulative histogram
n, bins, patches = ax.hist(number_of_triples, n_bins, histtype='barstacked',linewidth=1,alpha=0.8,width=0.8)
print(n)
ax.grid(True)
current_values = ax.get_yticks()
ax.set_xticklabels('')
#ax.set_yticklabels(['{:.0f}'.format(x) for x in current_values])
ax.xaxis.set_minor_locator(ticker.FixedLocator([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]))
ax.set_xticklabels(['1','2','3','4','5','6','7','8'], minor=True)

ax.set_title(' histogram')
ax.set_ylabel('total occurance')
ax.set_xlabel('number of triples')
fig.savefig('nb_triple_distribution update.jpg')
