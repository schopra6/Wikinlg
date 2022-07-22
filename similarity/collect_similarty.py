import numpy as np
import json, os
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
path = f'data.jsonl'
n_bins = 50
with jsonlines.open(path) as reader:
    similarity = [d['similarity'] for d in reader]
np.save('similarity.npy', similarity)
fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(similarity, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
ax.set_ylabel('total occurance')

fig.savefig('cumulative distribution.jpg')






