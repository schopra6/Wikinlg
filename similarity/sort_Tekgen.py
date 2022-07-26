import json, os
import jsonlines
import numpy as np
from scipy.stats.stats import pearsonr
path = 'data.jsonl'
s_filter_list=[]
s_filter_score=[]
with jsonlines.open(path) as reader:
    for d in reader:
        s_filter_list.append(d['triples'])
        s_filter_score.append(d['similarity'])

path = '../corpus/newfile.tsv'
p_filter_list=[]
f_score=[]
precision=[]
recall =[]
with jsonlines.open(path) as reader:
   for d in reader:

    p_filter_list.append(d['triples'] )
    f_score.append(d['f1'] )
    precision.append(d['precision'])
    recall.append(d['recall'])

sorted_similarity = [s_filter_score[s_filter_list.index(element)] for element in p_filter_list ]

np.save('sorted_similarity.npy', sorted_similarity)

diff = np.array(f_score) - np.array(sorted_similarity)
#print(s_filter_list)
#print(p_filter_list.index([['Michael Kumbirai', 'occupation', 'rugby union player'], ['Michael Kumbirai', 'date of birth', '09 May 1996']]))
path = '../corpus/newfile.tsv'
with jsonlines.open(path) as reader:
     records = [d for d in reader]
filtered_records = []

f1_rho = pearsonr(f_score ,sorted_similarity)[0]
precision_rho = pearsonr(precision, sorted_similarity)[0]
recall_rho = pearsonr(recall, sorted_similarity)[0]
\)
print(f'precision {precision_rho}  rcall {recall_rho} f1 score {f1_rho}')
for i,e in enumerate(records):
   if diff[i] > 0.1 and f_score[i] > 0.68:
       filtered_records.append(e)
np.save('filtered_records.npy', filtered_records)
