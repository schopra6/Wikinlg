import json, os
import jsonlines
import numpy as np
from scipy.stats.stats import pearsonr
path = 'data.jsonl'
s_filter_list=[]
s_filter_score=[]
record=[]
with jsonlines.open(path) as reader:
    for d in reader:
        s_filter_score.append(d['similarity'])
        record.append(d)

path = '../Simgen/output.jsonl'

f_score=[]
precision=[]
recall =[]

with jsonlines.open(path) as reader:
    for d in reader:
        f_score.append(d['f1'] )
        precision.append(d['precision'])
        recall.append(d['recall'])


diff = np.array(f_score) - np.array(s_filter_score)
#print(s_filter_list)
#print(p_filter_list.index([['Michael Kumbirai', 'occupation', 'rugby union player'], ['Michael Kumbirai', 'date of birth', '09 May 1996']]))
path = '../parent/output.jsonl'

filtered_records = []

f1_rho = pearsonr(f_score ,s_filter_score)[0]
precision_rho = pearsonr(precision, s_filter_score)[0]
recall_rho = pearsonr(recall, s_filter_score)[0]
print(f'precision {precision_rho}  rcall {recall_rho} f1 score {f1_rho}')
for i,e in enumerate(record):
    if diff[i] > 0.1 and f_score[i] > 0.68:
        filtered_records.append(e)
np.save('filtered_records.npy', filtered_records)
