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
        s_filter_list.append(d['triples'])
        s_filter_score.append(d['similarity'])
        record.append(d)

path = '../Textgen/output.jsonl'
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

sorted_p = []
sorted_r = []
sorted_f = []
exclude = []
indexes=[]
sorted_similarity
print('reading done 1')
for i,element in enumerate(s_filter_list) :
    try:
       index = p_filter_list.index(element)
       indexes.append(index)
       sorted_p.append(precision[index])
       sorted_f.append(f_score[index])
       sorted_r.append(recall[index])
    except:
        exclude.append(i)
np.save('indexes,npy',indexes)

print('sorting done')
for index in sorted(exclude, reverse=True):
    del s_filter_list[index]
    del s_filter_score[index]
    print(f'out of index {index}')



diff = np.array(sorted_f) - np.array(s_filter_score)
#print(s_filter_list)
#print(p_filter_list.index([['Michael Kumbirai', 'occupation', 'rugby union player'], ['Michael Kumbirai', 'date of birth', '09 May 1996']]))
path = '../Textgen/output.jsonl'

filtered_records = []

f1_rho = pearsonr(sorted_f ,s_filter_score)[0]
precision_rho = pearsonr(sorted_p, s_filter_score)[0]
recall_rho = pearsonr(sorted_r, s_filter_score)[0]
print(f'precision {precision_rho}  rcall {recall_rho} f1 score {f1_rho}')
for i,e in enumerate(record):
   if diff[i] > 0.1 and f_score[i] > 0.68:
       filtered_records.append(e)
np.save('filtered_records.npy', filtered_records)
