import json
import jsonlines
import pandas as pd
import evaluate
import sacrebleu
import argparse
import numpy as np
from multiprocessing import Pool
import multiprocessing
from parent.format_textgen import format_text
from parent.compute_parent import compute_parent
from collections import defaultdict
triplesetgleu = defaultdict(list)
triplesetsimilarity = defaultdict(list)
triplesetbleu = defaultdict(list)
triplesetparent = defaultdict(list)
gleuStats = []
bleuStats = []
similarityStats = []
parentStats = []
averageStats =[]
google_bleu = evaluate.load("google_bleu")


def parse_args():
    parser = argparse.ArgumentParser(description='filter the dataset.')
    parser.add_argument('--input-path', type=str,
                        help='path to the dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the  dataset')
    parser.add_argument('--save-filename', type=str,
                        help='filename to save the  dataset')
    args = parser.parse_args()
    return args

def process_data(d):

            #filtering data having triple size less than 6

            if len(d['triples']) < 6:
                #ngrams of text present in the the graph
                gleu_score = google_bleu.compute(predictions=[d['text']],references = [[d['serialized_triples']]])
                bleu_score = sacrebleu.corpus_bleu([d['serialized_triples']],[[d['text']]]).score
                gleu_score= round(gleu_score["google_bleu"], 2)
                d['gleu'] = gleu_score
                d['bleu'] = bleu_score
                average= (d['bleu']/100) + d['gleu'] + d ['f1'] + d['similarity']
                d['average_score'] = (average/4)
                #triplesetgleu[len(d['triples'])].append(gleu_score)
                #triplesetbleu[len(d['triples'])].append(bleu_score)
                #triplesetsimilarity[len(d['triples'])].append(d['similarity'])
                return gleu_score,bleu_score,d
            else:
                return None
                #bleuStats.append(bleu_score)
                #json.dump(d, outfile)
                #outfile.write('\n')
                #outfile.flush()




if __name__ == '__main__':

    args = parse_args()
    #preprocessing for computing parent score
    format_text(args.input_path,args.save_path)
    #computing parent score and adding to the dataset (f1 score is considered as parent score
    compute_parent(args.input_path,args.save_path,args.save_filename)
    with jsonlines.open(args.save_path+args.save_filename) as reader:

      with open(args.save_path+'parent' + args.save_filename, 'w') as outfile:
        # final output file with all the filter score

        pool = Pool(multiprocessing.cpu_count())                         # Create a multiprocessing Pool
        for result in pool.imap(process_data, reader):
            if result:# if result is not null , less than 6 tripleset
                 # result = (gleu_score,bleu_score,jsonline)
                 gleuStats.append(result[0])
                 bleuStats.append(result[1])
                 similarityStats.append(result[2]['similarity'])
                 parentStats.append(result[2]['f1'])
                 averageStats.append(result[2]['average_score'])
                 triplesetgleu[len(result[2]['triples'])].append(result[0])
                 triplesetbleu[len(result[2]['triples'])].append(result[1])
                 triplesetsimilarity[len(result[2]['triples'])].append(result[2]['similarity'])
                 triplesetparent[len(result[2]['triples'])].append(result[2]['f1'])
                 triplesetparent[len(result[2]['triples'])].append(result[2]['average'])
                 json.dump(result[2], outfile)
                 outfile.write('\n')




print(pd.DataFrame(bleuStats).describe())
print(pd.DataFrame(gleuStats).describe())
print(pd.DataFrame(similarityStats).describe())
print(pd.DataFrame(parentStats).describe())
print(pd.DataFrame(averageStats).describe())

for k in range(1,6):
 print(f'bleu for size {k} : {np.mean(triplesetbleu[k])}')
 print(f'similarity for size {k} : {np.mean(triplesetsimilarity[k])}')
 print(f'gleu for size {k} : {np.mean(triplesetgleu[k])}')
 print(f'parent for size {k} : {np.mean(triplesetparent[k])}')
