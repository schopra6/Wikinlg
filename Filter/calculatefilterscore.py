import json
import jsonlines
import pandas as pd
import evaluate
import sacrebleu
import argparse
import numpy as np
from multiprocessing import Pool
import multiprocessing
from collections import defaultdict
triplesetgleu = defaultdict(list)
triplesetsimilarity = defaultdict(list)
triplesetbleu = defaultdict(list)
gleuStats = []
bleuStats = []
similarityStats = []
google_bleu = evaluate.load("google_bleu")


def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--input-path', type=str,
                        help='path to the dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the  dataset')
    args = parser.parse_args()
    return args

def process_data(d):
                                                                                                           
            #filtering data having triple size less than 6 
                                                                            
            if len(d['triples']) < 6:                                                                                       
                gleu_score = google_bleu.compute(predictions=[d['serialized_triples']],references = [[d['text']]])          
                bleu_score = sacrebleu.corpus_bleu([d['serialized_triples']],[[d['text']]]).score                           
                gleu_score= round(gleu_score["google_bleu"], 2)                                                             
                d['gleu'] = gleu_score                                                                                      
                d['bleu'] = bleu_score                                                                                      
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
    with jsonlines.open(args.input_path) as reader:

      with open(args.save_path, 'w') as outfile: 
        pool = Pool(multiprocessing.cpu_count())                         # Create a multiprocessing Pool
        for result in pool.imap(process_data, reader):
            if result:
             gleuStats.append(result[0])
             bleuStats.append(result[1])
             similarityStats.append(result[2]['similarity'])
             triplesetgleu[len(result[2]['triples'])].append(result[0])
             triplesetbleu[len(result[2]['triples'])].append(result[1])
             triplesetsimilarity[len(result[2]['triples'])].append(result[2]['similarity'])
             json.dump(result[2], outfile)
             outfile.write('\n')                                                                                       




print(pd.DataFrame(bleuStats).describe())
print(pd.DataFrame(gleuStats).describe())
print(pd.DataFrame(similarityStats).describe())

for k in range(1,6):
 #print(f'bleu before {k} : {bleubefore[k]}')
 #print(percentage(bleuafter[i],bleubefore[i]),end =' & ')
  #print(f'bleu after {i} : {bleuafter[i]}')
 print(f'bleu for size {k} : {np.mean(triplesetbleu[k])}')
 print(f'similarity for size {k} : {np.mean(triplesetsimilarity[k])}')
 print(f'gleu for size {k} : {np.mean(triplesetsimilarity[k])}')
