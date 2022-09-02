import json, os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import evaluate
import numpy as np
import sacrebleu
import argparse
from collections import defaultdict
triplesetgleu = defaultdict(list)
triplesetSimilarity = defaultdict(list)
gleuStats=[]
google_bleu = evaluate.load("google_bleu")
gleu =[0,0,0,0,0]
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--input-path', type=str,
                        help='path to the dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the  dataset')
    args = parser.parse_args()
    return args


def process_data():



if __name__ == '__main__':

    args = parse_args()
    outfile1= open(args.save_path + '/gleubin1.jsonl', 'w')
    outfile2= open(args.save_path + '/gleubin2.jsonl', 'w')
    outfile3= open(args.save_path + '/gleubin3.jsonl', 'w')
    outfile4= open(args.save_path + '/gleubin4.jsonl', 'w')
    outfile5= open(args.save_path + '/gleubin5.jsonl', 'w')
    pool = Pool(8)
    with jsonlines.open(args.input_path) as reader:
            for d in reader:
                record ={}
                record['serialized_triples'] = d['serialized_triples']
                record['text'] = d['text']
                record['gleu'] = d['gleu']
                record['bleu'] = d['bleu']
                record['similarity'] = d['similarity']

                #filtering data having triple size less than 6
                if d['gleu'] <= 0.2:
                    gleu[0] += 1
                    json.dump(record, outfile1)
                    outfile1.write('\n')
                elif d['gleu'] > 0.2 and d['gleu'] <= 0.4:
                    gleu[1] += 1
                    json.dump(record, outfile2)
                    outfile2.write('\n')
                elif d['gleu'] > 0.4 and d['gleu'] <= 0.6:
                    gleu[2] += 1
                    json.dump(record, outfile3)
                    outfile3.write('\n')
                elif d['gleu'] > 0.6 and d['gleu'] <= 0.8:
                    gleu[3] += 1
                    json.dump(record, outfile4)
                    outfile4.write('\n')
                else :
                    gleu[4] += 1
                    json.dump(record, outfile5)
                    outfile5.write('\n')
    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()
    outfile5.close()
    for i in range(0,5):
        print(f'{i} : {gleu[i]}')


