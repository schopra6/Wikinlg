import pandas as pd
import jsonlines
from datasets import concatenate_datasets,load_dataset
import argparse
df=[]
triples = []
sentence = []
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset-path', type=str,
                        help='path to the noisy test dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the split dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()


    with jsonlines.open(args.dataset_path) as reader:
        for d in reader:
                triples.append(d['triples'])
                sentence.append(d['text'])
    df = pd.DataFrame([triples,sentence], index=['triples','sentence']).T
    df.to_csv(args.save_path+'/noisydata.csv')
    dataset = load_dataset("web_nlg",'release_v3.0_en')
    dataset_cc = concatenate_datasets([dataset['train'], dataset['dev'],dataset['test']])
    features = list(dataset_cc)
    web_triples = [list(map(lambda x:[y.strip() for y in  x.split('|')],e['original_triple_sets']['otriple_set'][0])) for e in features]
    df_webnlg = pd.DataFrame([web_triples], index=['triples']).T
    df_webnlg.to_csv(args.save_path+'webnlg.csv')