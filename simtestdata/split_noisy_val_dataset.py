import argparse
import pandas as pd
import itertools
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--noisy-dataset-path', type=str,
                        help='path to the noisy test dataset')
    parser.add_argument('--webnlg-dataset-path', type=str,
                        help='path to webnlg train dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the split dataset')
    parser.add_argument('--language-code', type=str,
                        help='language code')
    args = parser.parse_args()
    return args
                       

    
TRIPLE_DISTRIBUTION = [400, 400, 300, 200, 200]

if __name__ == '__main__':
    args = parse_args()
    
    wb_df = pd.read_csv(args.webnlg_dataset_path)
    
    wb_entities = list(map(lambda x: list(map(lambda y: [y[0], y[2]], eval(x))), wb_df['triples']))
    wb_entities = list(itertools.chain(*wb_entities))
    wb_entities = list(itertools.chain(*wb_entities))
    
    wb_entities = set(wb_entities)
    
    wb_props = list(map(lambda x: list(map(lambda y: y[1], eval(x))), wb_df['triples']))
    wb_props = list(itertools.chain(*wb_props))
    wb_props = set(wb_props)
    
    wb_triples = list(map(lambda x: eval(x), wb_df['triples']))
    wb_triples = list(itertools.chain(*wb_triples))
    #wb_triples = list(itertools.chain(*wb_triples))
    #wb_triples = set(wb_triples)
    
    
    my_df = pd.read_csv(args.noisy_dataset_path)
    my_df['unknown_properties'] = ['No'] * len(my_df)
    my_df['unknown_entities'] = ['No'] * len(my_df)
    my_df['unknown_triples'] = ['No'] * len(my_df)
    
    for i in range(len(my_df)):
        triples = eval(my_df.iloc[i]['triples'])
        known_entities = False
        known_properties = False
        known_triples = False
        for trip in triples:
            if trip[1] in wb_props:
                known_properties = True
            if trip[0] in wb_entities or trip[2] in wb_entities:
                known_entities = True
            if trip in wb_triples:
                known_triples = True
        if not known_entities:
            my_df.iloc[i, 3] = "Yes"
        if not known_properties:
            my_df.iloc[i, 2] = "Yes"
        if not known_triples:
            my_df.iloc[i, 4] = "Yes"
            
    my_df = my_df[(my_df['sentence'].str.len() > 50) & (my_df['sentence'].str.len() < 250)]
    my_df = my_df[np.array(list(map(lambda x: len(eval(x)), my_df['triples'].values))) <= 5]
    
    my_df_props = my_df[my_df['unknown_properties'] == 'Yes']
    num_triples_props = np.array(list(map(lambda x: len(eval(x)), my_df_props['triples'].values)))

    mask_props_by_triples_num = [num_triples_props == i for i in range(1, 6)]
    
    mask_props = np.zeros(len(my_df_props), dtype=bool)
    
    for i in range(5):
        mask_props |= (np.random.random_sample(len(my_df_props)) * len(my_df_props[mask_props_by_triples_num[i]]) < TRIPLE_DISTRIBUTION[i]) & mask_props_by_triples_num[i]
        
    my_df_props[mask_props].to_csv(f"{args.save_path}/no_props_{args.language_code}.csv", index=False)
    my_df.drop(my_df_props[mask_props].index).to_csv(f"{args.save_path}/train_no_props_{args.language_code}.csv", index=False)

    #x = my_df.drop(mask_props, axis=0,inplace=True)
    #x.to_csv(f"{args.save_path}/train_no_props_{args.language_code}.csv", index=False)
    
    
    my_df_ents = my_df[my_df['unknown_entities'] == 'Yes']
    num_triples_ents = np.array(list(map(lambda x: len(eval(x)), my_df_ents['triples'].values)))

    mask_ents_by_triples_num = [num_triples_ents == i for i in range(1, 6)]
    
    mask_ents = np.zeros(len(my_df_ents), dtype=bool)
    
    for i in range(5):
        mask_ents |= (np.random.random_sample(len(my_df_ents)) * len(my_df_ents[mask_ents_by_triples_num[i]]) < TRIPLE_DISTRIBUTION[i]) & mask_ents_by_triples_num[i]
        
    my_df_ents[mask_ents].to_csv(f"{args.save_path}/no_ents_{args.language_code}.csv", index=False)
    my_df.drop(my_df_ents[mask_ents].index).to_csv(f"{args.save_path}/train_no_ents_{args.language_code}.csv", index=False)
    #my_df_trips[np.logical_not(mask_trips)].to_csv(f"{args.save_path}/train_no_trips_{args.language_code}.csv", index=False)

    
    
    my_df_trips = my_df[my_df['unknown_triples'] == 'Yes']
    num_triples_trips = np.array(list(map(lambda x: len(eval(x)), my_df_trips['triples'].values)))

    mask_trips_by_triples_num = [num_triples_trips == i for i in range(1, 6)]
    
    mask_trips = np.zeros(len(my_df_trips), dtype=bool)
    
    for i in range(5):
        mask_trips |= (np.random.random_sample(len(my_df_trips)) * len(my_df_trips[mask_trips_by_triples_num[i]]) < TRIPLE_DISTRIBUTION[i]) & mask_trips_by_triples_num[i]
        
    my_df_trips[mask_trips].to_csv(f"{args.save_path}/no_trips_{args.language_code}.csv", index=False)
    my_df.drop(my_df_trips[mask_trips].index).to_csv(f"{args.save_path}/train_no_trips_{args.language_code}.csv", index=False)


