import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from functools import partial
import pandas as pd
from sklearn.model_selection import train_test_split

from triple_dataset import TripleDataset, text_gen_collator_inference

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="t5")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--language", type=str, default="breton")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out-pred-file", type=str, default=None, required=True)
    parser.add_argument("--out-gold-file", type=str, default=None, required=True)

    return parser.parse_args()

       
if __name__ == '__main__':
    
    args = parse_args()
    
    #dataset_val = TripleDataset(args.data_path)
    dataset_val = load_dataset("web_nlg",'release_v3.0_en')['test']
    lang_gen ='en_XX'
    #if None:
    #    _, dataset_val = train_test_split(dataset, test_size=.2, random_state=42)
    
    if args.model == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
    elif args.model == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.checkpoint_dir + args.checkpoint_name)
    
    tokens_list = ['en_XX', '<subj>', '<obj>', '<rel>', '<trip>']
    
    if args.language == 'breton':
        #tokens_list.append('br_XX')
        lang_gen = 'br_XX'
    
    
    tokenizer.add_special_tokens({'additional_special_tokens': tokens_list})
    
    if args.model == 't5':
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_dir + args.checkpoint_name).to(args.device).eval()
    elif args.model == 'bart':
        model = BartForConditionalGeneration.from_pretrained(args.checkpoint_dir + args.checkpoint_name).to(args.device).eval()

    loader = DataLoader(dataset_val, batch_size=args.batch_size,
                        collate_fn=partial(text_gen_collator_inference, tokenizer=tokenizer, lang_gen=lang_gen))

    # predict output sequences
    pred_ys = []
    ideal_ys = []
    for batch, sentences in tqdm(loader):
        batch = { k: v.to(args.device, non_blocking=True) for k, v in batch.items() }
        with torch.no_grad():
            generated_tokens = model.generate(**batch, max_length=30)
            preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            pred_ys += preds
            ideal_ys += sentences
    #    break
    #print(pred_ys)

    # write outputs to file
    with open(args.out_pred_file, "w") as f:
        for pred in pred_ys:
            f.write(f"{pred}\n")
        print(f"Generated sentences written to {args.out_pred_file}")
        
    #with open(args.out_gold_file, "w") as f:
    #    for sent in ideal_ys:
    #        f.write(f"{sent}\n")
    np.save(str(args.out_gold_file),ideal_ys)
    print(f"Ideal sentences written to {args.out_gold_file}")
        
        
