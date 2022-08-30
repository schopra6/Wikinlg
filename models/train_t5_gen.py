import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from functools import partial

import argparse

from triple_dataset import TripleDataset, text_gen_collator


def parse_args():
    parser = argparse.ArgumentParser(description='Run training for generating sentences')
    parser.add_argument('--data-path', type=str,
                        help='path to the transformers-like triple dataset')
    parser.add_argument('--language', type=str,
                        default="breton",
                        help='language to generate in')
    
    parser.add_argument("--model-name", default='t5-gen-br')
    parser.add_argument("--save-dir", type=str, default='../models/')
    parser.add_argument("--log-dir", type=str, default='../log/')
                        
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--similarity_score", type=int, default=0)
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    
    args = parse_args()
    
    dataset = TripleDataset(args.data_path,args.similarity_score)
    dataset_train, dataset_val = train_test_split(dataset, test_size=.2, random_state=42)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    tokens_list = ['en_XX', '<subj>', '<obj>', '<rel>', '<trip>']
    
    if args.language == 'breton':
        tokens_list.append('br_XX')
        lang_gen = 'br_XX'
    lang_gen = 'en_XX' 
    
    tokenizer.add_special_tokens({'additional_special_tokens': tokens_list})
    
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    
    
    train_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.save_dir}{args.model_name}",
        do_train=True,
        do_eval=True,
        evaluation_strategy ='steps',
        eval_steps = 1000,
        save_steps = 1000,
        logging_steps = 1000,
        save_total_limit = 1,
        load_best_model_at_end=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        adam_beta2=0.999,
        label_smoothing_factor=0,
        num_train_epochs=args.epochs,
        logging_dir=f"{args.log_dir}{args.model_name}",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=partial(text_gen_collator, tokenizer=tokenizer, lang_gen=lang_gen),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )
                        
    # Starting Evaluation
    trainer.evaluate()

    # Run Training
    trainer.train()

