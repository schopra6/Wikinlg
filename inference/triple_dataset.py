import torch
import pandas as pd


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, with_extra=False):
        self.df = pd.read_csv(filename)
        self.with_extra = with_extra

    def __getitem__(self, idx):
        sentence, triples, title = self.df.iloc[idx]['sentence'], self.df.iloc[idx]['triples'], self.df.iloc[idx]['title']
        
        if not self.with_extra:
            return sentence, eval(triples), title,
        
        extra = {'alias': [title] + eval(self.df.iloc[idx]['alias']), 'description': self.df.iloc[idx]['description']}
        
        return sentence, eval(triples), title, extra

    def __len__(self):
        return len(self.df)


def text_gen_collator(features, tokenizer, lang_gen):
    # Parse Data
    sentences = [e['lex']['text'] for e in features]
    triples = [e['original_triple_sets']['otriple_set'][0] for e in features]
  
    
    def form_input(triples):
        res = "en_XX "
        for trip in triples:
            subj,rel,obj = trip.split('|')
            res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
        res += lang_gen
        return res
            
    
    formed_input = [form_input(trips) for trips in triples]

    inputs = tokenizer(formed_input, max_length=128, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')
    
    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Encode outputs
    labels = tokenizer(sentences, max_length=128, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt').input_ids
    labels_padding_mask = labels.eq(tokenizer.pad_token_id)
    labels[labels_padding_mask] = -100
    
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    return batch


def text_gen_with_extra_collator(features, tokenizer, lang_gen):
    # Parse Data
    sentences = [e[0] for e in features]
    triples = [e[1] for e in features]
    titles = [e[2] for e in features]
    
    extras = [e[3] for e in features]
    
    def form_input(triples, extra):
        res = "en_XX extra "
        for subj, rel, obj in triples:
            res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
        for alias in extra['alias']:
            res += f"<al> {alias} "
        res += f"<desc> {extra['description']} "
        res += lang_gen
        return res
            
    
    formed_input = [form_input(trips, extra) for trips, extra in zip(triples, extras)]

    inputs = tokenizer(formed_input, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')
    
    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Encode outputs
    labels = tokenizer(sentences, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt').input_ids
    labels_padding_mask = labels.eq(tokenizer.pad_token_id)
    labels[labels_padding_mask] = -100
    
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    return batch


def text_gen_to_extra_collator(features, tokenizer, lang_gen):
    # Parse Data
    triples = [e[1] for e in features]
    titles = [e[2] for e in features]
    
    extras = [e[3] for e in features]
    
    def form_input(triples):
        res = "Generate Description "
        for subj, rel, obj in triples:
            res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
        res += lang_gen
        return res
    
    def form_output(extra):
        res = ""
        for alias in extra['alias']:
            res += f"<al> {alias} "
        res += f"<desc> {extra['description']} "
        return res
            
    
    formed_input = [form_input(trips) for trips in triples]

    inputs = tokenizer(formed_input, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')
    
    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Encode outputs
    formed_output = [form_output(extra) for extra in extras]
    labels = tokenizer(formed_output, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt').input_ids
    labels_padding_mask = labels.eq(tokenizer.pad_token_id)
    labels[labels_padding_mask] = -100
    
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    return batch

def text_gen_collator_inference(features, tokenizer, lang_gen):
    # Parse Data
    sentences = [e['lex']['text'] for e in features]
    triples = [e['original_triple_sets']['otriple_set'][0] for e in features]
  
    
    def form_input(triples):
        res = "en_XX "
        for trip in triples:
            subj,rel,obj = trip.split('|')
            res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
        res += lang_gen
        return res
    
    formed_input = [form_input(trips) for trips in triples]
    print(formed_input[0])

    inputs = tokenizer(formed_input, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')
    
    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    return batch, sentences
