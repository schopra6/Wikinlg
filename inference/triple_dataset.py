import torch
import pandas as pd


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, with_extra=False):
        self.df = pd.read_csv(filename)
        #self.with_extra = with_extra

    def __getitem__(self, idx):
        sentence, triples = self.df.iloc[idx]['sentence'], self.df.iloc[idx]['triples']
        
        return sentence, eval(triples)

    def __len__(self):
        return len(self.df)


def text_gen_collator_inference(features, tokenizer, lang_gen):
    # Parse Data
    sentences = [e[0] for e in features]
    triples = [e[1] for e in features]
  
    
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
