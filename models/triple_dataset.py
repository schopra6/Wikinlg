import torch
import pandas as pd
import jsonlines


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, filename,similarity_score=0, with_extra=False):
        self.df=[]
        triples = []
        sentence = []
        #with jsonlines.open(filename) as reader:
        #    for d in reader:

        #           triples.append(d['triples'])
        #           sentence.append(d['text'])
        #self.df = pd.DataFrame([triples,sentence], index=['triples','sentence']).T
        self.df = pd.read_csv(filename)
        self.df.reset_index(drop=True, inplace=True)
    def __getitem__(self, idx):
        sentence, triples = self.df.iloc[idx]['sentence'], self.df.iloc[idx]['triples']


        return sentence, triples
    def __len__(self):
        return len(self.df)





def text_gen_collator(features, tokenizer, lang_gen):
    # Parse Data
    sentences = [e[0] for e in features]
    triples = [e[1] for e in features]
    #titles = [e[2] for e in features]
    #print(triples[0])

    def form_input(triple):
        res = "en_XX "
        #print(triple)
        for trip in triple:
            #print(trip)
            try:
                subj,rel,obj =trip[0],trip[1],trip[2]
                res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
            except:
                print(trip)
                #pass
        res += lang_gen
        return res


    formed_input = [form_input(eval(trip)) for trip in triples]

    inputs = tokenizer(formed_input, max_length=256, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')

    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Encode outputs
    labels = tokenizer(sentences, max_length=256, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt').input_ids
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
    sentences = [sentence for sentence, _, _ in features]
    triples = [triples for _, triples, _ in features]
    titles = [title for _, _, title in features]

    def form_input(triples):
        res = "en_XX "
        for subj, rel, obj in triples:
            res += f"<trip> <subj> {subj} <rel> {rel} <obj> {obj} "
        res += lang_gen
        return res


    formed_input = [form_input(trips) for trips in triples]

    inputs = tokenizer(formed_input, max_length=512, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt')

    # Encode inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask



    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    return batch, sentences
