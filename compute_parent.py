
from  Parent import parent
import numpy as np
import json, os
import jsonlines

with open('Textgen/train_tables.jl', mode="r", encoding='utf8') as f:
    tables = [json.loads(line) for line in f if line.strip()]

with open('Textgen/train_output.txt', mode="r", encoding='utf8') as f:
    references = [line.strip().split() for line in f if line.strip()]
print(len(tables) == len(references))
path = f'corpus/train.tsv'
with jsonlines.open(path) as reader:
    dataset = [d for d in reader]






p, r, f = parent(dataset, references, tables)


