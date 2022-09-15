
from  Parent import parent
import numpy as np
import json, os
import jsonlines


def compute_parent(inputpath,outputpath,filename):
        output_filename = f'{outputpath}/train_output.txt'
        table_filename = f'{outputpath}/train_tables.jl'

        with open(table_filename, mode="r", encoding='utf8') as f:
            tables = [json.loads(line) for line in f if line.strip()]

        with open(output_filename, mode="r", encoding='utf8') as f:
            references = [line.strip().split() for line in f if line.strip()]
        #print(references)

        print(len(tables) == len(references))
        #path = f'../corpus/train.tsv'
        with open(inputpath) as reader:
            dataset = [json.loads(d.replace("\'", "\"")) for d in reader]

        parent(dataset, references, tables,outputpath,filename)


