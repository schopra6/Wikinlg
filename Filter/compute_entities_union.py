import argparse
import pandas as pd
import itertools
import numpy as np

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))
def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--noisy-dataset-path', type=str,
                        help='path to the noisy test dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the split dataset')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    wb_df = pd.read_csv(args.noisy_dataset_path)

    wb_entities = list(map(lambda x: list(map(lambda y: [y[0], y[2]], eval(x))), wb_df['triples']))
    wb_entities = list(itertools.chain(*wb_entities))
    wb_entities = list(itertools.chain(*wb_entities))

    wb_entities = set(wb_entities)

    wb_props = list(map(lambda x: list(map(lambda y: y[1], eval(x))), wb_df['triples']))
    wb_props = list(itertools.chain(*wb_props))
    wb_props = set(wb_props)

    wb_triples = list(map(lambda x: eval(x), wb_df['triples']))
    wb_triples = list(itertools.chain(*wb_triples))
    wb_triples = sort_and_deduplicate(wb_triples)
    #wb_triples = list(itertools.chain(*wb_triples))
    #wb_triples = set(wb_triples)
    wb_graph = list(map(lambda x: eval(x), wb_df['triples']))
    print(wb_graph[0])
    wb_graph = sort_and_deduplicate(wb_graph)
    print(f'properties {len(wb_props)} entities {len(wb_entities)} triples {len(wb_triples)} graph {len(wb_graph)}')
    with open(args.save_path +'/noisy_properties.txt', 'w') as f:
        for line in wb_props:
            f.write(f"{line}\n")
    with open(args.save_path +'/noisy_entities.txt', 'w') as f:
        for line in wb_entities:
            f.write(f"{line}\n")
    with open(args.save_path +'/noisy_triples.txt', 'w') as f:
        for line in wb_triples:
            f.write(f"{str(line)}\n")
    with open(args.save_path +'/noisy_graph.txt', 'w') as f:
        for line in wb_graph:
            f.write(f"{str(line)}\n")