import jsonlines
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--input-path', type=str,
                        help='path to the dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the  dataset')
    parser.add_argument('--average-score', type=float,
                        help='path to save the  dataset')

    args = parser.parse_args()
    return args





if __name__ == '__main__':

    args = parse_args()
    outfile1= open(args.save_path + '/filtereddata.jsonl', 'w')
    with jsonlines.open(args.input_path) as reader:
        for d in reader:
            if d['average_score'] >= args.average_score:
                json.dump(d, outfile1)
                outfile1.write('\n')
    outfile1.close()

