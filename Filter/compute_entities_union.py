import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Split the noisy test dataset.')
    parser.add_argument('--input-path', type=str,
                        help='path to the dataset')
    parser.add_argument('--save-path', type=str,
                        help='path to save the  dataset')
    parser.add_argument('--save-filename', type=str,
                        help='filename to save the  dataset')
    args = parser.parse_args()
    return args
