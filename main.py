import argparse

from utils import load_data
from fundamental_matrix import eight_point, seven_point, ransac_plots
from triangulation import triangulate, colmap_reconstruct

QUESTIONS = {
    'q1a': eight_point,
    'q1b': seven_point,
    'q2': ransac_plots,
    'q3': triangulate,
    'q4': colmap_reconstruct
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1a', 'q1b', 'q2', 'q3', 'q4'], required=True)
    parser.add_argument('-i', '--image', choices=['teddy', 'chair','toybus', 'toytrain'])
    parser.add_argument('-r', '--ransac', action='store_true')
    parser.add_argument('-v', '--viz', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    data = load_data(args)
    QUESTIONS[args.question](**data)

if __name__ == '__main__':
    args = parse_args()
    main(args)