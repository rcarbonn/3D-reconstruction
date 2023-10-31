import argparse

from utils import load_data
from fundamental_matrix import eight_point, seven_point
from triangulation import triangulate, colmap_reconstruct

QUESTIONS = {
    'q1a': eight_point,
    'q1b': seven_point,
    'q2a': eight_point,
    'q2b': seven_point,
    'q3': triangulate,
    'q5': eight_point,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1a', 'q1b', 'q2a', 'q2b', 'q3', 'q5'], required=True)
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