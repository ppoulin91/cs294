import argparse
import csv
import os
import pickle

import numpy as np


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_folder', type=str, help="Evaluation folder to fetch results from")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.isdir(args.eval_folder):
        raise FileNotFoundError("Experiment folder not found: {}".format(args.eval_folder))

    experiments = []

    for eval_file in os.listdir(args.eval_folder):
        exp_name = eval_file.rstrip('.pkl')
        data = None
        with open(os.path.join(args.eval_folder, eval_file), 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        returns = data['returns']
        returns_mean = np.mean(returns)
        returns_std = np.std(returns)

        to_save = {}
        to_save['name'] = exp_name
        to_save['returns_mean'] = returns_mean
        to_save['returns_std'] = returns_std
        experiments.append(to_save)

    with open("results.csv", 'w', newline='') as csv_file:
        dict_keys = ["name", "returns_mean", "returns_std"]
        dictwriter = csv.DictWriter(csv_file, dict_keys)
        dictwriter.writeheader()
        dictwriter.writerows(experiments)


if __name__ == '__main__':
    main()
