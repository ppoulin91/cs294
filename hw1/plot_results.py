import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def handle_experiment(exp_folder):
    hyperparams_json = os.path.join(exp_folder, "hyperparams.json")
    with open(hyperparams_json, 'r') as json_file:
        hyperparams = json.load(json_file)
        dataset_name = os.path.basename(hyperparams['dataset'])

    with open(os.path.join(exp_folder, "losses.pkl"), 'rb') as losses_file:
        losses = pickle.load(losses_file)

    return dataset_name, losses

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', type=str, nargs='+', help="Experiment folder to fetch results from")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    experiments = {}
    for experiment_folder in args.experiments:

        if not os.path.isdir(experiment_folder):
            raise FileNotFoundError("Experiment folder not found: {}".format(experiment_folder))

        if os.path.exists(os.path.join(experiment_folder, 'losses.pkl')):
            dataset_name, losses = handle_experiment(experiment_folder)
            exp_name = os.path.basename(experiment_folder)

            if dataset_name not in experiments:
                experiments[dataset_name] = {}
            experiments[dataset_name][exp_name] = losses

        elif os.path.exists(os.path.join(experiment_folder, 'experiments')):
            experiment_folder = os.path.join(experiment_folder, 'experiments')

            for d in os.listdir(experiment_folder):
                exp_folder = os.path.join(experiment_folder, d)

                try:
                    dataset_name, losses = handle_experiment(exp_folder)
                    exp_name = d

                    if dataset_name not in experiments:
                        experiments[dataset_name] = {}
                    experiments[dataset_name][exp_name] = losses
                except FileNotFoundError:
                    continue

    fig = plt.figure()
    fig.suptitle("Behavioral Cloning")
    n_rows = min([4, len(experiments)])
    n_cols = max([1, len(experiments) // n_rows])
    for i, (dataset_name, dataset_experiments) in enumerate(experiments.items(), 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.set_title("Dataset: {}".format(dataset_name))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')

        for exp_name, losses in dataset_experiments.items():
            ax.plot(np.arange(len(losses)), losses, label="{}".format(exp_name))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
