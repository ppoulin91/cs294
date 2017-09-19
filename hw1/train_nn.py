#!/usr/bin/env python
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import load_policy
from network_utils import ACTIVATION_FUNCTIONS, OPTIMIZERS, build_model


def build_behavior_cloning_subparser(subparser):
    DESCRIPTION = "Train model using behavior cloning"
    p = subparser.add_parser("behavior_cloning", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument('dataset', type=str)


def build_dagger_subparser(subparser):
    DESCRIPTION = "Train model using DAGGER"
    p = subparser.add_parser("dagger", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument('dataset', type=str)
    p.add_argument('expert', type=str, help="Reference expert if using DAGGER (.pkl)")
    p.add_argument('envname', type=str)
    p.add_argument('--num-rollouts', type=int, default=20)
    p.add_argument('--max-timesteps', type=int)
    p.add_argument('--n-loops', type=int, default=5)


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=64)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0.01, help="Regularization strength on weights")
    parser.add_argument('--seed', type=float, default=1234)
    parser.add_argument('-a', '--activation', type=str, choices=ACTIVATION_FUNCTIONS, help="Activation function to use. [{}]".format(", ".join(
        ACTIVATION_FUNCTIONS.keys())))
    parser.add_argument('-o', '--optimizer', type=str, choices=OPTIMIZERS, help="Optimizer to use. [{}]".format(", ".join(OPTIMIZERS.keys())))
    parser.add_argument('--name', type=str)
    parser.add_argument('-v', '--view', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')

    subparser = parser.add_subparsers(title="Training method", dest="method")
    subparser.required = True
    build_behavior_cloning_subparser(subparser)
    build_dagger_subparser(subparser)

    return parser


def run_dagger(sess, env, expert_fn, inputs, outputs, num_rollouts, max_steps):
    model_observations = []
    expert_actions = []
    for n in range(num_rollouts):
        print('iter', n)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            model_action = np.squeeze(sess.run(outputs, feed_dict={inputs: obs[None, :]}))
            expert_action = expert_fn(obs[None, :])
            model_observations.append(obs)
            expert_actions.append(expert_action)
            obs, _, done, _ = env.step(model_action)
            steps += 1
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
    return model_observations, np.squeeze(expert_actions)


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print(args)
    print("Using tensorflow v.{}".format(tf.VERSION))

    hyperparams = OrderedDict(sorted(vars(args).items()))
    del hyperparams['view']
    del hyperparams['force']
    del hyperparams['name']

    experiment_name = args.name
    if experiment_name is None:
        experiment_name = hashlib.sha256(repr(hyperparams).encode()).hexdigest()
    experiment_path = os.path.join(".", "experiments", experiment_name)
    if os.path.isdir(experiment_path):
        if not args.force:
            print("Experiment directory already exists. Use --force to overwrite. {}".format(experiment_path))
            return
        else:
            shutil.rmtree(experiment_path)
    os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, "hyperparams.json"), 'w') as json_file:
        json.dump(hyperparams, json_file)

    tf.set_random_seed(hyperparams['seed'])
    np.random.seed(hyperparams['seed'])

    expert_data = pickle.load(open(hyperparams['dataset'], 'rb'))
    observations = expert_data['observations']
    actions = np.squeeze(expert_data['actions'])

    print("Original dataset size : {}".format(observations.shape[0]))

    input_size = observations.shape[1]
    output_size = actions.shape[1]
    print("Input size: {}; Output size: {}".format(input_size, output_size))

    expert_fn = None
    n_loops = 1
    if hyperparams['method'] == "dagger":
        expert_fn = load_policy.load_policy(hyperparams["expert"])
        import gym
        env = gym.make(args.envname)
        max_steps = hyperparams['max_timesteps'] or env.spec.timestep_limit
        num_rollouts = hyperparams['num_rollouts']
        n_loops = hyperparams['n_loops']

    inputs = tf.placeholder(tf.float32, [None, input_size], name="inputs")
    targets = tf.placeholder(tf.float32, [None, output_size], name="targets")

    # Build FFNN
    outputs = build_model(inputs, input_size, args.hidden_sizes, output_size, hyperparams["activation"], reg=hyperparams['reg'])
    outputs = tf.identity(outputs, name="output")

    # MSE
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(outputs - targets), axis=1), name="loss")
    # Regularization
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.reduce_sum(reg_variables)
    # Total loss
    loss = mse + reg_loss

    optimizer = OPTIMIZERS[hyperparams['optimizer']]
    train_step = optimizer(args.lr).minimize(loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()

        tf.global_variables_initializer().run()

        losses = []

        for i in range(n_loops):
            for epoch in range(hyperparams['max_epochs']):
                n_data = observations.shape[0]
                epoch_ids = np.arange(n_data)
                np.random.shuffle(epoch_ids)
                batches_per_epoch = int(math.ceil(n_data / min((hyperparams['batch_size'], n_data))))

                epoch_losses = []
                for batch_idx in range(batches_per_epoch):
                    batch_ids = epoch_ids[batch_idx*hyperparams['batch_size']:(batch_idx+1)*hyperparams['batch_size']]
                    train_loss, _ = sess.run([loss, train_step], feed_dict={inputs: observations[batch_ids], targets: actions[batch_ids]})
                    epoch_losses.append(train_loss)

                train_loss = np.mean(epoch_losses)
                losses.append(train_loss)
                print("Epoch: {} - train loss: {}".format(epoch, train_loss))
                saver.save(sess, os.path.join(experiment_path, "model.ckpt"))

            if hyperparams['method'] == "dagger":
                print("Iteration {} finished. Collecting more data using DAGGER.".format(i))
                model_observations, expert_actions = run_dagger(sess, env, expert_fn, inputs, outputs, num_rollouts, max_steps)
                observations = np.concatenate((observations, model_observations))
                actions = np.concatenate((actions, expert_actions))

        print("Training finished!")

        with open(os.path.join(experiment_path, "losses.pkl"), 'wb') as losses_file:
            pickle.dump(losses, losses_file)

    if args.view:
        fig = plt.figure()
        method_name = hyperparams['method']
        dataset_name = os.path.split(hyperparams['dataset'])[1]
        fig.suptitle("{} using dataset: {}".format(method_name, dataset_name))

        ax = fig.add_subplot('111')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')

        ax.plot(np.arange(len(losses)), losses, label="train loss")

        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
