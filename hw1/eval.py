#!/usr/bin/env python

"""
Code to load a trained policy model and evaluate returns
Example usage:
    python eval.py experiments/model1 Humanoid-v1 --render --num_rollouts 20
"""
import json
import os
import pickle

import numpy as np
import tensorflow as tf

import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, help="Experiment folder if trained agent, otherwise pkl file if expert")
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--output", type=str, default="output.pkl", help="Output file to save the roll-out data")
    args = parser.parse_args()

    with tf.Session() as sess:
        is_expert = False
        if args.agent.endswith(".pkl"):
            is_expert = True

        print('loading and building agent')
        if is_expert:
            policy_fn = load_policy.load_policy(args.agent)
        else:
            hyperparams_json = os.path.join(args.agent, "hyperparams.json")
            with open(hyperparams_json, 'r') as json_file:
                hyperparams = json.load(json_file)

            saver = tf.train.import_meta_graph(os.path.join(args.agent, "model.ckpt.meta"))
            saver.restore(sess, tf.train.latest_checkpoint(args.agent))
            graph = tf.get_default_graph()
            tf_model_prediction = graph.get_tensor_by_name("output:0".format(len(hyperparams['hidden_sizes'])))
            tf_inputs = graph.get_tensor_by_name("inputs:0")
        print('loaded and built')

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                if is_expert:
                    action = policy_fn(obs[None,:])
                else:
                    action = np.squeeze(sess.run(tf_model_prediction, feed_dict={tf_inputs: obs[None, :]}))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': returns,
                       'environment': args.envname}

        pickle.dump(expert_data, open(args.output, 'wb'))


if __name__ == '__main__':
    main()
