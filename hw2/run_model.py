import argparse
import json
import os

import numpy as np
import tensorflow as tf


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Folder of the model to run")
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of model roll outs')

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    hyperparams_json = os.path.join(args.model, "hyperparams.json")
    with open(hyperparams_json, 'r') as json_file:
        hyperparams = json.load(json_file)

    import gym
    env = gym.make(hyperparams['envname'])
    max_steps = args.max_timesteps or env.spec.timestep_limit

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(args.agent, "model.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(args.agent))
        graph = tf.get_default_graph()
        tf_inputs_ob = graph.get_tensor_by_name("ob:0")
        tf_sampled_ac = graph.get_tensor_by_name("sampled_ac:0")

        returns = []

        for i in range(args.num_rollouts):
            print("********** Iteration %i ************" % i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = sess.run(tf_sampled_ac, feed_dict={tf_inputs_ob: obs[None, :]})
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                env.render()

                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
