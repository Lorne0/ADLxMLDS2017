import numpy as np
import tensorflow as tf
import random, sys
import gym
from PG import PolicyGradient

def set_config(gpu_fraction=0.05):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options = gpu_options)
    return config

def run_PG():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1229)
    sess = tf.Session(config=set_config())
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    PG = PolicyGradient(sess, n_features, n_actions, lr=0.02, gamma=0.99, num_units=10)

    result = [] #result for each episode
    episodes = 1000
    for e in range(episodes):
        print("%d/%d" %(e+1, episodes))
        obs = env.reset()
        episode_reward = 0
        ss, aa, rr = [], [], [] 
        t = 0
        while True:
            #if e % 100 == 0:
            #    env.render()
            
            a = PG.choose_action(obs)
            obs_, true_r, done, info = env.step(a)
            episode_reward += true_r

            x, x_dot, theta, theta_dot = obs_
            r1 = (env.x_threshold - abs(x) ) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            t += 1

            ss.append(obs)
            aa.append(a)
            rr.append(r)

            if done or t >= 40000:
                vt = PG.learn(ss, aa, rr)
                result.append(episode_reward)
                print("Reward:", episode_reward)
                break

            obs = obs_

        rr = np.mean(result[-30:])
        print("Last 30 Reward: %f" %(rr))

    env.close()
    np.save('../result/Cartpole_PG_result.npy',result)

if __name__ == '__main__':
    run_PG()
