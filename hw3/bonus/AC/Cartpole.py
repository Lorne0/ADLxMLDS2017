import numpy as np
import tensorflow as tf
import random, sys
import gym
from AC import Actor, Critic

def set_config(gpu_fraction=0.005):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
    return config

def run_DQN():

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1229)
    
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    config = set_config()
    sess = tf.Session(config=config)
    actor = Actor(sess, n_features, n_actions, 0.001, 64)
    critic = Critic(sess, n_features, 0.01, 64)
    sess.run(tf.global_variables_initializer())

    episodes = 1000
    timestep = 0
    result = [] #result for each episode
    for e in range(episodes):
        print("%d/%d" %(e+1, episodes))
        obs = env.reset()
        episode_reward = 0
        t = 0
        while True:
            #if e > 400:
            #    env.render()

            a = actor.choose_action(obs)
            obs_, true_r, done, info = env.step(a)
            episode_reward += true_r

            x, x_dot, theta, theta_dot = obs_
            r1 = (env.x_threshold - abs(x) ) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            
            td_error = critic.learn(obs, r, obs_)
            actor.learn(obs, a, td_error)

            timestep += 1
            t += 1   
            obs = obs_

            if done or t>=40000:
                print("Reward: ", episode_reward)
                result.append(episode_reward)
                break

        rr = np.mean(result[-30:])
        print("Last 30 Reward: %f" %(rr))

    env.close()
    np.save('../result/Cartpole_AC_result.npy',result)
    
if __name__ == '__main__':
    run_DQN()
