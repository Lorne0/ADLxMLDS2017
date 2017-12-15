import numpy as np
import scipy.misc
# https://github.com/keras-team/keras/issues/4613, make the program use only CPUs
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import *
from keras.layers import *
from keras.initializers import *
import gym
import multiprocessing as mp

#tf.reset_default_graph()
####### hyper-parameter #######
INT = 2**32 - 1
N_KID = 100
N_GENERATION = 1000
#N_CORE = mp.cpu_count()-1
N_CORE = 100
LR = 1e-4
SIGMA = 0.01
env = gym.make('Pong-v0')
n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

def preprocess(o):
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, [80,80])
    return np.expand_dims(resized.astype(np.float32),axis=2)

def sign(k_id): return -1.0 if k_id % 2 == 0 else 1.0  # mirrored sampling

def build_model():
    model = Sequential()
    #model.add(Reshape((80, 80, 1), input_shape=(n_features,)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=(80,80,1), kernel_initializer='he_uniform'))
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_actions, kernel_initializer='he_uniform'))
    return model

def make_action(model, x):
    actions = model.predict(np.expand_dims(x, axis=0))[0]
    return np.argmax(actions)

def play_game(W, env, seed_and_id=None):
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        for i in range(len(W)):
            W[i] += sign(k_id)*SIGMA*np.random.standard_normal(W[i].shape)
    model = build_model()
    model.set_weights(W)
    s = env.reset()
    prev_x = None
    ep_r = 0.0
    while(True):
        cur_x = preprocess(s)
        x = cur_x - prev_x if prev_x is not None else np.zeros((80,80,1))
        prev_x = cur_x

        a = make_action(model, x)
        s, r, done, _ = env.step(a)
        ep_r += r
        if done: break
    del model
    return ep_r

def train(model, env, utility, pool):
    noise_seed = np.random.randint(0, INT, size=N_KID, dtype=np.uint32).repeat(2)
    W = model.get_weights()
    jobs = [pool.apply_async(play_game, (W.copy(), env, [noise_seed[k_id], k_id])) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]

    for i in range(len(W)):
        cumulative_update = np.zeros_like(W[i])
        for ui, k_id in enumerate(kids_rank):
            np.random.seed(noise_seed[k_id])
            cumulative_update += utility[ui] * sign(k_id) * np.random.standard_normal(W[i].shape)
        W[i] += LR*cumulative_update/(2*N_KID*SIGMA)
    model.set_weights(W)
    return model

if __name__ == "__main__":
	# utility instead reward for update parameters (rank transformation)
    # Wierstra, Daan, et al. "Natural evolution strategies." 2014
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    model = build_model()
    env = gym.make('Pong-v0').unwrapped
    pool = mp.Pool(processes=N_CORE)
    result = []
    for g in range(N_GENERATION):
        # Train
        model = train(model, env, utility, pool)
        # Test
        W = model.get_weights()
        ep_r = play_game(W.copy(), env, None)
        result.append(ep_r)
        print("Episode: %d | Reward: %d | Last 30 Reward: %f" %(g, ep_r, np.mean(result[-30:])))
        np.save('es_pong_result.npy', result)




















