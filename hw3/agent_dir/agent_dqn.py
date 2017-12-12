from agent_dir.agent import Agent
import random
import scipy.misc
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

def get_session(gpu_fraction=0.05):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.online_model = load_model('./model/dqn_keras_online_model.h5')
            self.env = env
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
        else:
            self.env = env
            self.state_size = (84, 84, 4)
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
            self.exploration_delta = 9.5*1e-7 # after 1000000, exploration_rate will be 0.05
            self.lr = 1e-4
            self.gamma = 0.99
            self.batch_size = 32
            self.timestep = 0

            self.Memory = {}
            self.memory_limit = 10000
            self.memory_count = 0

            self.online_update_frequency = 4
            self.target_update_frequency = 1000
            self.online_model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model()
            

            #self.optimizer = self.optimizer()
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #K.set_session(self.sess)
            #self.sess = tf.Session()
            #self.sess.run(tf.global_variables_initializer())

    '''
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
        py_x = self.online_model.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        #opt = RMSprop(lr=self.lr, epsilon=0.01)
        opt = Adam(lr=self.lr)
        updates = opt.get_updates(self.online_model.trainable_weights, [], loss)
        train = K.function([self.online_model.input, a, y], [loss], updates=updates)
        return train
    '''
    
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size, kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        #model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='linear'))
        model.add(LeakyReLU())
        model.add(Dense(self.action_size, kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=True):
        epsilon = 0.0005 if test==True else self.exploration_rate
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q = self.online_model.predict(np.expand_dims(observation, axis=0))
            return np.argmax(q[0])

    def memory_store(self, m):
        self.Memory[self.memory_count % self.memory_limit] = m
        self.memory_count += 1

    def learn(self):
        ids = np.random.choice(min(self.memory_count, self.memory_limit), size=self.batch_size)
        s = np.zeros((self.batch_size, 84, 84, 4))
        actions = np.zeros(self.batch_size, dtype=np.int)
        rewards = np.zeros(self.batch_size)
        s_ = np.zeros((self.batch_size, 84, 84, 4))
        done = np.zeros(self.batch_size, dtype=np.int)

        for i in range(self.batch_size):
            s = self.Memory[ids[i]][0]
            a = self.Memory[ids[i]][1]
            r = self.Memory[ids[i]][2]
            s_ = self.Memory[ids[i]][3]
            done = self.Memory[ids[i]][4]
        
        q_online = self.online_model.predict(s)
        q_target = self.target_model.predict(s_)
        y = q_online.copy()
        for i in range(self.batch_size):
            if done[i]==1:
                y[i, actions[i]] = rewards[i]
            else:
                y[i, actions[i]] = rewards[i] + self.gamma * np.max(q_target, axis=1)[i]

        loss = self.online_model.train_on_batch(s, y)
        
        #q_target = self.target_model.predict(s_)
        #y = rewards + self.gamma * np.max(q_target, axis=1)
        #loss = self.optimizer([s, actions, y])
        #return loss

    def train(self):
        #saver = tf.train.Saver()
        episodes = 1000000
        result = [] #result for each episode
        for e in range(1, episodes+1):
            obs = self.env.reset()
            episode_reward = 0
            max_q = 0
            num_step = 0
            total_loss = 0
            while True:
                a = self.make_action(obs, test=False)

                # add Max_Q value
                #q = self.online_model.predict(np.expand_dims(obs, axis=0))
                #max_q += np.max(q[0])
                #num_step += 1

                obs_, r, done, info = self.env.step(a)
                episode_reward += r
                self.memory_store((obs.copy(), a, r, obs_.copy(), int(done)))

                self.timestep += 1

                if self.timestep > self.memory_limit and (self.timestep%self.target_update_frequency)==0:
                    self.update_target_model()
                    #self.sess.run(self.update_target_op)

                if self.timestep > self.memory_limit and (self.timestep%self.online_update_frequency)==0:
                    self.learn()
                    #total_loss += loss

                if self.exploration_rate > 0.05:
                    self.exploration_rate -= self.exploration_delta

                obs = obs_.copy()

                if done:
                    result.append(episode_reward)
                    break

            rr = np.mean(result[-100:])
            print("Episode: %d | Reward: %d | Last 100: %f | step: %d | explore: %f" %(e, episode_reward, rr, self.timestep, self.exploration_rate))
            if (e%10) == 0:
                np.save('./result/dqn_keras_result.npy',result)
                self.online_model.save('./model/dqn_keras_online_model.h5')
                self.target_model.save('./model/dqn_keras_target_model.h5')
                #save_path = saver.save(self.sess, "./model/dqn_model03")



