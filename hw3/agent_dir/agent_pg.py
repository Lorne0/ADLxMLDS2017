from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
def get_session(gpu_fraction=0.6):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        self.env = env
        self.n_features = 80*80
        self.n_actions = 6
        self.lr = 0.001
        self.gamma = 0.99
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.prev_x = None

        self.model = self._build_model()
        self.result = []
        self.start_episode = 0
        # self.model = load_model('./model/pg_keras_model.h5')
        # self.result = np.load('./result/pg_keras_result.npy')
        # self.start_episode = 0

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((80, 80, 1), input_shape=(self.n_features,)))
        model.add(Conv2D(filters=32, kernel_size=(6,6), strides=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.n_actions, activation='softmax'))
        opt = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def init_game_setting(self):
        self.prev_x = None
    
    def preprocess(self, I):
        I = I[34:194]
        I = I[::2, ::2, 0] # 80x80
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()
        
    def remember(self, state, action, prob, reward):
        y = np.zeros(self.n_actions)
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def make_action(self, observation, test=True):
        if test == True:
            cur_x = self.preprocess(observation)
            observation = cur_x - self.prev_x if self.prev_x is not None else np.zeros(state_size)
            self.prev_x = cur_x

        observation = observation.reshape([1, observation.shape[0]])
        aprob = self.model.predict(observation, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.n_actions, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            #if rewards[t] != 0:
            #    running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def learn(self):
        gradients = np.vstack(self.gradients)
        rewards = np.array(self.rewards)
        #rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = np.reshape(rewards, (-1, 1))
        gradients *= rewards
        #X = np.squeeze(np.vstack([self.states]))
        X = np.array(self.states)
        Y = self.probs + self.lr * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def train(self):
        n_features = 80*80
        n_actions = 6
        episodes = 20000
        se = self.start_episode
        for episode in range(se, episodes+1):
            state = self.env.reset()
            score = 0
            self.prev_x = None
            while True:

                cur_x = self.preprocess(state)
                x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(n_features)
                self.prev_x = cur_x

                action, prob = self.make_action(x, test=False)
                state, reward, done, info = self.env.step(action)
                score += reward
                self.remember(x, action, prob, reward)

                if done:
                    self.learn()
                    self.result.append(score)
                    print('Episode: %d - Score: %d. - Last 30: %f' % (episode, score, np.mean(self.result[-30:])))
                    score = 0
                    if episode % 10 == 0:
                        self.model.save('./model/pg_keras_model.h5')
                        np.save('./result/pg_keras_result.npy',self.result)
                    break
                
        self.env.close()
                
    
