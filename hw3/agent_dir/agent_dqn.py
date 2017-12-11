from agent_dir.agent import Agent
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

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
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
            self.online_net = torch.load('./model/dqn_torch_online_model.pt')

        else:
            self.env = env
            self.state_size = (84, 84, 4)
            self.action_size = self.env.action_space.n
            self.exploration_rate = 1.0
            self.exploration_delta = 9.5*1e-7 # after 1000000, exploration_rate will be 0.05
            #self.exploration_delta = 6.33333*1e-7 # after 1500000, exploration_rate will be 0.05
            self.lr = 1e-4
            self.gamma = 0.99
            self.batch_size = 32
            self.timestep = 0

            self.Memory = {}
            self.memory_limit = 10000
            self.memory_count = 0

            self.online_update_frequency = 4
            self.target_update_frequency = 1000

            self.online_net = DQN(self.action_size).cuda()
            self.target_net = DQN(self.action_size).cuda()
            self.optimizer = torch.optim.RMSprop(self.online_net.parameters(),lr=self.lr)
            self.loss_func = nn.MSELoss()


    def update_target_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        #self.target_net.eval

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=True):
        if observation.shape[0] != 4:
            observation = np.transpose(observation, (2,0,1))
        #obs = np.expand_dims(observation, axis=0) # -> (1,4,84,84)
        epsilon = 0.005 if test==True else self.exploration_rate

        x = Variable(torch.unsqueeze(torch.FloatTensor(observation), 0)).cuda()

        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_size)
        else:
            actions_value = self.online_net.forward(x)
            action = (torch.max(actions_value, 1)[1].data)[0]
            return action

    def memory_store(self, m):
        self.Memory[self.memory_count % self.memory_limit] = m
        self.memory_count += 1

    def learn(self):
        ids = np.random.choice(self.memory_limit, size=self.batch_size)
        s = np.zeros((self.batch_size, 4, 84, 84))
        actions = np.zeros(self.batch_size, dtype=np.int)
        rewards = np.zeros(self.batch_size)
        s_ = np.zeros((self.batch_size, 4, 84, 84))
        for i in range(self.batch_size):
            s[i] = self.Memory[ids[i]][0]
            actions[i] = self.Memory[ids[i]][1]
            rewards[i] = self.Memory[ids[i]][2]
            s_[i] = self.Memory[ids[i]][3]

        v_s = Variable(torch.FloatTensor(s)).cuda()
        v_a = Variable(torch.LongTensor(actions.tolist())).cuda()
        v_r = Variable(torch.FloatTensor(rewards)).cuda()
        v_s_ = Variable(torch.FloatTensor(s_)).cuda()

        q_eval = self.online_net(v_s).gather(1, v_a.view(-1,1))
        q_next = self.target_net(v_s_).detach()
        q_target = v_r.view(-1,1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        q_target = q_target.cuda()
        loss = self.loss_func(q_eval, q_target)
        loss = loss.cuda()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

    def train(self):
        #saver = tf.train.Saver()
        episodes = 1000000
        result = [] #result for each episode
        for e in range(1, episodes+1):
            obs = self.env.reset()
            obs = np.transpose(obs, (2,0,1)) # (84,84,4)->(4,84,84)
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
                obs_ = np.transpose(obs_, (2,0,1)) # (84,84,4)->(4,84,84)
                episode_reward += r
                self.memory_store((obs.copy(), a, r, obs_.copy()))

                self.timestep += 1

                if self.timestep > self.memory_limit and (self.timestep%self.target_update_frequency)==0:
                    self.update_target_model()

                if self.timestep > self.memory_limit and (self.timestep%self.online_update_frequency)==0:
                    self.learn()

                if self.exploration_rate > 0.05:
                    self.exploration_rate -= self.exploration_delta

                obs = obs_.copy()

                if done:
                    result.append(episode_reward)
                    break

            rr = np.mean(result[-30:])
            #print("Episode: %d | Reward: %d | Last 30: %f | step: %d | explore: %f | Max_Q: %f" %(e, episode_reward, rr, self.timestep, self.exploration_rate, max_q/num_step))
            print("Episode: %d | Reward: %d | Last 30: %f | step: %d | explore: %f" %(e, episode_reward, rr, self.timestep, self.exploration_rate))
            if (e%10) == 0:
                np.save('./result/dqn_torch_result.npy',result)
                torch.save(self.online_net, './model/dqn_torch_online_model.pt')
                torch.save(self.target_net, './model/dqn_torch_target_model.pt')



