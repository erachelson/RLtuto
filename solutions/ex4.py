import numpy as np
import torch
import torch.nn as nn

class DQN_agent:
    def __init__(self, config, model):
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'])
        self.epsilon = config['epsilon']
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config['learning_rate'])
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode = 0
        state = env.reset()
        while episode < max_episode:
            # select epsilon-greedy action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.nb_actions)
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            # train
            self.gradient_step()
            # next transition
            if done:
                episode += 1
                state = env.reset()
            else:
                state = next_state