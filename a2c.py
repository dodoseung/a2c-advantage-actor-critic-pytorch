# Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. PMLR, 2016.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
from collections import deque

class A2CNet(nn.Module):
    def __init__(self, input, output):
        super(A2CNet, self).__init__()
        self.input = nn.Linear(input, 16)
        self.fc = nn.Linear(16, 16)
	
        self.value = nn.Linear(16, 1)
        self.policy = nn.Linear(16, output)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        
        value = self.value(x)
        policy = F.softmax(self.policy(x))
        return value, policy
    
class A2C():
    def __init__(self, env, actor_ratio=0.5, gamma=0.95, learning_rate=1e-3):
        super(A2C, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
               
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.a2c_net = A2CNet(self.state_num, self.action_num).to(self.device)
        self.optimizer = optim.Adam(self.a2c_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.actor_ratio = actor_ratio
        
    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        _, policy = self.a2c_net(state)
        policy = policy.cpu().detach().numpy()
        action = np.random.choice(self.action_num, 1, p=policy[0])
        return action[0]

    # Learn the policy
    # j: Policy objective function
    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        value, policy = self.a2c_net(state)
        next_value, _ = self.a2c_net(next_state)
        target = reward + self.gamma * next_value * (1-done)

        advantage = target - value
        log_prob = torch.log(policy)
        j = advantage * log_prob[action]
        
        actor_loss = -j
        critic_loss = advantage.pow(2)
        loss = self.actor_ratio * actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    ep_rewards = deque(maxlen=100)
    
    env = gym.make("CartPole-v0")
    agent = A2C(env, actor_ratio=0.2, gamma=0.99, learning_rate=1e-3)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        rewards = []

        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)
   
            agent.learn(state, action, reward, next_state, done)
            rewards.append(reward)
            
            if done:
                ep_rewards.append(sum(rewards))
                
                if i % 100 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()
