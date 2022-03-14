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
    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1,1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1,1)
        
        values, policies = self.a2c_net(states)
        next_values, _ = self.a2c_net(next_states)
        target = rewards + self.gamma * next_values * (1-dones)

        advantage = target - values
        log_prob = torch.log(policies)
        j = advantage * log_prob[range(len(actions)), actions].view(-1,1)
        
        actor_loss = -j.mean()
        critic_loss = advantage.pow(2).mean()
        loss = self.actor_ratio * actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    ep_rewards = deque(maxlen=100)
    
    env = gym.make("CartPole-v0")
    agent = A2C(env, actor_ratio=0.2, gamma=0.99, learning_rate=5e-3)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)
   
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            if done:
                ep_rewards.append(sum(rewards))
                agent.learn(states, actions, rewards, next_states, dones)
                
                if i % 100 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()
