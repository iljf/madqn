from model_cen import G_DQN, G_ReplayBuffer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam

dim_act = 13


class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, dim_act, entire_state, device='cpu', buffer_size=10000, args=None):
        self.entire_state = entire_state
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.device = device
        self.args = args

        # initialize epsilon
        self.epsilon = self.args.eps

        # Graph adjacency must match H*W nodes of entire_state.
        self.global_adj = torch.tensor(self.king_adj(self.entire_state[0])).float()

        self.gdqns = [G_DQN(self.dim_act, self.entire_state).to(self.device) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_targets = [G_DQN(self.dim_act, self.entire_state).to(self.device) for _ in range(self.n_predator1 + self.n_predator2)]

        self.buffers = [G_ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=(self.args.lr if self.args is not None else 0.001)) for x in self.gdqns]

        self.criterion = nn.MSELoss()

        self.adj = None
        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.buffer = None


    def king_adj(self, n: int):
        
        A = np.zeros((n ** 2, n ** 2), dtype=np.float32)

        for i in range(n ** 2):
            if i // n == 0:
                if i % n == 0:
                    A[i, i + 1] = 1
                    A[i, i + n] = 1
                    A[i, i + 1 + n] = 1
                elif i % n == n - 1:
                    A[i, i - 1] = 1
                    A[i, i - 1 + n] = 1
                    A[i, i + n] = 1
                else:
                    A[i, i - 1] = 1
                    A[i, i + 1] = 1
                    A[i, i - 1 + n] = 1
                    A[i, i + n] = 1
                    A[i, i + 1 + n] = 1

            elif i // n == n - 1:
                if i % n == 0:
                    A[i, i - n] = 1
                    A[i, i + 1 - n] = 1
                    A[i, i + 1] = 1
                elif i % n == n - 1:
                    A[i, i - n] = 1
                    A[i, i - 1 - n] = 1
                    A[i, i - 1] = 1
                else:
                    A[i, i - 1] = 1
                    A[i, i + 1] = 1
                    A[i, i - 1 - n] = 1
                    A[i, i - n] = 1
                    A[i, i + 1 - n] = 1

            else:
                if i % n == 0:
                    A[i, i - n] = 1
                    A[i, i + 1 - n] = 1
                    A[i, i + 1] = 1
                    A[i, i + n] = 1
                    A[i, i + 1 + n] = 1
                elif i % n == n - 1:
                    A[i, i - 1 - n] = 1
                    A[i, i - n] = 1
                    A[i, i - 1] = 1
                    A[i, i - 1 + n] = 1
                    A[i, i + n] = 1
                else:
                    A[i, i - 1 - n] = 1
                    A[i, i - n] = 1
                    A[i, i + 1 - n] = 1
                    A[i, i - 1] = 1
                    A[i, i + 1] = 1
                    A[i, i - 1 + n] = 1
                    A[i, i + n] = 1
                    A[i, i + 1 + n] = 1

        for i in range(n ** 2):
            A[i, i] = 1
        return A




    def target_update(self):

        for i in range(self.n_predator1 + self.n_predator2):
            weights = self.gdqns[i].state_dict()
            self.gdqn_targets[i].load_state_dict(weights)


    def set_agent_model(self,agent):
        self.gdqn = self.gdqns[agent]
        self.gdqn_target = self.gdqn_targets[agent]



    def set_agent_info(self, agent):

        if agent[9] == "1":
            self.idx = int(agent[11:])
        else:
            self.idx = int(agent[11:]) + self.n_predator1

        # same global adjacency for all agents in centralized setting
        self.adj = self.global_adj



        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]


    def get_action(self, state, mask=None):

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        adj_t = self.adj.to(self.device)
        q_value = self.gdqn(state_t, adj_t)


        if self.args is not None:
            self.epsilon *= self.args.eps_decay
            self.epsilon = max(self.epsilon, self.args.eps_min)

        if np.random.random() < self.epsilon:
            return random.randint(0, self.dim_act - 1)
        return torch.argmax(q_value).item()

    def replay(self):
        for _ in range(self.args.replay_times if self.args is not None else 1):
            self.gdqn_optimizer.zero_grad()

            observations, actions, rewards, next_observations, termination, truncation = self.buffer.sample()

            obs = torch.as_tensor(observations[0], dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(next_observations[0], dtype=torch.float32, device=self.device)
            action = int(actions[0])
            reward = float(rewards[0])
            done = bool(termination[0]) or bool(truncation[0])

            adj = self.adj.to(self.device)

            q_all = self.gdqn(obs, adj)              
            q_val = q_all[action]

            next_q_all = self.gdqn_target(next_obs, adj) 
            next_q_max = torch.max(next_q_all)

            gamma = self.args.gamma if self.args is not None else 0.95
            target = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not done:
                target = target + next_q_max * gamma

            loss = self.criterion(q_val, target.detach())
            loss.backward()
            self.gdqn_optimizer.step()

