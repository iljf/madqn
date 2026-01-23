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

        # adjacency sizes based on view ranges
        pred1_view = self.args.predator1_view_range
        pred2_view = self.args.predator2_view_range
        self.predator1_adj = ((pred1_view) ** 2, (pred1_view) ** 2)
        self.predator2_adj = ((pred2_view) ** 2, (pred2_view) ** 2)

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
            self.adj = torch.ones(self.predator1_adj)
        else:
            self.idx = int(agent[11:]) + self.n_predator1
            self.adj = torch.ones(self.predator2_adj)



        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]


    def get_action(self, state, mask=None):

        q_value = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device))


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

            next_observations = torch.tensor(next_observations)
            observations = torch.tensor(observations)

            dim_feat = self.args.dim_feature if self.args is not None else 1
            next_observations = next_observations.reshape(-1, dim_feat)
            observations = observations.reshape(-1, dim_feat)


            # to device
            observations = observations.to(self.device)
            next_observations = next_observations.to(self.device)
            adj = self.adj.to(self.device)

            q_values = self.gdqn(observations.unsqueeze(0), adj.unsqueeze(0))
            q_values = q_values[0][actions]


            next_q_values = self.gdqn_target(next_observations.unsqueeze(0), adj.unsqueeze(0))
            next_q_values = torch.max(next_q_values)

            gamma = self.args.gamma if self.args is not None else 0.95
            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * gamma
            loss = self.criterion(q_values, targets.detach())
            loss.backward()
            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass
