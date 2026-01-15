import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
import random


class G_DQN(nn.Module):
    def __init__(self,  dim_act, observation_state, eps_decay: float):
        super(G_DQN, self).__init__()
        self.eps_decay = eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, self.dim_feature)

        #info
        self.dim_info = observation_state[2]
        self.gnn1_info = DenseSAGEConv(self.dim_feature,  128)
        self.gnn2_info = DenseSAGEConv(128, self.dim_feature)

        #DQN
        self.dim_input = observation_state[0] * observation_state[1] * observation_state[2]*2
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        #Book linear + Gate
        self.out_linear = nn.Linear(2*self.dim_feature, self.dim_feature)
        self.gate_net = nn.Linear(self.dim_feature, self.dim_feature)

        self.relu = nn.ReLU()


    def forward(self, x, adj, info):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        else:
            pass

        # ---- observation GNN1 ----
        x_pre = x.reshape(-1, self.dim_feature)
        x1 = self.gnn1(x_pre, adj)
        x1 = self.tanh(x1[0])
        x1 = self.gnn2(x1, adj).squeeze()
        x1 = self.tanh(x1)
        x1 = x1.reshape(self.observation_state)

        # ---- Book GNN2 ----
        info_pre = info.reshape(-1, self.dim_feature)
        x2 = self.gnn1_info(info_pre, adj)
        x2 = self.tanh(x2)
        x2 = self.gnn2_info(x2[0], adj).squeeze()
        x2 = self.tanh(x2)
        x2 = x2.reshape(self.observation_state)

        # ---- DQN ----
        q_input = torch.cat((x1,x2), dim=0).reshape(-1, self.dim_input).squeeze()
        x3 = self.FC1(q_input)
        x3 = self.tanh(x3)
        x3 = self.FC2(x3)

        # ---- Book outtake ----

        x1_detach = x1.detach() # observation
        x2_detach = x2.detach() # incoming info

        x2_1 = torch.cat((x1_detach,x2_detach), dim= -1) # detach so that graident won't flow
        x2_1 = self.out_linear(x2_1)

        gate = self.sigmoid(self.gate_net(x2_1)) # gate network

        x2_2 = x2_1 * gate
        shared1 = x2_2.reshape(self.observation_state)

        # L2 norms for outtake ratio
        l2_before = torch.norm(x, p=2)
        l2_outtake = torch.norm(shared1, p=2)
        outtake_ratio = l2_outtake / (l2_before + 1e-8)
        shared_sum = torch.mean(shared1)

        l2_intake = torch.norm(x2, p=2) / (torch.norm(info, p=2) + 1e-8)

        # np.mean(shared) 는 에이전트가 정보를 어떻게 남기는지 보려고 하는 것
        return x3, shared1, l2_before, l2_outtake, shared_sum, l2_intake , x2, outtake_ratio

    # def forward(self, x, adj, info):
    #
    #     try:
    #         torch.cuda.empty_cache()
    #     except:
    #         pass
    #
    #     if isinstance(x, np.ndarray):
    #         x = torch.tensor(x).float()
    #     else:
    #         pass
    #
    #     x_pre = x.reshape(-1, self.dim_feature)
    #
    #     x = self.gnn1(x_pre, adj)
    #     x = self.tanh(x[0])
    #     x = self.gnn2(x, adj).squeeze()
    #     x = self.tanh(x)
    #
    #     dqn = x[:, :self.dim_feature]
    #
    #     shared = self.tanh(x[:, self.dim_feature:])
    #
    #     shared = dqn * shared
    #     shared = shared.reshape(self.observation_state)
    #     # dqn = dqn.reshape(self.observation_state)
    #
    #     info_pre = info.reshape(-1, self.dim_feature)
    #     x1 = self.gnn1_info(info_pre, adj)
    #     x1 = self.tanh(x1)
    #     x1 = self.gnn2_info(x1[0], adj).squeeze()
    #     x1 = self.tanh(x1)
    #     x1 = x1.reshape(self.observation_state)
    #
    #     x = torch.cat((shared, x1), dim=0).reshape(-1, self.dim_input).squeeze()
    #
    #     x = self.FC1(x)
    #     x = self.tanh(x)
    #     x = self.FC2(x)
    #
    #     return x, shared.detach()



class ReplayBuffer:
   def __init__(self, capacity):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation, book , action , reward, next_observation, book_next, termination, truncation):
       self.buffer.append([observation, book, action , reward, next_observation, book_next, termination, truncation])



   def sample(self):
       sample = random.sample(self.buffer, 1)

       observation, book , action , reward, next_observation, book_next, termination, truncation = zip(*sample)
       return observation, book , action , reward, next_observation, book_next, termination, truncation

   def size(self):
      return len(self.buffer)



