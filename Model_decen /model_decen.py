import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
import random


class G_DQN_1(nn.Module):
    def __init__(self,  dim_act, observation_state, args):
        super(G_DQN_1, self).__init__()
        if args is None:
            raise ValueError("G_DQN_1 requires args to be passed")

        self.args = args
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, 32)
        self.sig = nn.Sigmoid()

        #DQN
        self.dim_input = ((args.predator1_view_range *2)**2)*32
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        self.relu = nn.ReLU()


    def forward(self, x, adj):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        H, W, C = self.observation_state
        num_nodes = H * W

        # normalize input to [B, N, C]
        if x.dim() == 3:
            x_b = x.reshape(num_nodes, C).unsqueeze(0)
        elif x.dim() == 4:
            x_b = x.reshape(-1, num_nodes, C)
        else:
            x_b = x.reshape(-1, num_nodes, C)

        # ensure adj has batch dimension [B, N, N]
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0)
        else:
            adj_b = adj

        x1 = self.gnn1(x_b, adj_b)
        x1 = self.tanh(x1)
        x1 = self.gnn2(x1, adj_b)
        x1 = self.tanh(x1)

        x2 = x1.reshape(x1.size(0), -1)  # [B, N*32]
        x2 = self.FC1(x2)
        x2 = self.tanh(x2)
        q = self.FC2(x2)

        if q.size(0) == 1:
            return q.squeeze(0)
        return q

class G_DQN_2(nn.Module):
    def __init__(self,  dim_act, observation_state, args):
        super(G_DQN_2, self).__init__()
        if args is None:
            raise ValueError("G_DQN_2 requires args to be passed")

        self.args = args
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, 32)
        self.sig = nn.Sigmoid()

        #DQN
        self.dim_input = ((args.predator2_view_range *2)**2)*32
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        self.relu = nn.ReLU()


    def forward(self, x, adj):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        H, W, C = self.observation_state
        num_nodes = H * W

        # normalize input to [B, N, C]
        if x.dim() == 3:
            x_b = x.reshape(num_nodes, C).unsqueeze(0)
        elif x.dim() == 4:
            x_b = x.reshape(-1, num_nodes, C)
        else:
            x_b = x.reshape(-1, num_nodes, C)

        # ensure adj has batch dimension [B, N, N]
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0)
        else:
            adj_b = adj

        x1 = self.gnn1(x_b, adj_b)
        x1 = self.tanh(x1)
        x1 = self.gnn2(x1, adj_b)
        x1 = self.tanh(x1)

        x2 = x1.reshape(x1.size(0), -1)  # [B, N*32]
        x2 = self.FC1(x2)
        x2 = self.tanh(x2)
        q = self.FC2(x2)

        if q.size(0) == 1:
            return q.squeeze(0)
        return q


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def put(self, observation , action , reward, next_observation, termination, truncation):
       self.buffer.append([observation, action , reward, next_observation, termination, truncation])



    def sample(self):
       sample = random.sample(self.buffer, 1)

       observation, action , reward, next_observation, termination, truncation = zip(*sample)
       return observation, action , reward, next_observation, termination, truncation

    def size(self):
      return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, dim_act, observation_state):
        super(DQN, self).__init__()
        self.observation_state = observation_state
        self.dim_act = dim_act

        # DQN
        self.dim_input = np.prod(observation_state)
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        self.relu = nn.ReLU()

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        else:
            pass

        x = state.reshape(-1, self.dim_input)
        x = self.relu(self.FC1(x))
        x = self.FC2(x)

        return x
