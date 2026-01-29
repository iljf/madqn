from model_decen import G_DQN_1,G_DQN_2, ReplayBuffer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam


dim_act = 13

def king_adj(n) :

    A = np.zeros((n**2, n**2))

    for i in range(n**2):

        if i // n == 0 :

            if i % n == 0 :

                A[i, i+1] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

            elif i % n == n-1:

                A[i, i-1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1

            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

        elif i // n == n-1:

            if i % n == 0:

                A[i, i-n] = 1
                A[i, i+1-n] = 1
                A[i, i+1] = 1
            elif i % n == n-1:

                A[i, i-n] = 1
                A[i, i-1-n] = 1
                A[i, i-1] = 1
            else:
                A[i, i - 1] = 1
                A[i, i + 1] = 1
                A[i, i - 1-n] = 1
                A[i, i-n] = 1
                A[i, i+ 1-n] = 1

        else:
                if i % n == 0:
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i+1] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1

                elif i % n == n-1:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i-1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                else:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i-1] = 1
                    A[i, i+1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1


    for i in range(n**2):
        A[i,i] = 1

    return A

class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , buffer_size ,device = 'cpu', args=None):
        if args is None:
            raise ValueError("MADQN requires args to be passed")

        self.args = args
        self.device = device

        self.predator1_view_range = args.predator1_view_range
        self.predator2_view_range = args.predator2_view_range
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.n_prey = args.n_prey

        # shapes
        self.shared_shape = (args.map_size + (self.predator1_view_range-2)*2,
                             args.map_size + (self.predator1_view_range-2)*2, 3)
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs

        self.dim_act = dim_act
        self.epsilon = args.eps

        # adjacency matrices
        predator1_adj = king_adj(self.predator1_view_range*2)
        predator2_adj = king_adj(self.predator2_view_range*2)
        self.predator1_adj = torch.tensor(predator1_adj).float()
        self.predator2_adj = torch.tensor(predator2_adj).float()

        # networks
        self.gdqns = [G_DQN_1(self.dim_act, self.predator1_obs, args).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN_2(self.dim_act, self.predator2_obs, args).to(self.device) for _ in range(self.n_predator2)]

        self.gdqn_targets = [G_DQN_1(self.dim_act, self.predator1_obs, args).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN_2(self.dim_act, self.predator2_obs, args).to(self.device) for _ in range(self.n_predator2)]

        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=args.lr) for x in self.gdqns]

        self.criterion = nn.MSELoss()

        self.adj = None
        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.target_optimizer = None

        self.buffer = None


    def target_update(self, idx: int | None = None):
        if idx is None:
            for i in range(len(self.gdqns)):
                self.gdqn_targets[i].load_state_dict(self.gdqns[i].state_dict())
            return

        self.gdqn_targets[idx].load_state_dict(self.gdqns[idx].state_dict())


    def set_agent_info(self, agent):

        if agent[9] == "1":
            self.idx = int(agent[11:])
            self.adj = self.predator1_adj

        else:
            self.idx = int(agent[11:]) + self.n_predator1
            self.adj = self.predator2_adj

        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def get_action(self, state, mask=None):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        q_value = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device))

        # epsilon update
        self.epsilon *= self.args.eps_decay
        self.epsilon = max(self.epsilon, self.args.eps_min)

        if np.random.random() < self.epsilon:
            return random.randint(0, self.dim_act - 1)
        return torch.argmax(q_value).item()


    def replay(self):
        for _ in range(self.args.replay_times):

            self.gdqn_optimizer.zero_grad()

            observations, actions, rewards, next_observations, termination, truncation = self.buffer.sample()

            observation = observations[0]
            next_observation = next_observations[0]
            action = int(actions[0])
            reward = float(rewards[0])
            done = bool(termination[0]) or bool(truncation[0])

            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            next_obs_t = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
            adj = self.adj.to(self.device)

            q_all = self.gdqn(obs_t, adj)
            q_val = q_all[action]

            with torch.no_grad():
                next_q_all = self.gdqn_target(next_obs_t, adj)
                next_q_max = torch.max(next_q_all)

                target = torch.tensor(reward, dtype=torch.float32, device=self.device)
                if not done:
                    target = target + next_q_max * self.args.gamma

            loss = self.criterion(q_val, target)
            loss.backward()

            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass
