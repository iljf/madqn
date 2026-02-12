from pyparsing import deque
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

        self.team_idx = 0
        self.ep_move_count_pred = {0: 0, 1: 0}
        self.step_move_count_pred = {0: 0, 1: 0}
        self.step_tag_count_pred = {0: 0, 1: 0}

        self.agent_action_deque_dict = {}  # 각 에이전트가 avg에 따라 액션(움직임,가만히있음,태그)을 어떻게 하는지 저장하기 위한 딕셔너리
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_action_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        dq_len = int(getattr(args, "deque_len", 400)) if args is not None else 400
        self.avg_move_deque_pred1 = deque(maxlen=dq_len)
        self.avg_move_deque_pred2 = deque(maxlen=dq_len)
        self.avg_tag_deque_pred1 = deque(maxlen=dq_len)
        self.avg_tag_deque_pred2 = deque(maxlen=dq_len)

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

    def set_team_idx(self,idx):
        self.team_idx = idx

    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def set_agent_pos(self, pos):
        self.pos = pos

    def set_agent_shared(self, view_range):
        self.view_range = view_range

    def set_agent_model(self,agent):
        self.gdqn = self.gdqns[agent]
        self.gdqn_target = self.gdqn_targets[agent]



    #########################################
    ##############move & count###############
    #########################################

    # move count 에 대한 매서드
    # for ep team move counting
    def ep_move_count(self):
        self.ep_move_count_pred[self.team_idx] += 1

    def reset_ep_move_count(self):
        self.ep_move_count_pred[0] = 0
        self.ep_move_count_pred[1] = 0

    # for step team move counting
    # 매 스텝이 시작할때마다 reset 을 해주어야함
    def step_move_count(self):
        self.step_move_count_pred[self.team_idx] += 1

    def reset_step_move_count(self):
        self.step_move_count_pred[0] = 0
        self.step_move_count_pred[1] = 0

    # tag count 에 대한 매서드
    # for ep team move counting
    def step_tag_count(self):
        self.step_tag_count_pred[self.team_idx] += 1

    def reset_step_tag_count(self):
        self.step_tag_count_pred[0] = 0
        self.step_tag_count_pred[1] = 0

    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    #avg_move
    def avg_move_append_pred1(self, move):
        self.avg_move_deque_pred1.append(move)

    def avg_move_append_pred2(self, move):
        self.avg_move_deque_pred2.append(move)

    #avg_tag
    def avg_tag_append_pred1(self, tag):
        self.avg_tag_deque_pred1.append(tag)

    def avg_tag_append_pred2(self, tag):
        self.avg_tag_deque_pred2.append(tag)
        

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

