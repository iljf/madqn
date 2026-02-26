from magent2_backup.environments import hetero_adversarial_v1
from MADQN import MADQN

import numpy as np
import torch as th
import argparse
import wandb
import os


def _to_float(x):
    if th.is_tensor(x):
        return float(x.detach().item())
    return float(x)


def get_args():
    parser = argparse.ArgumentParser(description='MADQN')

    parser.add_argument('--dim_feature', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--eps_decay', type=float, default=0.9999)
    parser.add_argument('--eps_min', type=float, default=0.01)
    parser.add_argument('--max_update_steps', type=int, default=3000)
    parser.add_argument('--total_ep', type=int, default=7)
    parser.add_argument('--book_decay', type=float, default=0.1)
    parser.add_argument("--book_term", type=int, default=4)
    parser.add_argument('--ep_save', type=int, default=1000)
    parser.add_argument('--jitter_std', type=float, default=0.5)

    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--trainstart_buffersize', type=int, default=6000)
    parser.add_argument('--deque_len', type=int, default=400)
    parser.add_argument('--plot_term', type=int, default=10)

    parser.add_argument('--replay_times', type=int, default=32)
    parser.add_argument('--target_update', type=int, default=10)

    parser.add_argument('--tau_predator1', type=float, default=0.2)
    parser.add_argument('--tau_predator2', type=float, default=0.2)
    parser.add_argument('--lamda_predator1', type=float, default=0)
    parser.add_argument('--lamda_predator2', type=float, default=0)

    parser.add_argument('--map_size', type=int, default=24)
    parser.add_argument('--predator1_view_range', type=int, default=10)
    parser.add_argument('--predator2_view_range', type=int, default=5)
    parser.add_argument('--n_predator1', type=int, default=9)
    parser.add_argument('--n_predator2', type=int, default=9)
    parser.add_argument('--n_prey', type=int, default=36)
    parser.add_argument('--tag_reward', type=float, default=3) # 3
    parser.add_argument('--tag_penalty', type=float, default=-0.2) # -0.2
    parser.add_argument('--move_penalty', type=float, default=-0.15) # -0.15

    parser.add_argument('--seed', type=int, default=125)

    return parser.parse_args()

args = get_args()
wandb.init(project="madqn_test", entity='hails',config=args.__dict__)
wandb.run.name = 'R1_lamda_0_default'

device = th.device("cuda" if th.cuda.is_available() else "cpu")

render_mode = 'rgb_array'

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey
dim_feature = args.dim_feature

shared_shape = (
args.map_size + (predator1_view_range - 2) * 2, args.map_size + (predator1_view_range - 2) * 2, dim_feature)
predator1_obs = (predator1_view_range * 2, predator1_view_range * 2, dim_feature)
predator2_obs = (predator2_view_range * 2, predator2_view_range * 2, dim_feature)
dim_act = 13

batch_size = 1
shared = th.zeros(shared_shape).to(device)
madqn = MADQN(args, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, shared_shape, shared, args.buffer_size,
              device)

def process_array_1(arr):  # predator1 (obs, team, team_hp, predator2, predator2 hp, prey, prey hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]))

    return result


def process_array_2(arr):  # predator2 (obs, team, team_hp, prey, prey hp, predator2, predator2 hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 3], arr[:, :, 2]))

    return result



def check_zero_size_min_pred1(list):
    if list.size > 0:
        min_value = np.min(list)
    else:
        min_value = args.predator1_view_range + 1

    return min_value


def check_zero_size_min_pred2(list):
    if list.size > 0:
        min_value = np.min(list)
    else:
        min_value = args.predator2_view_range + 1

    return min_value


def check_zero_size_avg_pred1(list):
    if list.size > 0:
        avg_value = np.mean(list)
    else:
        avg_value = args.predator1_view_range + 1

    return avg_value


def check_zero_size_avg_pred2(list):
    if list.size > 0:
        avg_value = np.mean(list)
    else:
        avg_value = args.predator2_view_range + 1

    return avg_value


def main():
    env = hetero_adversarial_v1.env(map_size=args.map_size,
                                    minimap_mode=False,
                                    tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps,
                                    extra_features=False,
                                    render_mode=render_mode,
                                    predator1_view_range=args.predator1_view_range,
                                    predator2_view_range=args.predator2_view_range,
                                    n_predator1=args.n_predator1,
                                    n_predator2=args.n_predator2,
                                    n_prey=args.n_prey,
                                    tag_reward=args.tag_reward,

    )

    step_idx_ep = 0

    for ep in range(args.total_ep):

        # shared book reset every episode
        shared = th.zeros(shared_shape)
        madqn.reset_shared(shared)

        # reset ep_move_count
        madqn.reset_ep_move_count()

        # env reset
        env.reset(seed=args.seed)

        observations_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            observations_dict[agent_idx] = []

        reward_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            reward_dict[agent_idx] = []

        move_penalty_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            move_penalty_dict[agent_idx] = []

        action_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            action_dict[agent_idx] = []

        termination_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            termination_dict[agent_idx] = []

        truncation_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            truncation_dict[agent_idx] = []

        book_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            book_dict[agent_idx] = []

        shared_info_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            shared_info_dict[agent_idx] = []

        agent_pos = {}
        for agent_idx in range(n_predator1 + n_predator2):
            agent_pos[agent_idx] = []

        entire_pos = []

        handles = env.env.env.env.env.get_handles()
        pos_predator1 = env.env.env.env.env.get_pos(handles[0])
        pos_predator2 = env.env.env.env.env.get_pos(handles[1])
        entire_pos.append(np.concatenate((pos_predator1, pos_predator2)))

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0

        # episode reward without move penalty
        ep_reward_np = 0
        ep_reward_pred1_np = 0
        ep_reward_pred2_np = 0

        # iteration for single agent
        n_iteration = 0

        step_action_type_count = {
            0: {0: 0, 1: 0, 2: 0},  # predator1 team
            1: {0: 0, 1: 0, 2: 0},  # predator2 team
        }
        
        print("ep:", ep, '*' * 30)
        
        for agent in env.agent_iter():

            n_agents = (args.n_predator1 + args.n_predator2 + args.n_prey)
            step_idx_ep = n_iteration // n_agents

            '''
            from this point on, every append to dict and logging is based on **POST STEP**
            which means that the information is from the last step ([-1] index)
            '''
            # ---- step boundary (log previous completed step) ----
            if ((n_iteration % n_agents) == 0) and (n_agents > 0):
                
                if step_idx_ep > 0:
                    madqn.shared_decay() 

                    ### Book append & decay ###
                    if step_idx_ep != args.max_update_steps:

                        if step_idx_ep <= args.book_term:

                            for idx in range(n_predator1 + n_predator2):
                                madqn.set_agent_pos(agent_pos[idx][-1]) # agent position

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                
                                if len(action_dict[idx]) > 0 and action_dict[idx][-1] == 2:
                                    madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

                        else:
                            # erase last step book information
                            for idx in range(n_predator1 + n_predator2):
                                madqn.set_agent_pos(agent_pos[idx][-(args.book_term + 1)]) # agent position

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                # self.to_guestbook(shared_info.to('cpu'))
                                # remove only if that past step actually wrote into the guestbook (action==2)
                                if (len(action_dict[idx]) >= (args.book_term + 1) and action_dict[idx][-(args.book_term + 1)] == 2):
                                    madqn.to_guestbook(-(args.book_decay ** (args.book_term)) * shared_info_dict[idx][-(args.book_term + 1)].to('cpu'))

                            # Add recent Step information
                            for idx in range(n_predator1 + n_predator2):
                                madqn.set_agent_pos(agent_pos[idx][-1])

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                # add recent step info only when the agent took action==2
                                if len(action_dict[idx]) > 0 and action_dict[idx][-1] == 2:
                                    madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))


                    madqn.avg_dist_append_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))  # step
                    madqn.min_dist_append_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))  # step
                    madqn.avg_move_append_pred1((madqn.step_move_count_pred[0]) / n_predator1)  # step
                    madqn.avg_tag_append_pred1((madqn.step_tag_count_pred[0]) / n_predator1)  # step

                    madqn.avg_dist_append_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))  # step
                    madqn.min_dist_append_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))  # step
                    madqn.avg_move_append_pred2((madqn.step_move_count_pred[1]) / n_predator2)  # step
                    madqn.avg_tag_append_pred2((madqn.step_tag_count_pred[1]) / n_predator2)  # step

                    # action-type ratios for the previous completed step
                    predator1_total = step_action_type_count[0][0] + step_action_type_count[0][1] + step_action_type_count[0][2]
                    predator2_total = step_action_type_count[1][0] + step_action_type_count[1][1] + step_action_type_count[1][2]
                    all_total = predator1_total + predator2_total

                    def ratio_count(i, j):
                        return float(i) / float(j) if j > 0 else 0.0

                    pred1_ratio_stay = ratio_count(step_action_type_count[0][0], predator1_total)
                    pred1_ratio_move = ratio_count(step_action_type_count[0][1], predator1_total)
                    pred1_ratio_tag = ratio_count(step_action_type_count[0][2], predator1_total)

                    pred2_ratio_stay = ratio_count(step_action_type_count[1][0], predator2_total)
                    pred2_ratio_move = ratio_count(step_action_type_count[1][1], predator2_total)
                    pred2_ratio_tag = ratio_count(step_action_type_count[1][2], predator2_total)

                    all_ratio_stay = ratio_count(step_action_type_count[0][0] + step_action_type_count[1][0], all_total)
                    all_ratio_move = ratio_count(step_action_type_count[0][1] + step_action_type_count[1][1], all_total)
                    all_ratio_tag = ratio_count(step_action_type_count[0][2] + step_action_type_count[1][2], all_total)

                    madqn.reset_step_move_count()
                    madqn.reset_step_tag_count()
                    madqn.reset_summation_team_dist()

                    # reset
                    step_action_type_count[0] = {0: 0, 1: 0, 2: 0}
                    step_action_type_count[1] = {0: 0, 1: 0, 2: 0}

                    handles = env.env.env.env.env.get_handles()
                    pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                    pos_predator2 = env.env.env.env.env.get_pos(handles[1])

                    pos_list = np.concatenate((pos_predator1, pos_predator2), axis=0)
                    ratio_matrix = madqn.calculate_Overlap_ratio(pos_list)

                    for idx, value in enumerate(ratio_matrix[:, 0]):
                        madqn.agent_graph_overlap_pred1_deque_dict[idx].append(value)

                    for idx, value in enumerate(ratio_matrix[:, 1]):
                        madqn.agent_graph_overlap_pred2_deque_dict[idx].append(value)

                    prev_step = step_idx_ep - 1 # logging based on previous  step

                    step_rewards = 0.0
                    step_penalty_rewards = 0.0

                    # team logging
                    step_reward_pred1 = 0.0
                    step_reward_pred2 = 0.0
                    step_penalty_pred1 = 0.0
                    step_penalty_pred2 = 0.0
                    
                    '''
                    In step 0, no dict has any value yet. So skip
                    '''
                    for i, agent_rewards in enumerate(reward_dict.values()):  # both pred1 & pred2
                        if len(agent_rewards) < 1:
                            continue
                        r = float(np.sum(agent_rewards[-1]))
                        step_rewards += r
                        if i < len(reward_dict) // 2:
                            step_reward_pred1 += r
                        else:
                            step_reward_pred2 += r

                    for i, penalties in enumerate(move_penalty_dict.values()):  # both pred1 & pred2
                        if len(penalties) < 1:
                            continue
                        p = float(np.sum(penalties[-1]))
                        step_penalty_rewards += p
                        if i < len(move_penalty_dict) // 2:
                            step_penalty_pred1 += p
                        else:
                            step_penalty_pred2 += p

                    step_rewards_total = step_rewards + step_penalty_rewards
                    step_rewards_pred1 = step_reward_pred1 + step_penalty_pred1
                    step_rewards_pred2 = step_reward_pred2 + step_penalty_pred2

                    ep_reward += step_rewards_total
                    ep_reward_pred1 += step_rewards_pred1
                    ep_reward_pred2 += step_rewards_pred2

                    ep_reward_pred1_np += step_reward_pred1
                    ep_reward_pred2_np += step_reward_pred2
                    ep_reward_np += (step_reward_pred1 + step_reward_pred2)

                    if step_idx_ep > 1:
                        # transition reward for replay buffer
                        transition_rewards = 0.0
                        transition_penalty_rewards = 0.0

                        for agent_rewards in reward_dict.values():
                            if len(agent_rewards) >= 2:
                                transition_rewards += float(np.sum(agent_rewards[-2]))

                        for penalties in move_penalty_dict.values():
                            if len(penalties) >= 2:
                                transition_penalty_rewards += float(np.sum(penalties[-2]))

                        transition_rewards_total = transition_rewards + transition_penalty_rewards

                        for idx in range(n_predator1 + n_predator2):
                            madqn.set_agent_buffer(idx)

                            if not (
                                len(observations_dict[idx]) >= 2
                                and len(book_dict[idx]) >= 2
                                and len(action_dict[idx]) >= 2
                                and len(termination_dict[idx]) >= 2
                                and len(truncation_dict[idx]) >= 2
                            ):
                                continue

                            madqn.buffer.put(
                                observations_dict[idx][-2],
                                book_dict[idx][-2],
                                action_dict[idx][-2],
                                transition_rewards_total,
                                observations_dict[idx][-1],
                                book_dict[idx][-1],
                                termination_dict[idx][-2],
                                truncation_dict[idx][-2],
                            )

                    metrics = {
                        "steps/total_step_reward": step_rewards_total,
                        "predator1/step_reward": step_rewards_pred1,
                        "predator2/step_reward": step_rewards_pred2,
                        "steps/total_step_reward_np": step_reward_pred1 + step_reward_pred2,
                        "predator1/step_reward_np": step_reward_pred1,
                        "predator2/step_reward_np": step_reward_pred2,
                        "predator1/avg_tag_count": madqn.avg_tag_deque_pred1[-1],
                        "predator2/avg_tag_count": madqn.avg_tag_deque_pred2[-1],
                        "predator1/avg_move_count": madqn.avg_move_deque_pred1[-1],
                        "predator2/avg_move_count": madqn.avg_move_deque_pred2[-1],

                        # per-step action-type ratios (0=stay, 1=move, 2=tag)
                        "predator1/action_ratio_stay": pred1_ratio_stay,
                        "predator1/action_ratio_move": pred1_ratio_move,
                        "predator1/action_ratio_tag": pred1_ratio_tag,
                        "predator2/action_ratio_stay": pred2_ratio_stay,
                        "predator2/action_ratio_move": pred2_ratio_move,
                        "predator2/action_ratio_tag": pred2_ratio_tag,
                        "steps/action_ratio_stay": all_ratio_stay,
                        "steps/action_ratio_move": all_ratio_move,
                        "steps/action_ratio_tag": all_ratio_tag,
                    }

                    for idx, value in madqn.agent_graph_overlap_pred1_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator1
                            metrics[f"predator1/Overlap_ratio_predator1_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator1
                            metrics[f"predator2/Overlap_ratio_predator1_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.agent_graph_overlap_pred2_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator2
                            metrics[f"predator1/Overlap_ratio_predator2_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator2
                            metrics[f"predator2/Overlap_ratio_predator2_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.intake_sum_with_pred1_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator1
                            metrics[f"predator1/Sum_intake_diff_predator1_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator1
                            metrics[f"predator2/Sum_intake_diff_predator1_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.intake_sum_with_pred2_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator2
                            metrics[f"predator1/Sum_intake_diff_predator2_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator2
                            metrics[f"predator2/Sum_intake_diff_predator2_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.intake_inner_with_pred1_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator1
                            metrics[f"predator1/inner_intake_diff_predator1_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator1
                            metrics[f"predator2/inner_intake_diff_predator1_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.intake_inner_with_pred2_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1: # predator1 agents with predator2
                            metrics[f"predator1/inner_intake_diff_predator2_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1 # predator2 agents with predator2
                            metrics[f"predator2/inner_intake_diff_predator2_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.l2_outtake_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/total_outtake_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/total_outtake_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.l2_intake_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/total_intake_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/total_intake_P2_{pred2_idx}"] = last_value

                    for idx, dq in madqn.agent_action_deque_dict.items():
                        if len(dq) == 0:
                            continue
                        last_value = dq[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/action_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/action_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.tiles_number_with_pred1_deque_dict.items(): # overlapping tiles + ratio
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        # normalize by self view-area tiles (per-agent ratio in [0,1])
                        agent_view_range = predator1_view_range if idx < n_predator1 else predator2_view_range
                        total_tiles = float((agent_view_range * 2) ** 2)
                        last_ratio = float(last_value) / total_tiles if total_tiles > 0 else 0.0

                        if idx < n_predator1:
                            metrics[f"predator1/n_overlapping_tiles_predator1_P1_{idx}"] = last_value
                            metrics[f"predator1/overlap_tiles_ratio_predator1_P1_{idx}"] = last_ratio
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/n_overlapping_tiles_predator1_P2_{pred2_idx}"] = last_value
                            metrics[f"predator2/overlap_tiles_ratio_predator1_P2_{pred2_idx}"] = last_ratio

                    for idx, value in madqn.tiles_number_with_pred2_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        # normalize by self view-area tiles (per-agent ratio in [0,1])
                        agent_view_range = predator1_view_range if idx < n_predator1 else predator2_view_range
                        total_tiles = float((agent_view_range * 2) ** 2)
                        last_ratio = float(last_value) / total_tiles if total_tiles > 0 else 0.0

                        if idx < n_predator1:
                            metrics[f"predator1/n_overlapping_tiles_predator2_P1_{idx}"] = last_value
                            metrics[f"predator1/overlap_tiles_ratio_predator2_P1_{idx}"] = last_ratio
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/n_overlapping_tiles_predator2_P2_{pred2_idx}"] = last_value
                            metrics[f"predator2/overlap_tiles_ratio_predator2_P2_{pred2_idx}"] = last_ratio

                    for idx, value in madqn.agent_avg_dist_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/avg_distance_prey_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/avg_distance_prey_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.agent_min_dist_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/shortest_distance_prey_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/shortest_distance_prey_P2_{pred2_idx}"] = last_value

                    for idx, value in madqn.prey_number_deque_dict.items():
                        if len(value) == 0:
                            continue
                        last_value = value[-1]

                        if idx < n_predator1:
                            metrics[f"predator1/n_prey_observation_P1_{idx}"] = last_value
                        else:
                            pred2_idx = idx - n_predator1
                            metrics[f"predator2/n_prey_observation_P2_{pred2_idx}"] = last_value

                    def _team_mean(d, start, end):
                        vals = []
                        for i in range(start, end):
                            dq = d.get(i, [])
                            if len(dq) > 0:
                                vals.append(dq[-1])
                        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

                    def _team_mean_tiles_ratio(d, start, end):
                        vals = []
                        for i in range(start, end):
                            dq = d.get(i, [])
                            if len(dq) == 0:
                                continue
                            agent_view_range = predator1_view_range if i < n_predator1 else predator2_view_range
                            total_tiles = float((agent_view_range * 2) ** 2)
                            vals.append(float(dq[-1]) / total_tiles if total_tiles > 0 else 0.0)
                        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

                    n1 = n_predator1
                    n2 = n_predator1 + n_predator2

                    team_metrics = {
                        "steps/predator1_mean_avg_distance_prey": _team_mean(madqn.agent_avg_dist_deque_dict, 0, n1),
                        "steps/predator2_mean_avg_distance_prey": _team_mean(madqn.agent_avg_dist_deque_dict, n1, n2),
                        "steps/predator1_mean_shortest_distance_prey": _team_mean(madqn.agent_min_dist_deque_dict, 0, n1),
                        "steps/predator2_mean_shortest_distance_prey": _team_mean(madqn.agent_min_dist_deque_dict, n1, n2),
                        "steps/predator1_mean_n_prey_observation": _team_mean(madqn.prey_number_deque_dict, 0, n1),
                        "steps/predator2_mean_n_prey_observation": _team_mean(madqn.prey_number_deque_dict, n1, n2),
                        "steps/predator1_mean_total_outtake": _team_mean(madqn.l2_outtake_deque_dict, 0, n1),
                        "steps/predator2_mean_total_outtake": _team_mean(madqn.l2_outtake_deque_dict, n1, n2),
                        "steps/predator1_mean_total_intake": _team_mean(madqn.l2_intake_deque_dict, 0, n1),
                        "steps/predator2_mean_total_intake": _team_mean(madqn.l2_intake_deque_dict, n1, n2),
                        "steps/predator1_mean_outtake_ratio": _team_mean(madqn.outtake_ratio_deque_dict, 0, n1),
                        "steps/predator2_mean_outtake_ratio": _team_mean(madqn.outtake_ratio_deque_dict, n1, n2),
                        "steps/predator1_gate_mean": _team_mean(madqn.gate_mean_deque_dict, 0, n1),
                        "steps/predator2_gate_mean": _team_mean(madqn.gate_mean_deque_dict, n1, n2),
                        "steps/predator1_mean_overlap_tiles_pred1": _team_mean(madqn.tiles_number_with_pred1_deque_dict, 0, n1),
                        "steps/predator2_mean_overlap_tiles_pred1": _team_mean(madqn.tiles_number_with_pred1_deque_dict, n1, n2),
                        "steps/predator1_mean_overlap_tiles_pred2": _team_mean(madqn.tiles_number_with_pred2_deque_dict, 0, n1),
                        "steps/predator2_mean_overlap_tiles_pred2": _team_mean(madqn.tiles_number_with_pred2_deque_dict, n1, n2),
                        "steps/predator1_mean_overlap_tiles_ratio_pred1": _team_mean_tiles_ratio(madqn.tiles_number_with_pred1_deque_dict, 0, n1),
                        "steps/predator2_mean_overlap_tiles_ratio_pred1": _team_mean_tiles_ratio(madqn.tiles_number_with_pred1_deque_dict, n1, n2),
                        "steps/predator1_mean_overlap_tiles_ratio_pred2": _team_mean_tiles_ratio(madqn.tiles_number_with_pred2_deque_dict, 0, n1),
                        "steps/predator2_mean_overlap_tiles_ratio_pred2": _team_mean_tiles_ratio(madqn.tiles_number_with_pred2_deque_dict, n1, n2),
                        "steps/predator1_mean_action": _team_mean(madqn.agent_action_deque_dict, 0, n1),
                        "steps/predator2_mean_action": _team_mean(madqn.agent_action_deque_dict, n1, n2),
                    }

                    metrics["steps/step"] = int(prev_step)
                    metrics.update(team_metrics)
                    wandb.log(metrics)


            if agent[:8] == "predator":

                handles = env.env.env.env.env.get_handles()
                pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                pos_predator2 = env.env.env.env.env.get_pos(handles[1])

                observation, reward, termination, truncation, _ = env.last()

                # for predator 1
                if agent[9] == "1": # predator predator1_0, predator1_1 .. etc
                    idx = int(agent[11:])
                    pos = pos_predator1[idx]
                    view_range = predator1_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_1(observation)
                    madqn.set_team_idx(0)

                    dist_list = np.array([np.mean(madqn.dist(observation_temp))], dtype=float)
                    # print(dist_list)
                    madqn.concat_dist(dist_list)

                    overlap_tiles_pred1, overlap_tiles_pred2 = madqn.coor_list_pred1(self_pos=pos,
                                                                               pred1_positions=pos_predator1,
                                                                               pred2_positions=pos_predator2)

                # for predator 2
                else:
                    idx = int(agent[11:]) + n_predator1
                    pos = pos_predator2[idx - n_predator1]
                    view_range = predator2_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_2(observation)
                    madqn.set_team_idx(1)

                    dist_list = np.array([np.mean(madqn.dist(observation_temp))], dtype=float)
                    madqn.concat_dist(dist_list)


                    overlap_tiles_pred1, overlap_tiles_pred2 = madqn.coor_list_pred2(self_pos=pos,
                                                                              pred1_positions=pos_predator1,
                                                                              pred2_positions=pos_predator2)

                madqn.set_agent_info(agent, pos, view_range)

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    n_iteration += 1
                    continue

                else: # training steps
                    action, book, shared_info, l2_before, l2_outtake, shared_sum, l2_intake, after_gnn = madqn.get_action(
                        state=observation_temp, mask=None)
                    env.step(action)

                    # intake_sum: overlap_tiles_pred1
                    intake_sum_with_pred1 = madqn.intake_sum(book, after_gnn,
                                                       overlap_tiles_pred1)  
                    intake_sum_with_pred2 = madqn.intake_sum(book, after_gnn,
                                                       overlap_tiles_pred2)  
                    val1 = _to_float(intake_sum_with_pred1)
                    val2 = _to_float(intake_sum_with_pred2)
                    madqn.intake_sum_with_pred1_deque_dict[idx].append(val1)
                    madqn.intake_sum_with_pred2_deque_dict[idx].append(val2)

                    intake_inner_with_pred1 = madqn.intake_inner(book, after_gnn,
                                                           overlap_tiles_pred1)  
                    intake_inner_with_pred2 = madqn.intake_inner(book, after_gnn,
                                                           overlap_tiles_pred2)  
                    inner1 = _to_float(intake_inner_with_pred1)
                    inner2 = _to_float(intake_inner_with_pred2)
                    madqn.intake_inner_with_pred1_deque_dict[idx].append(inner1)
                    madqn.intake_inner_with_pred2_deque_dict[idx].append(inner2)

                    madqn.tiles_number_with_pred1_deque_dict[idx].append(
                        len(overlap_tiles_pred1))  
                    madqn.tiles_number_with_pred2_deque_dict[idx].append(
                        len(overlap_tiles_pred2))  

                    avg_dist = madqn.avg_dist(observation_temp)  
                    min_dist = madqn.min_dist(observation_temp)  
                    madqn.agent_avg_dist_deque_dict[idx].append(avg_dist)  
                    madqn.agent_min_dist_deque_dict[idx].append(min_dist)  

                    number = madqn.prey_number(observation_temp)  
                    madqn.prey_number_deque_dict[idx].append(number)  

                    madqn.shared_mean_deque_dict[idx].append(_to_float(th.mean(shared_info)))
                    madqn.l2_before_outtake_deque_dict[idx].append(_to_float(l2_before))
                    madqn.l2_outtake_deque_dict[idx].append(_to_float(l2_outtake))
                    madqn.l2_intake_deque_dict[idx].append(_to_float(l2_intake))

                    # move
                    if action in [0, 1, 3, 4]: 
                        # predator1
                        if idx < n_predator1:
                            move_penalty_dict[idx].append(args.move_penalty)
                        else:
                            move_penalty_dict[idx].append(0)
                        madqn.ep_move_count()
                        madqn.step_move_count()
                        move = 1

                    # tag
                    elif action in [5, 6, 7, 8, 9, 10, 11, 12]:
                        move_penalty_dict[idx].append(0)
                        madqn.step_tag_count()
                        move = 2

                    # stay
                    else:
                        move_penalty_dict[idx].append(0)
                        move = 0

                    # accumulate per-step action-type counts (team-wise)
                    team_for_ratio = 0 if idx < n_predator1 else 1
                    step_action_type_count[team_for_ratio][move] += 1

                    madqn.agent_action_deque_dict[idx].append(move)

                    reward = env._cumulative_rewards[agent]  # cumulative reward or iteration reward?

                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward)

                    book_cpu = book.detach().cpu()
                    shared_info_cpu = shared_info.detach().cpu()

                    book_dict[idx].append(book_cpu)
                    shared_info_dict[idx].append(shared_info_cpu)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)
                    agent_pos[idx].append(pos)

                if madqn.buffer.size() >= args.trainstart_buffersize:
                    madqn.replay()


            else:  # prey
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    n_iteration += 1
                    continue

                else:
                    action = env.action_space(agent).sample()
                    env.step(action)

            n_iteration += 1

            # intake calculation
            if (((n_iteration) % n_agents) == 0): # when 1 step ends

                completed_step_idx = n_iteration // n_agents

                handles = env.env.env.env.env.get_handles()
                pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                pos_predator2 = env.env.env.env.env.get_pos(handles[1])
                entire_pos_list = np.concatenate((pos_predator1, pos_predator2))
                entire_pos.append(entire_pos_list)

                if completed_step_idx == 1:
                    if len(entire_pos) >= 2:
                        past = entire_pos[-2]
                        now = entire_pos[-1]
                        ratio_matrix = madqn.calculate_Overlap_ratio_intake(past, now)

                        for idx, value in enumerate(ratio_matrix[:, 0]):
                            madqn.intake_overlap_with_pred1[idx].append(value)

                        for idx, value in enumerate(ratio_matrix[:, 1]):
                            madqn.intake_overlap_with_pred2[idx].append(value)
                    else:
                        for idx in range(n_predator1 + n_predator2):
                            madqn.intake_overlap_with_pred1[idx].append(0)
                            madqn.intake_overlap_with_pred2[idx].append(0)


                elif completed_step_idx != args.max_update_steps and len(entire_pos) >= 2:
                    past = entire_pos[-2]  # s_t-1
                    now = entire_pos[-1]  # s_t

                    ratio_matrix = madqn.calculate_Overlap_ratio_intake(past, now)

                    for idx, value in enumerate(ratio_matrix[:, 0]):
                        madqn.intake_overlap_with_pred1[idx].append(value)

                    for idx, value in enumerate(ratio_matrix[:, 1]):
                        madqn.intake_overlap_with_pred2[idx].append(value)

                else:
                    pass

        pred1_move = max(madqn.ep_move_count_pred[0], 1) # predator 1
        pred2_move = max(madqn.ep_move_count_pred[1], 1) # predator 2

        episode_log = {
            "episode/episode": ep,
            "episode/total_reward": ep_reward,
            "episode/predator1_reward": ep_reward_pred1,
            "episode/predator2_reward": ep_reward_pred2,
            "episode/total_reward_np": ep_reward_np,
            "episode/predator1_reward_np": ep_reward_pred1_np,
            "episode/predator2_reward_np": ep_reward_pred2_np,
            "episode/predator1_move_count": madqn.ep_move_count_pred[0],
            "episode/predator2_move_count": madqn.ep_move_count_pred[1],
            "episode/predator1_reward_over_move": ep_reward_pred1 / pred1_move,
            "episode/predator2_reward_over_move": ep_reward_pred2 / pred2_move,
        }
        wandb.log(episode_log)

        observations_dict.clear()
        reward_dict.clear()
        move_penalty_dict.clear()
        action_dict.clear()
        termination_dict.clear()
        truncation_dict.clear()
        book_dict.clear()
        shared_info_dict.clear()
        agent_pos.clear()
        entire_pos.clear()

        if ep > args.total_ep:  # 30
            print('*' * 10, 'train over', '*' * 10)
            print(n_iteration)
            break

        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):
                madqn.set_agent_model(agent)
                madqn.target_update()

        if (ep % args.ep_save) == 0:
            for i in range(len(madqn.gdqns)):
                path = 'model_save/' + 'model_' + str(i) + '_ep' + str(ep) + '.pt'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                th.save(madqn.gdqns[i].state_dict(), path)

                path_t = 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep) + '.pt'
                os.makedirs(os.path.dirname(path_t), exist_ok=True)
                th.save(madqn.gdqn_targets[i].state_dict(), path_t)

    env.close()

    print('*' * 10, 'train over', '*' * 10)

if __name__ == '__main__':
    main()

    print('done')


