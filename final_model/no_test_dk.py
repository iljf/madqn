from magent2_backup.environments import hetero_adversarial_v1
from MADQN import MADQN

import numpy as np
import torch as th
import argparse
import wandb
import os


def get_args():
    parser = argparse.ArgumentParser(description='MADQN')

    parser.add_argument('--dim_feature', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--eps_decay', type=float, default=0.9999)
    parser.add_argument('--eps_min', type=float, default=0.01)
    parser.add_argument('--max_update_steps', type=int, default=3000)
    parser.add_argument('--total_ep', type=int, default=100)
    parser.add_argument('--book_decay', type=float, default=0.1)
    parser.add_argument("--book_term", type=int, default=4)
    parser.add_argument('--ep_save', type=int, default=1000)
    parser.add_argument('--jitter_std', type=float, default=0.5)

    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--trainstart_buffersize', type=int, default=9000)
    parser.add_argument('--deque_len', type=int, default=400)
    parser.add_argument('--plot_term', type=int, default=10)

    parser.add_argument('--replay_times', type=int, default=32)
    parser.add_argument('--target_update', type=int, default=10)

    parser.add_argument('--map_size', type=int, default=24)
    parser.add_argument('--predator1_view_range', type=int, default=10)
    parser.add_argument('--predator2_view_range', type=int, default=5)
    parser.add_argument('--n_predator1', type=int, default=9)
    parser.add_argument('--n_predator2', type=int, default=9)
    parser.add_argument('--n_prey', type=int, default=36)
    parser.add_argument('--tag_reward', type=float, default=3)
    parser.add_argument('--tag_penalty', type=float, default=-0.2)
    parser.add_argument('--move_penalty', type=float, default=-0.15)

    parser.add_argument('--seed', type=int, default=874)

    return parser.parse_args()

args = get_args()
wandb.init(project="MADQN_01", entity='hails',config=args.__dict__)
wandb.run.name = 'p1:9_p2:9_36'
# define a custom step metric so we can log our own step values (monotonic in UI)
try:
    wandb.define_metric("steps/step", step_metric=True)
except Exception:
    pass

# Wrap wandb.log to automatically include our custom step metric when available
_wandb_log = wandb.log
def _wb_log_auto_step(d, **kwargs):
    try:
        if isinstance(d, dict) and "steps/step" not in d:
            # attempt to add current step index if defined
            d["steps/step"] = step_idx_ep
    except Exception:
        pass
    return _wandb_log(d, **kwargs)
wandb.log = _wb_log_auto_step


device = th.device("cuda" if th.cuda.is_available() else "cpu")

render_mode = 'rgb_array'
# render_mode = 'human'

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

        # reset ep_move_count : ? ?????? Plot? ??? ?? ?? action? ???? ????? ???? ??
        madqn.reset_ep_move_count()

        # env reset
        env.reset(seed=args.seed)

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0

        n_iteration = 0

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

        print("ep:", ep, '*' * 80)

        for agent in env.agent_iter():

            n_agents = (args.n_predator1 + args.n_predator2 + args.n_prey)
            step_idx_ep = n_iteration // n_agents


            if (((n_iteration) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0) and (n_agents > 0):

                if step_idx_ep > 0:
                    madqn.shared_decay() 

                    if step_idx_ep != args.max_update_steps: # shared_book update

                        if step_idx_ep <= args.book_term:

                            for idx in range(n_predator1 + n_predator2):


                                madqn.set_agent_pos(agent_pos[idx][-1]) # agent position

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

                        else:
                            # erase last step book information
                            for idx in range(n_predator1 + n_predator2):

                                madqn.set_agent_pos(agent_pos[idx][-(args.book_term + 1)])

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                # self.to_guestbook(shared_info.to('cpu'))
                                madqn.to_guestbook(-(args.book_decay ** (args.book_term)) * shared_info_dict[idx][
                                    -(args.book_term + 1)].to('cpu'))

                            # Add recent Step information
                            for idx in range(n_predator1 + n_predator2):

                                madqn.set_agent_pos(agent_pos[idx][-1])

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))


                        madqn.avg_dist_append_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))  # step
                        madqn.min_dist_append_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))  # step
                        madqn.avg_move_append_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)  # step
                        madqn.avg_tag_append_pred1((madqn.step_tag_count_pred[0]) / n_predator1)  # step

                        madqn.avg_dist_append_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))  # step
                        madqn.min_dist_append_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))  # step
                        madqn.avg_move_append_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)  # step
                        madqn.avg_tag_append_pred2((madqn.step_tag_count_pred[1]) / n_predator2)  # step

                        # capture step-level move/tag counts for batching into later metrics
                        last_move_count_pred0 = madqn.step_move_count_pred[0]
                        last_move_count_pred1 = madqn.step_move_count_pred[1]
                        last_tag_count_pred0 = madqn.step_tag_count_pred[0]
                        last_tag_count_pred1 = madqn.step_tag_count_pred[1]

                        # If this is the first step, log them immediately (no later batch)
                        if step_idx_ep == 1:
                            try:
                                wandb.log({
                                    "predator1/move_count": last_move_count_pred0,
                                    "predator2/move_count": last_move_count_pred1,
                                    "predator1/tag_count": last_tag_count_pred0,
                                    "predator2/tag_count": last_tag_count_pred1,
                                    "steps/step": step_idx_ep,
                                })
                            except Exception:
                                pass

                        madqn.reset_step_move_count()
                        madqn.reset_step_tag_count()
                        madqn.reset_summation_team_dist()

                        # out take? ??? plot(==case2)? ?? ?? ??? ??? ???? ??
                        pos_list = np.concatenate((pos_predator1, pos_predator2), axis=0)
                        # ? ?????? preator1? predator2? ??? ??? ??? ??? ?? -> ???: predator1? ??? ?? , ????: predator2? ??? ??
                        ratio_matrix = madqn.calculate_Overlap_ratio(pos_list)

                        #? ????? ?? Observation ?? ?? predator1? ??? ??
                        for idx, value in enumerate(ratio_matrix[:, 0]):
                            madqn.agent_graph_overlap_pred1_deque_dict[idx].append(value)

                        #? ????? ?? Observation ?? ?? predator2? ??? ??
                        # a? ? ?? ?? ??? self.agent_graph_overlap_pred2_deque_dict? ??
                        for idx, value in enumerate(ratio_matrix[:, 1]):
                            madqn.agent_graph_overlap_pred2_deque_dict[idx].append(value)

                    # else:
                    #     madqn.avg_dist_append_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))
                    #     madqn.min_dist_append_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))
                    #     madqn.avg_move_append_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)
                    #     madqn.avg_tag_append_pred1((madqn.step_tag_count_pred[0]) / n_predator1)
                    #
                    #     # ??? ?? predator2 ? avg(distance) ,min(distance) ? avg(count)??? ??? ??
                    #     madqn.avg_dist_append_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))
                    #     madqn.min_dist_append_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))
                    #     madqn.avg_move_append_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)
                    #     madqn.avg_tag_append_pred2((madqn.step_tag_count_pred[1]) / n_predator2)
                    #
                    #     # ??? ??? ??????? ?? ??? ???? plotting ? ?? ??? ?? ??->?? ??
                    #     madqn.reset_step_move_count()  # ????? ?? ????? ??.
                    #     madqn.reset_step_tag_count()  # ????? ?? ????? ??.
                    #     madqn.reset_summation_team_dist()  # ??? ????? ??.
                    #
                    #     # out take case2 ????? ? ??
                    #     # pos_predator1 ? pos_predator2 ? ?? ???? ?????, ??? ?? ?? ???, ???? ???? ????.
                    #     pos_list = np.concatenate((pos_predator1, pos_predator2), axis=0)
                    #     ratio_matrix = madqn.calculate_Overlap_ratio(pos_list)
                    #     # ? ??? ??? ??.
                    #     for idx, value in enumerate(ratio_matrix[:, 0]):
                    #         madqn.agent_graph_overlap_pred1_deque_dict[idx].append(value)
                    #
                    #     # a? ? ?? ?? ??? self.agent_graph_overlap_pred2_deque_dict? ??
                    #     for idx, value in enumerate(ratio_matrix[:, 1]):
                    #         madqn.agent_graph_overlap_pred2_deque_dict[idx].append(value)

                    if step_idx_ep > 1:

                        step_rewards = 0
                        step_penalty_rewards = 0

                        # ? ??? reward? ???? ?? ??
                        step_reward_pred1 = 0
                        step_reward_pred2 = 0
                        step_penalty_pred1 = 0
                        step_penalty_pred2 = 0
                        step_rewards_pred1 = 0
                        step_rewards_pred2 = 0

                        ##########################################################################
                        # ?? ????? total reward ?? -> ?????? ?? reward ? ???? ???? ???? ??
                        ##########################################################################

                        for agent_rewards in reward_dict.values():
                            step_rewards += np.sum(agent_rewards[-1])

                        for penalty in move_penalty_dict.values():
                            step_penalty_rewards += np.sum(penalty[-2])

                        # ?? ? ??? ??? ? ???? ?? reward? ???? ??
                        step_rewards = step_rewards + step_penalty_rewards

                        # ep ? ?? ?? reward? ????? ??
                        ep_reward += step_rewards

                        ##########################################################################
                        # ???? reward ?? -> ?????? ?? reward ? ???? ???? ???? ??
                        ##########################################################################

                        # ? ??? ??????? reward ??
                        for i, agent_rewards in enumerate(reward_dict.values()):
                            if i < len(reward_dict) // 2:
                                step_reward_pred1 += np.sum(agent_rewards[-1])
                            else:
                                step_reward_pred2 += np.sum(agent_rewards[-1])

                        # ? ??? ??????? reward ??
                        for i, agent_penalty in enumerate(move_penalty_dict.values()):
                            if i < len(reward_dict) // 2:
                                step_penalty_pred1 += np.sum(agent_penalty[-1])
                            else:
                                step_penalty_pred2 += np.sum(agent_penalty[-1])

                        step_rewards_pred1 = step_reward_pred1 + step_penalty_pred1
                        step_rewards_pred2 = step_reward_pred2 + step_penalty_pred2

                        ep_reward_pred1 += step_rewards_pred1
                        ep_reward_pred2 += step_rewards_pred2

                        for idx in range(n_predator1 + n_predator2):
                            madqn.set_agent_buffer(idx)

                            madqn.buffer.put(observations_dict[idx][-2],
                                             book_dict[idx][-2],
                                             action_dict[idx][-2],
                                             step_rewards,
                                             observations_dict[idx][-1],
                                             book_dict[idx][-1],
                                             termination_dict[idx][-2],
                                             truncation_dict[idx][-2])

                        metrics = {
                            "steps/total_step_reward": step_rewards,
                            "predator1/step_reward": step_rewards_pred1,
                            "predator2/step_reward": step_rewards_pred2,
                            "predator1/avg_tag_count": madqn.avg_tag_deque_pred1[-1],
                            "predator2/avg_tag_count": madqn.avg_tag_deque_pred2[-1],
                            "predator1/avg_move_count": madqn.avg_move_deque_pred1[-1],
                            "predator2/avg_move_count": madqn.avg_move_deque_pred2[-1],
                            "steps/step": step_idx_ep,
                        }

                        # include the last step move/tag counts if available (fallback to current counts)
                        try:
                            metrics["predator1/move_count"] = last_move_count_pred0
                            metrics["predator2/move_count"] = last_move_count_pred1
                            metrics["predator1/tag_count"] = last_tag_count_pred0
                            metrics["predator2/tag_count"] = last_tag_count_pred1
                        except Exception:
                            metrics["predator1/move_count"] = madqn.step_move_count_pred[0]
                            metrics["predator2/move_count"] = madqn.step_move_count_pred[1]
                            metrics["predator1/tag_count"] = madqn.step_tag_count_pred[0]
                            metrics["predator2/tag_count"] = madqn.step_tag_count_pred[1]

                        ############################################
                        ####################???####################
                        ############################################


                        #########################
                        #######Y?? ?? ??######
                        #########################

                        # ? ????? ?? Observation ?? ?? predator1? ??? ??
                        for idx, value in madqn.agent_graph_overlap_pred1_deque_dict.items():
                            if len(value) == 0:
                                continue
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator1
                                metrics[f"predator1/Overlap_ratio_predator1_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator1
                                metrics[f"predator2/Overlap_ratio_predator1_P2_{pred2_idx}"] = last_value

                        # ? ????? ?? Observation ?? ?? predator2? ??? ??
                        for idx, value in madqn.agent_graph_overlap_pred2_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator2
                                metrics[f"predator1/Overlap_ratio_predator2_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator2
                                metrics[f"predator2/Overlap_ratio_predator2_P2_{pred2_idx}"] = last_value

                        # team1? view? ??? ??? intake? ??? ????? ????? ??
                        for idx, value in madqn.intake_sum_with_pred1_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator1
                                metrics[f"predator1/Sum_intake_diff_predator1_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator1
                                metrics[f"predator2/Sum_intake_diff_predator1_P2_{pred2_idx}"] = last_value

                        # team2? view? ??? ??? intake? ??? ????? ????? ??
                        for idx, value in madqn.intake_sum_with_pred2_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator2
                                metrics[f"predator1/Sum_intake_diff_predator2_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator2
                                metrics[f"predator2/Sum_intake_diff_predator2_P2_{pred2_idx}"] = last_value

                        # team1? view? ??? ??? intake? ??? ????? ??? ???? ??
                        for idx, value in madqn.intake_inner_with_pred1_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator1
                                metrics[f"predator1/inner_intake_diff_predator1_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator1
                                metrics[f"predator2/inner_intake_diff_predator1_P2_{pred2_idx}"] = last_value

                        # team2? view? ??? ??? intake? ??? ????? ??? ???? ??
                        for idx, value in madqn.intake_inner_with_pred2_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1: # predator1 agents with predator2
                                metrics[f"predator1/inner_intake_diff_predator2_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1 # predator2 agents with predator2
                                metrics[f"predator2/inner_intake_diff_predator2_P2_{pred2_idx}"] = last_value

                        # # ??? observation ? GNN ? ?? ???? ?? observation ? L2 ?
                        # for idx, value in enumerate(madqn.l2_before_outtake_deque_dict):
                        #     # ? deque? ??? ?? ??
                        #     last_value = value[-1]
                        #     wandb.log(
                        #         {f"Total before outtake_{idx}": last_value, "step": iteration_number})

                        # ?? ?? ??? view ? ???? ?? ?? outtake ??? ???
                        for idx, value in madqn.l2_outtake_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/total_outtake_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/total_outtake_P2_{pred2_idx}"] = last_value

                        # ?? ?? ??? view ? ???? ?? ?? intake ??? ???
                        for idx, value in madqn.l2_intake_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/total_intake_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/total_intake_P2_{pred2_idx}"] = last_value


                        # ?  ????? action
                        for idx, dq in madqn.agent_action_deque_dict.items():
                            if len(dq) == 0:
                                continue
                            last_value = dq[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/action_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/action_P2_{pred2_idx}"] = last_value


                        #########################
                        #######Y?? ?? ??######
                        #########################

                        # ?? ??? ??? ??? ??? ???? ?? ??? ??? ???, ??? ??? ?? ????? ??? ?? '??? ?'? ???? ?
                        # team1 ? ??? ?? ?
                        for idx, value in madqn.tiles_number_with_pred1_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/n_overlapping_tiles_predator1_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/n_overlapping_tiles_predator1_P2_{pred2_idx}"] = last_value


                        # team2 ? ??? ?? ?
                        for idx, value in madqn.tiles_number_with_pred2_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/n_overlapping_tiles_predator2_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/n_overlapping_tiles_predator2_P2_{pred2_idx}"] = last_value

                        # prey??? ?? ??
                        for idx, value in madqn.agent_avg_dist_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/avg_distance_prey_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/avg_distance_prey_P2_{pred2_idx}"] = last_value

                        # prey??? ?? ??
                        for idx, value in madqn.agent_min_dist_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/shortest_distance_prey_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/shortest_distance_prey_P2_{pred2_idx}"] = last_value

                        # ?  ???? ?? ???? prey ?
                        for idx, value in madqn.prey_number_deque_dict.items():
                            # ? deque? ??? ?? ??
                            last_value = value[-1]

                            if idx < n_predator1:
                                metrics[f"predator1/n_prey_observation_P1_{idx}"] = last_value
                            else:
                                pred2_idx = idx - n_predator1
                                metrics[f"predator2/n_prey_observation_P2_{pred2_idx}"] = last_value

                        # log all batched metrics once for this step
                        try:
                            wandb.log(metrics)
                        except Exception:
                            pass

            if agent[:8] == "predator":

                # for each step ( doesn't change until all predators move )
                handles = env.env.env.env.env.get_handles()

                pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                pos_predator2 = env.env.env.env.env.get_pos(handles[1])

                entire_pos_list = np.concatenate((pos_predator1, pos_predator2))

                observation, reward, termination, truncation, info = env.last()

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

                else:
                    # action : action ?
                    # book : ?? ??? ???? shared graph ?? ??? ??
                    # shared_info : ????? observation? gnn? ?? shared graph? ???? ? ???
                    # l2_before: ??? observation ? Norm ?
                    # l2_outtake : ??? Observation ? out take ??? GNN ?? ?? ??? ??????? ??? ??? ?
                    # shared_sum :  shared_info ? summation ?
                    # l2_intake :  ? ????? shared graph ?? ??? ??? ??? ????? ??? ??? ?
                    # after_gnn : shared graph ??? ??? intake ??? gnn? ?? ?? ?
                    action, book, shared_info, l2_before, l2_outtake, shared_sum, l2_intake, after_gnn = madqn.get_action(
                        state=observation_temp, mask=None)
                    env.step(action)

                    # ??? intake
                    # ???? l2_intake, l2_outtake ?? ??? ?? ???? ?????, ?? ?? ????? ???? view_range? ???? ?? ??????.
                    # ??? ?? ?????? ?? ??? ??? ??? ??? ???(outtake), ??? ??? ?????(intake) ??? ??? ??.
                    # ??? ??? ??, intake_sum ? ?? ??? ???? ??? ????.

                    # intake_sum: overpal_tiles_pred1(??? observation? predator1? observation? ?? ??? ??? ??)? ???? book ? after_gnn ? ??? ??? ??? ??? ?
                    intake_sum_with_pred1 = madqn.intake_sum(book, after_gnn,
                                                       overlap_tiles_pred1)  # ????? predator1? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    intake_sum_with_pred2 = madqn.intake_sum(book, after_gnn,
                                                       overlap_tiles_pred2)  # ????? predator2? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    if isinstance(intake_sum_with_pred1, th.Tensor):
                        val1 = float(intake_sum_with_pred1.detach().cpu().item())
                    else:
                        val1 = float(intake_sum_with_pred1)
                    if isinstance(intake_sum_with_pred2, th.Tensor):
                        val2 = float(intake_sum_with_pred2.detach().cpu().item())
                    else:
                        val2 = float(intake_sum_with_pred2)
                    madqn.intake_sum_with_pred1_deque_dict[idx].append(val1)
                    madqn.intake_sum_with_pred2_deque_dict[idx].append(val2)

                    intake_inner_with_pred1 = madqn.intake_inner(book, after_gnn,
                                                           overlap_tiles_pred1)  # ????? predator1? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    intake_inner_with_pred2 = madqn.intake_inner(book, after_gnn,
                                                           overlap_tiles_pred2)  # ????? predator2? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    if isinstance(intake_inner_with_pred1, th.Tensor):
                        inner1 = float(intake_inner_with_pred1.detach().cpu().item())
                    else:
                        inner1 = float(intake_inner_with_pred1)
                    if isinstance(intake_inner_with_pred2, th.Tensor):
                        inner2 = float(intake_inner_with_pred2.detach().cpu().item())
                    else:
                        inner2 = float(intake_inner_with_pred2)
                    madqn.intake_inner_with_pred1_deque_dict[idx].append(inner1)
                    madqn.intake_inner_with_pred2_deque_dict[idx].append(inner2)

                    # ?? ??? ??? ??? ??? ???? ?? ??? ??? ???, ??? ??? ?? ????? ??? ?? '??? ?'? ???? ?
                    madqn.tiles_number_with_pred1_deque_dict[idx].append(
                        len(overlap_tiles_pred1))  # ????? predator1? ??? ??? ? ??
                    madqn.tiles_number_with_pred2_deque_dict[idx].append(
                        len(overlap_tiles_pred2))  # ????? predator2? ??? ??? ? ??

                    avg_dist = madqn.avg_dist(observation_temp)  # ?? observation ?? ?? prey??? ????
                    min_dist = madqn.min_dist(observation_temp)  # ?? observation ?? ?? prey??? ????
                    madqn.agent_avg_dist_deque_dict[idx].append(avg_dist)  # ??
                    madqn.agent_min_dist_deque_dict[idx].append(min_dist)  # ??

                    number = madqn.prey_number(observation_temp)  # ?? observation ?? ?? prey ? ?
                    madqn.prey_number_deque_dict[idx].append(number)  # ??

                    try:
                        shared_mean_val = float(th.mean(shared_info).detach().cpu().item())
                    except Exception:
                        shared_mean_val = float(th.mean(shared_info).item())
                    madqn.shared_mean_deque_dict[idx].append(shared_mean_val)

                    try:
                        madqn.l2_before_outtake_deque_dict[idx].append(float(l2_before.detach().cpu().item()))
                    except Exception:
                        madqn.l2_before_outtake_deque_dict[idx].append(float(l2_before.item()))

                    try:
                        madqn.l2_outtake_deque_dict[idx].append(float(l2_outtake.detach().cpu().item()))
                    except Exception:
                        madqn.l2_outtake_deque_dict[idx].append(float(l2_outtake.item()))

                    try:
                        madqn.l2_intake_deque_dict[idx].append(float(l2_intake.detach().cpu().item()))
                    except Exception:
                        madqn.l2_intake_deque_dict[idx].append(float(l2_intake.item()))

                    # move
                    if action in [0, 1, 3, 4]:
                        move_penalty_dict[idx].append(args.move_penalty)
                        madqn.ep_move_count()  # ep ? ?? move ? ??
                        madqn.step_move_count()  # step ??  ?? ??? ???? +1 ? ???.
                        move = 1

                    # tag
                    elif action in [5, 6, 7, 8, 9, 10, 11, 12]:
                        move_penalty_dict[idx].append(0)  # ???? ???? move_penalty? 0 ??? ??.
                        madqn.step_tag_count()  # ep ? ?? tag ? ??
                        move = 2

                    # stay
                    else:
                        move_penalty_dict[idx].append(0)  # ???? ???? move_penalty? 0 ??? ??.
                        move = 0

                    madqn.agent_action_deque_dict[idx].append(move)  # ? ????? ?? action ? ??? ????.

                    reward = env._cumulative_rewards[agent]  # agent

                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward)
                    try:
                        book_cpu = book.detach().cpu()
                    except Exception:
                        book_cpu = book

                    try:
                        shared_info_cpu = shared_info.detach().cpu()
                    except Exception:
                        shared_info_cpu = shared_info

                    book_dict[idx].append(book_cpu)
                    shared_info_dict[idx].append(shared_info_cpu)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)
                    agent_pos[idx].append(pos)
                    entire_pos.append(entire_pos_list)

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

            #  intake case1 ????? ? ??
            #  "? ????? ?? observation? ??"? "1step ??? ?? ?????? observation? ??? ??" ? ??? ???? ??
            # ?? ???? ? ???..!
            if (((n_iteration) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0):

                # ??? ????? shared graph ? ??? ?? ??? ??? ??? 0??? ? ? ??.
                if step_idx_ep == 1:
                    for idx in range(n_predator1 + n_predator2):
                        madqn.intake_overlap_with_pred1[idx].append(0)
                        madqn.intake_overlap_with_pred2[idx].append(0)


                # ???  ????? ??? ??? ?? ??? ??? ???? ??.
                elif step_idx_ep != args.max_update_steps:
                    past = entire_pos[-2]  # ?? ????? ? step ??? ????
                    now = entire_pos[-1]  # ?? ????? ?? ????

                    ratio_matrix = madqn.calculate_Overlap_ratio_intake(past, now)

                    # ????? ?? observation? ?? ? predator1 ? ??? ??? ??? ?? ???? ??
                    for idx, value in enumerate(ratio_matrix[:, 0]):
                        madqn.intake_overlap_with_pred1[idx].append(value)

                    # ????? ?? observation? ?? ? predator2 ? ??? ??? ??? ?? ???? ??
                    for idx, value in enumerate(ratio_matrix[:, 1]):
                        madqn.intake_overlap_with_pred2[idx].append(value)

                # ??? ??(? truncation step)??? ??? ??? ??.
                else:
                    pass

            step_idx_ep += 1


        if madqn.buffer.size() >= args.trainstart_buffersize:
            pred1_move = max(madqn.ep_move_count_pred[0], 1) # block Zerodivision
            pred2_move = max(madqn.ep_move_count_pred[1], 1)
            try:
                wandb.log({"episode/episode": ep,
                           "episode/total_reward": ep_reward,
                           "episode/predator1_reward": ep_reward_pred1,
                           "episode/predator2_reward": ep_reward_pred2,
                           "episode/predator1_move_count": madqn.ep_move_count_pred[0],
                           "episode/predator2_move_count": madqn.ep_move_count_pred[1],
                           "episode/predator1_reward_over_move": ep_reward_pred1 / pred1_move,
                           "episode/predator2_reward_over_move": ep_reward_pred2 / pred2_move,
                           "steps/step": step_idx_ep,
                           })
            except Exception:
                pass


        # === 에피소드 종료 후 메모리 누수 방지: dict/list 초기화 ===
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

        # wandb.plot
        # if (ep % args.plot_term == 0) and (ep > 0):
        #     madqn.plot(ep)

        if (ep % args.ep_save) == 0:
            for i in range(len(madqn.gdqns)):
                path = 'model_save/' + 'model_' + str(i) + '_ep' + str(ep) + '.pt'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                th.save(madqn.gdqns[i].state_dict(), path)

                path_t = 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep) + '.pt'
                os.makedirs(os.path.dirname(path_t), exist_ok=True)
                th.save(madqn.gdqn_targets[i].state_dict(), path_t)

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    print('done')


