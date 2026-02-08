# parse args locally so this script can be run standalone
from magent2_backup.environments import hetero_adversarial_v1
from MADQN_decen import MADQN
import argparse
import numpy as np
import torch as th
import wandb
import os


MOVE_ACTIONS = {0, 1, 3, 4}


def last(env):
    observation, reward, termination, truncation, info = env.last()
    if isinstance(observation, np.ndarray):
        observation = observation.copy()
    return observation, reward, termination, truncation, info



def get_args():
    parser = argparse.ArgumentParser(description='MADQN_decen')
    parser.add_argument('--dim_feature', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--eps_decay', type=float, default=0.9999)
    parser.add_argument('--eps_min', type=float, default=0.01)
    parser.add_argument('--max_update_steps', type=int, default=3000)
    parser.add_argument('--total_ep', type=int, default=7)
    parser.add_argument('--book_decay', type=float, default=0.1)
    parser.add_argument('--book_term', type=int, default=4)
    parser.add_argument('--ep_save', type=int, default=1000)
    parser.add_argument('--jitter_std', type=float, default=0.5)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--trainstart_buffersize', type=int, default=6000)
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
    parser.add_argument('--seed', type=int, default=125)

    return parser.parse_args()


args = get_args()

device = th.device("cuda" if th.cuda.is_available() else "cpu")
wandb.init(project="MADQN_01", entity='hails',config=args.__dict__)
wandb.run.name = 'model_decen'

render_mode = 'rgb_array'

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey
dim_feature = args.dim_feature


predator1_obs = (predator1_view_range * 2, predator1_view_range * 2, dim_feature)
predator2_obs = (predator2_view_range * 2, predator2_view_range * 2, dim_feature)
dim_act = 13

predator1_adj = ((predator1_view_range*2)**2, (predator1_view_range*2)**2)
predator2_adj = ((predator2_view_range*2)**2, (predator2_view_range*2)**2)

batch_size = 1

# target_update_point = (1+args.max_update_steps)*(args.n_predator1+args.n_predator2+args.n_prey)


madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, args.buffer_size, device, args=args)

def process_array_1(arr):  #predator1 (obs, team, team hp, predator2, predator2 hp, prey, prey hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]))

    return result


def process_array_2(arr): #predator2 (obs, team, team hp, prey, prey hp, predator2, predator2 hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 3], arr[:, :, 2]))
    
    return result


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

        # env reset
        env.reset(seed=args.seed)

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0

        # episode reward without move penalty
        ep_reward_np = 0
        ep_reward_pred1_np = 0
        ep_reward_pred2_np = 0

        # ep_move_count_pred1 = 0
        # ep_move_count_pred2 = 0

        n_iteration = 0

        observations_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            observations_dict[agent_idx] = []

        reward_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            reward_dict[agent_idx] = []

        action_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            action_dict[agent_idx] = []

        termination_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            termination_dict[agent_idx] = []

        truncation_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            truncation_dict[agent_idx] = []

        move_penalty_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            move_penalty_dict[agent_idx] = []

        reward_np_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            reward_np_dict[agent_idx] = []


        print("ep:",ep,'*' * 80)

        for agent in env.agent_iter():

            n_agents = (args.n_predator1 + args.n_predator2 + args.n_prey)
            step_idx_ep = n_iteration // n_agents

            if (n_agents > 0) and ((n_iteration % n_agents) == 0) and (step_idx_ep > 0):
                prev_step = step_idx_ep - 1

                step_reward_pred1 = 0.0
                step_reward_pred2 = 0.0
                step_reward_pred1_np = 0.0
                step_reward_pred2_np = 0.0
                step_move_pred1 = 0
                step_move_pred2 = 0

                for i in range(n_predator1):
                    if reward_dict[i]:
                        step_reward_pred1 += float(reward_dict[i][-1])
                        step_reward_pred1_np += float(reward_np_dict[i][-1])
                        if action_dict[i] and action_dict[i][-1] in MOVE_ACTIONS:
                            step_move_pred1 += 1
                for i in range(n_predator1, n_predator1 + n_predator2):
                    if reward_dict[i]:
                        step_reward_pred2 += float(reward_dict[i][-1])
                        step_reward_pred2_np += float(reward_np_dict[i][-1])
                        if action_dict[i] and action_dict[i][-1] in MOVE_ACTIONS:
                            step_move_pred2 += 1

                metrics = {
                    "steps/step": int(prev_step),
                    "predator1/step_reward": float(step_reward_pred1),
                    "predator2/step_reward": float(step_reward_pred2),
                    "predator1/step_reward_np": float(step_reward_pred1_np),
                    "predator2/step_reward_np": float(step_reward_pred2_np),
                    "predator1/move_count": int(step_move_pred1),
                    "predator2/move_count": int(step_move_pred2),
                    "steps/total_step_reward": float(step_reward_pred1 + step_reward_pred2),
                    "steps/total_step_reward_np": float(step_reward_pred1_np + step_reward_pred2_np),
                }
                wandb.log(metrics)

            if agent[:8] == "predator": # dont need pos info?
                # handles = env.env.env.env.env.get_handles()
                # pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                # pos_predator2 = env.env.env.env.env.get_pos(handles[1])
                observation, reward, termination, truncation, info = last(env)

                if agent[9] == "1":
                    idx = int(agent[11:])
                    observation_temp = process_array_1(observation)
                else:
                    idx = int(agent[11:]) + n_predator1
                    observation_temp = process_array_2(observation)

                madqn.set_agent_info(agent)

                if termination or truncation:
                    print(agent , 'is terminated')
                    env.step(None)
                    continue

                else:
                    action = madqn.get_action(state=observation_temp, mask=None)

                    env.step(action)
                    base_reward = float(env._cumulative_rewards[agent])
                    move_penalty = float(args.move_penalty) if ((idx < n_predator1) and (action in MOVE_ACTIONS)) else 0.0
                    reward_for_train = base_reward + move_penalty

                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward_for_train)
                    reward_np_dict[idx].append(base_reward)
                    move_penalty_dict[idx].append(move_penalty)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)

                    ep_reward += reward_for_train
                    ep_reward_np += base_reward
                    if idx < n_predator1:
                        ep_reward_pred1 += reward_for_train
                        ep_reward_pred1_np += base_reward
                        if action in MOVE_ACTIONS:
                            ep_move_count_pred1 += 1
                    else:
                        ep_reward_pred2 += reward_for_train
                        ep_reward_pred2_np += base_reward
                        if action in MOVE_ACTIONS:
                            ep_move_count_pred2 += 1

                    if len(observations_dict[idx]) >= 2:
                        madqn.buffer.put(
                            observations_dict[idx][-2],
                            action_dict[idx][-2],
                            reward_dict[idx][-2],
                            observations_dict[idx][-1],
                            termination_dict[idx][-1],
                            truncation_dict[idx][-1],
                        )



                if madqn.buffer.size() >= args.trainstart_buffersize:
                    madqn.replay()


            # prey
            elif agent[:4] == "prey":
                observation, reward, termination, truncation, info = last(env)
                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                else:
                    prey_action = env.action_space(agent).sample()
                    env.step(prey_action)

            n_iteration += 1

        if ep > args.total_ep: #100
            print('*' * 10, 'train over', '*' * 10)
            print(n_iteration)
            break


        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            madqn.target_update()

        pred1_move = max(ep_move_count_pred1, 1)
        pred2_move = max(ep_move_count_pred2, 1)
        episode_log = {
            "episode/episode": int(ep),
            "episode/total_reward": float(ep_reward),
            "episode/total_reward_np": float(ep_reward_np),
            "episode/predator1_reward": float(ep_reward_pred1),
            "episode/predator2_reward": float(ep_reward_pred2),
            "episode/predator1_reward_np": float(ep_reward_pred1_np),
            "episode/predator2_reward_np": float(ep_reward_pred2_np),
            "episode/predator1_move_count": int(ep_move_count_pred1),
            "episode/predator2_move_count": int(ep_move_count_pred2),
            "episode/predator1_reward_over_move": float(ep_reward_pred1 / pred1_move),
            "episode/predator2_reward_over_move": float(ep_reward_pred2 / pred2_move),
        }
        wandb.log(episode_log)


        if (ep % args.ep_save) ==0 :
            for i in range(len(madqn.gdqns)) :
                path = 'model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                th.save(madqn.gdqns[i].state_dict(), path)

                path_t = 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt'
                os.makedirs(os.path.dirname(path_t), exist_ok=True)
                th.save(madqn.gdqn_targets[i].state_dict(), path_t)


        env.close()

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    # for i in range(len(madqn.gdqns)) :
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')

    print('done')


