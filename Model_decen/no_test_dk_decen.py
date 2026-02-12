# parse args locally so this script can be run standalone
from magent2_backup.environments import hetero_adversarial_v1
from MADQN_decen import MADQN
import argparse
import numpy as np
import torch as th
import wandb
import os

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
wandb.init(project="madqn_test", entity='hails',config=args.__dict__)
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

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0

        # episode reward without move penalty
        ep_reward_np = 0
        ep_reward_pred1_np = 0
        ep_reward_pred2_np = 0

        ep_move_count_pred1 = 0
        ep_move_count_pred2 = 0

        n_iteration = 0

        print("ep:",ep,'*' * 80)

        for agent in env.agent_iter():

            n_agents = (args.n_predator1 + args.n_predator2 + args.n_prey)
            step_idx_ep = n_iteration // n_agents

            if ((n_iteration % n_agents) == 0) and (n_agents > 0):
                if step_idx_ep > 0:
                    prev_step = step_idx_ep - 1  # logging based on previous step

                    step_rewards = 0.0
                    step_penalty_rewards = 0.0

                    step_reward_pred1 = 0.0
                    step_reward_pred2 = 0.0
                    step_penalty_pred1 = 0.0
                    step_penalty_pred2 = 0.0

                    # transition reward for replay buffer
                    transition_rewards = 0.0
                    transition_penalty_rewards = 0.0

                    for idx in range(n_predator1 + n_predator2):
                        if len(reward_dict[idx]) >= 1:
                            r = float(np.sum(reward_dict[idx][-1]))
                            step_rewards += r
                            if idx < n_predator1:
                                step_reward_pred1 += r
                            else:
                                step_reward_pred2 += r

                        if len(move_penalty_dict[idx]) >= 1:
                            p = float(np.sum(move_penalty_dict[idx][-1]))
                            step_penalty_rewards += p
                            if idx < n_predator1:
                                step_penalty_pred1 += p
                            else:
                                step_penalty_pred2 += p

                        if len(reward_dict[idx]) >= 2:
                            transition_rewards += float(np.sum(reward_dict[idx][-2]))

                        if len(move_penalty_dict[idx]) >= 2:
                            transition_penalty_rewards += float(np.sum(move_penalty_dict[idx][-2]))

                    transition_rewards_total = transition_rewards + transition_penalty_rewards

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
                        for idx in range(n_predator1 + n_predator2):
                            madqn.set_agent_buffer(idx)

                            if not (
                                len(observations_dict[idx]) >= 2
                                and len(action_dict[idx]) >= 2
                                and len(termination_dict[idx]) >= 2
                                and len(truncation_dict[idx]) >= 2
                            ):
                                continue

                            madqn.buffer.put(
                                observations_dict[idx][-2],
                                action_dict[idx][-2],
                                transition_rewards_total,
                                observations_dict[idx][-1],
                                termination_dict[idx][-2],
                                truncation_dict[idx][-2],
                            )

                    madqn.avg_move_append_pred1(madqn.step_move_count_pred[0] / max(n_predator1, 1))
                    madqn.avg_tag_append_pred1(madqn.step_tag_count_pred[0] / max(n_predator1, 1))

                    madqn.avg_move_append_pred2(madqn.step_move_count_pred[1] / max(n_predator2, 1))
                    madqn.avg_tag_append_pred2(madqn.step_tag_count_pred[1] / max(n_predator2, 1))

                    madqn.reset_step_move_count()
                    madqn.reset_step_tag_count()

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
                        }

                    metrics["steps/step"] = int(prev_step)
                    wandb.log(metrics)

            if agent[:8] == "predator":
                observation, reward, termination, truncation, info = last(env)

                if agent[9] == "1":
                    idx = int(agent[11:])
                    observation_temp = process_array_1(observation)
                    view_range = predator1_view_range
                    madqn.set_team_idx(0)
                else:
                    idx = int(agent[11:]) + n_predator1
                    observation_temp = process_array_2(observation)
                    view_range = predator2_view_range
                    madqn.set_team_idx(1)

                madqn.set_agent_info(agent)

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    n_iteration += 1
                    continue

                action = madqn.get_action(state=observation_temp, mask=None)
                env.step(action)

                # move
                if action in [0, 1, 3, 4]:
                    if idx < n_predator1:
                        move_penalty_dict[idx].append(float(args.move_penalty))
                        ep_move_count_pred1 += 1
                    else:
                        move_penalty_dict[idx].append(0.0)
                        ep_move_count_pred2 += 1

                    madqn.ep_move_count()
                    madqn.step_move_count()
                    move = 1

                # tag
                elif action in [5, 6, 7, 8, 9, 10, 11, 12]:
                    move_penalty_dict[idx].append(0.0)
                    madqn.step_tag_count()
                    move = 2

                # stay
                else:
                    move_penalty_dict[idx].append(0.0)
                    move = 0

                madqn.agent_action_deque_dict[idx].append(move)

                reward = env._cumulative_rewards[agent]

                observations_dict[idx].append(observation_temp)
                action_dict[idx].append(action)
                reward_dict[idx].append(reward)
                termination_dict[idx].append(termination)
                truncation_dict[idx].append(truncation)

                if madqn.buffer.size() >= args.trainstart_buffersize:
                    madqn.replay()

            # prey
            else:
                observation, reward, termination, truncation, _ = last(env)

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    n_iteration += 1
                    continue

                action = env.action_space(agent).sample()
                env.step(action)

            n_iteration += 1

        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):
                madqn.set_agent_model(agent)
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
                path = 'Model_decen/model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                th.save(madqn.gdqns[i].state_dict(), path)

                path_t = 'Model_decen/model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt'
                os.makedirs(os.path.dirname(path_t), exist_ok=True)
                th.save(madqn.gdqn_targets[i].state_dict(), path_t)


    env.close()

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    # for i in range(len(madqn.gdqns)) :
    #     th.save(madqn.gdqns[i].state_dict(), 'Model_decen/model_save/'+'model_'+ str(i) +'.pt')
    #     th.save(madqn.gdqns[i].state_dict(), 'Model_decen/model_save/' + 'model_target_' + str(i) + '.pt')

    print('done')


