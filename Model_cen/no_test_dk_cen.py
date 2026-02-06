from magent2_backup.environments import hetero_adversarial_v1
from MADQN_cen import MADQN
import argparse
import torch as th
import wandb
import tqdm
import numpy as np
import os


MOVE_ACTIONS = {0, 1, 3, 4}


def last(values, k=1, default=0.0):
	try:
		return values[-k]
	except Exception:
		return default


def get_agent_positions(env):
	handles = env.env.env.env.env.get_handles()
	pos_predator1 = env.env.env.env.env.get_pos(handles[0])
	pos_predator2 = env.env.env.env.env.get_pos(handles[1])

	return pos_predator1, pos_predator2


def get_args():
	parser = argparse.ArgumentParser(description='MADQN_cen')

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
	parser.add_argument('--tag_reward', type=float, default=3)
	parser.add_argument('--tag_penalty', type=float, default=-0.2)
	parser.add_argument('--move_penalty', type=float, default=-0.15)

	parser.add_argument('--seed', type=int, default=125)

	return parser.parse_args()


args = get_args()

wandb.init(project="MADQN_01", entity='hails',config=args.__dict__)
wandb.run.name = 'madqn_centralized'

device = th.device("cuda" if th.cuda.is_available() else "cpu")

render_mode = 'rgb_array'

entire_state = (args.map_size, args.map_size, args.dim_feature * 2) # H, W, C*2
dim_act = 13

n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey



madqn = MADQN(n_predator1, n_predator2, dim_act, entire_state, device=device, buffer_size=args.buffer_size, args=args)

# for i in range(n_predator1 + n_predator2):
# 	madqn.gdqns[i].load_state_dict(th.load(f'./model_cen_save/model_{i}_ep50.pt'))

		# model_file_name = f'./model_cen_save/model_{i}_ep50.pt'
		#
		# # ?? ?? ?? ??
		# model_state_dict = th.load(model_file_name)
		#
		# # ?? ?? ??? ?? ??? ??
		# madqn.gdqns[i].load_state_dict(model_state_dict)

# def _expand_team_and_pred2(result):
# 	pos_list1 = np.argwhere(result[:, :, 1] == 1)
# 	pos_list2 = np.argwhere(result[:, :, 2] == 1)
#
# 	H, W, _ = result.shape
#
# 	for i in pos_list1:
# 		r, c = int(i[0]), int(i[1])
# 		if r + 1 < H:
# 			result[r + 1, c, 1] = 1
# 		if c + 1 < W:
# 			result[r, c + 1, 1] = 1
# 		if (r + 1 < H) and (c + 1 < W):
# 			result[r + 1, c + 1, 1] = 1
#
# 	for i in pos_list2:
# 		r, c = int(i[0]), int(i[1])
# 		if r + 1 < H:
# 			result[r + 1, c, 2] = 1
# 		if c + 1 < W:
# 			result[r, c + 1, 2] = 1
# 		if (r + 1 < H) and (c + 1 < W):
# 			result[r + 1, c + 1, 2] = 1
#
# 	return result


def process_array_1(arr):
	arr = np.delete(arr, [2, 4, 6], axis=2)
	result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]))
	return result


def process_array_2(arr):
	arr = np.delete(arr, [2, 4, 6], axis=2)
	result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 3], arr[:, :, 2]))
	return result


def pos_overlay(global_obs, local_obs, pos, view_range): # (view_range*2, view_range*2, C) with pos
	overlay = np.zeros_like(global_obs)
	if pos is None:
		return overlay

	H, W, _ = global_obs.shape
	patch_h, patch_w, _ = local_obs.shape

	try:
		x_center = int(pos[1])
		y_center = int(pos[0])
	except Exception:
		return overlay


	x0 = x_center - (view_range - 1) # slice from global state - 2*view_range
	x1 = x_center + (view_range + 1)
	y0 = y_center - (view_range - 1)
	y1 = y_center + (view_range + 1)

	o_x0 = max(x0, 0)
	o_x1 = min(x1, H)
	o_y0 = max(y0, 0)
	o_y1 = min(y1, W)

	
	p_x0 = o_x0 - x0  # corresponding patch slice
	p_y0 = o_y0 - y0
	p_x1 = p_x0 + (o_x1 - o_x0)
	p_y1 = p_y0 + (o_y1 - o_y0)

	p_x1 = min(p_x1, patch_h) # patch clip
	p_y1 = min(p_y1, patch_w)
	o_x1 = o_x0 + (p_x1 - p_x0)
	o_y1 = o_y0 + (p_y1 - p_y0)

	if (o_x1 > o_x0) and (o_y1 > o_y0) and (p_x1 > p_x0) and (p_y1 > p_y0):
		overlay[o_x0:o_x1, o_y0:o_y1, :] = local_obs[p_x0:p_x1, p_y0:p_y1, :]
	return overlay

def main():

	for ep in range(args.total_ep):

		iteration_number = 0
		ep_reward = 0
		ep_reward_pred1 = 0
		ep_reward_pred2 = 0
		ep_reward_np = 0
		ep_reward_pred1_np = 0
		ep_reward_pred2_np = 0
		ep_move_count_pred1 = 0
		ep_move_count_pred2 = 0

		env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
									max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode,
									predator1_view_range=args.predator1_view_range,
									predator2_view_range=args.predator2_view_range,
									n_predator1=args.n_predator1,
									n_predator2=args.n_predator2,
									n_prey=args.n_prey,
									tag_reward=args.tag_reward,
									)

		env.reset(seed=args.seed)
		print("ep:",ep,'*' * 80)


		observations_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			observations_dict[agent_idx] = []

		agent_pos = {}
		for agent_idx in range(n_predator1 + n_predator2):
			agent_pos[agent_idx] = []

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

		for agent in env.agent_iter():

			step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)

			if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0
					and step_idx > 0 and step_idx != args.max_update_steps):

				total_last_rewards = 0
				total_move_penalty = 0
				total_last_rewards_np = 0
				step_reward_pred1 = 0
				step_reward_pred2 = 0
				step_penalty_pred1 = 0
				step_penalty_pred2 = 0
				step_move_count_pred1 = 0
				step_move_count_pred2 = 0

				for agent_rewards in reward_dict.values():
					total_last_rewards += np.sum(last(agent_rewards, k=1, default=0.0))
					total_last_rewards_np += np.sum(last(agent_rewards, k=1, default=0.0))

				for penalty in move_penalty_dict.values():
					total_move_penalty += np.sum(last(penalty, k=1, default=0.0))

				for idx in range(args.n_predator1 + args.n_predator2):
					agent_reward = float(last(reward_dict[idx], k=1, default=0.0))
					agent_penalty = float(last(move_penalty_dict[idx], k=1, default=0.0))
					agent_action = last(action_dict[idx], k=1, default=None)
					is_move = (agent_action is not None) and (int(agent_action) in MOVE_ACTIONS)

					if idx < args.n_predator1:
						step_reward_pred1 += agent_reward
						step_penalty_pred1 += agent_penalty
						step_move_count_pred1 += int(is_move)
					else:
						step_reward_pred2 += agent_reward
						step_penalty_pred2 += agent_penalty
						step_move_count_pred2 += int(is_move)


				total_last_rewards = total_last_rewards + total_move_penalty

				ep_reward += total_last_rewards
				ep_reward_np += total_last_rewards_np

				step_rewards_pred1 = step_reward_pred1 + step_penalty_pred1
				step_rewards_pred2 = step_reward_pred2 + step_penalty_pred2
				ep_reward_pred1 += step_rewards_pred1
				ep_reward_pred2 += step_rewards_pred2
				ep_reward_pred1_np += step_reward_pred1
				ep_reward_pred2_np += step_reward_pred2
				ep_move_count_pred1 += step_move_count_pred1
				ep_move_count_pred2 += step_move_count_pred2

				# Only start pushing transitions once we have (t-1, t) history
				if step_idx > 1:
					for idx in range(args.n_predator1 + args.n_predator2):
						madqn.set_agent_buffer(idx)
						madqn.buffer.put(
							observations_dict[idx][-2],
							action_dict[idx][-2],
							total_last_rewards,
							observations_dict[idx][-1],
							termination_dict[idx][-2],
							truncation_dict[idx][-2],
						)

				# print('ep:{}'.format(ep))
				# print("predator total_reward", total_last_rewards)
				# print("*"*10)

				metrics = {
					"steps/step": step_idx,
					"steps/total_step_reward": float(total_last_rewards),
					"steps/total_step_reward_np": float(total_last_rewards_np),
					"predator1/step_reward": float(step_rewards_pred1),
					"predator2/step_reward": float(step_rewards_pred2),
					"predator1/step_reward_np": float(step_reward_pred1),
					"predator2/step_reward_np": float(step_reward_pred2),
					"predator1/move_count": int(step_move_count_pred1),
					"predator2/move_count": int(step_move_count_pred2),
				}

				wandb.log(metrics)

				# if madqn.buffer.size() >= args.trainstart_buffersize:
				# 	wandb.log({"total_last_rewards": total_last_rewards })


			observation_local, reward, termination, truncation, info = env.last()
			observation_global = env.state()
			global_obs = process_array_1(observation_global)

			if agent[:8] == "predator":

				pos_predator1, pos_predator2 = get_agent_positions(env)

				if agent[9] == "1": # predator1
					idx = int(agent[11:])
					pos = pos_predator1[idx]
					view_range = args.predator1_view_range

				else: # predator2
					idx = int(agent[11:]) + n_predator1
					pos = pos_predator2[idx - n_predator1]
					view_range = args.predator2_view_range

				if agent[9] == "1":
					local_obs = process_array_1(observation_local)
				else:
					local_obs = process_array_2(observation_local)
				local_overlay = pos_overlay(
					global_obs=global_obs,
					local_obs=local_obs,
					pos=pos,
					view_range=view_range,
				)

				observation_temp = np.concatenate((global_obs, local_overlay), axis=2)

				madqn.set_agent_info(agent)

				if termination or truncation:
					print(agent , 'is terminated')
					env.step(None)
					continue

				else:
					action = madqn.get_action(state=observation_temp, mask=None)
					env.step(action)
					reward = env._cumulative_rewards[agent] # agent

					# move penalty to predator1
					if (idx < n_predator1) and (action in MOVE_ACTIONS):
						move_penalty_dict[idx].append(args.move_penalty)
					else:
						move_penalty_dict[idx].append(0)


					observations_dict[idx].append(observation_temp) #s
					action_dict[idx].append(action)					#a
					reward_dict[idx].append(reward)					#r
					termination_dict[idx].append(termination)		#t_{t-1]
					truncation_dict[idx].append(truncation)			#t_{t-1]
					agent_pos[idx].append(pos)

					if madqn.buffer.size() >= args.trainstart_buffersize:

						madqn.replay()

			else : #prey
				_, _, termination, truncation, _ = env.last()

				if termination or truncation:
					print(agent, 'is terminated')
					env.step(None)

					continue


				else:
					action = env.action_space(agent).sample()
					env.step(action)


			iteration_number += 1

		pred1_move = max(ep_move_count_pred1, 1)
		pred2_move = max(ep_move_count_pred2, 1)

		episode_log = {
			"episode/episode": ep,
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

		env.close()

		# if madqn.buffer.size() >= args.trainstart_buffersize:
		# 	wandb.log({"ep_reward": ep_reward})



		#ep_reward += total_last_rewards
		#print("ep_reward:", ep_reward)

		# if iteration_number > args.max_update_steps:
		# 	print('*' * 10, 'train over', '*' * 10)
		# 	print(iteration_number)
		# 	break


		if ep > args.total_ep: #100
			print('*' * 10, 'train over', '*' * 10)
			print(iteration_number)
			break

		if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
			for agent in range(args.n_predator1 + args.n_predator2):
				madqn.set_agent_model(agent)
				madqn.target_update()

		# if ((ep % 50) ==0) and ep >1 :
		# 	for i in range(len(madqn.gdqns)) :
		# 		th.save(madqn.gdqns[i].state_dict(), './model_cen_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
		# 		th.save(madqn.gdqn_targets[i].state_dict(), './model_cen_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

	print('*' * 10, 'train over', '*' * 10)

if __name__ == '__main__':
	main()
	#print('done')
	#??? ??# for i in range(len(madqn.gdqns)) :
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')


	print('done')
