import argparse
import numpy as np
import asyncio

from src.robot_control.scripts.marltoolkit.agents import BaseAgent
from src.robot_control.scripts.marltoolkit.data.ma_replaybuffer import EpisodeData, ReplayBuffer


def run_train_episode(
    env,
    agent: BaseAgent,
    rpm: ReplayBuffer,
    args: argparse.Namespace = None,
):

    episode_limit = args.episode_limit
    agent.reset_agent()
    episode_reward = 0.0
    episode_step = 0
    terminated = False
    state, obs = env.reset()
    episode_experience = EpisodeData(
        episode_limit=episode_limit,
        state_shape=args.state_shape,
        obs_shape=args.obs_shape,
        num_actions=args.n_actions,
        num_agents=args.n_agents,
    )

    while not terminated:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        actions_onehot = env._get_actions_one_hot(actions)
        next_state, next_obs, reward, terminated = env.step(actions)
        episode_reward += reward
        episode_step += 1
        
        episode_experience.add(state, obs, actions, actions_onehot,
                               available_actions, reward, terminated, 0)
        state = next_state
        obs = next_obs
        
        # await asyncio.sleep(0.01)
    
    # fill the episode
    for _ in range(episode_step, episode_limit):
        episode_experience.fill_mask()

    episode_data = episode_experience.get_data()

    rpm.store(**episode_data)

    mean_loss = []
    mean_td_error = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.update_learner_freq):
            batch = rpm.sample_batch(args.batch_size)
            loss, td_error = agent.learn(**batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None

    return episode_reward, episode_step, mean_loss, mean_td_error


def run_evaluate_episode(
    env,
    agent: BaseAgent,
    num_eval_episodes: int = 5,
):
    eval_reward_buffer = []
    eval_steps_buffer = []
    for _ in range(num_eval_episodes):
        print(f"qtran_runner:evaluation episode{num_eval_episodes}")
        agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        state, obs = env.reset()
        while not terminated:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions)
            state, obs, reward, terminated = env.step(actions)
            print(f"obs : {obs}")
            episode_step += 1
            episode_reward += reward

        eval_reward_buffer.append(episode_reward)
        eval_steps_buffer.append(episode_step)

    eval_rewards = np.mean(eval_reward_buffer)
    eval_steps = np.mean(eval_steps_buffer)

    return eval_rewards, eval_steps
