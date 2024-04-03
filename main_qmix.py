#!/usr/bin/env python3


# https://github.com/m2-farzan/ros2-asyncio

import asyncio
from MultiAgentEnv import MultiAgentEnv

import numpy as np
import random
import argparse
import time
import sys

import time
import os
from copy import deepcopy
import torch
from torch.utils.tensorboard import SummaryWriter

from src.robot_control.scripts.configs.arguments import get_common_args
from src.robot_control.scripts.configs.qmix_config import QMixConfig
from src.robot_control.scripts.marltoolkit.agents.qmix_agent import QMixAgent
from src.robot_control.scripts.marltoolkit.data.ma_replaybuffer import ReplayBuffer
from src.robot_control.scripts.marltoolkit.modules.actors import RNNActor
from src.robot_control.scripts.marltoolkit.modules.mixers import QMixerModel
from src.robot_control.scripts.marltoolkit.runners.episode_runner import (run_evaluate_episode, run_train_episode)
from src.robot_control.scripts.marltoolkit.utils import (ProgressBar, TensorboardLogger,
                               get_outdir, get_root_logger)
from src.robot_control.scripts.marltoolkit.data.ma_replaybuffer import EpisodeData, ReplayBuffer

async def main(args):

    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # 각 요소 class 
    gz_env = MultiAgentEnv(args)
    await gz_env.setup()
    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # QMIX 학습 관련
    qmix_config = QMixConfig()
    common_args = get_common_args()
    args = argparse.Namespace(**vars(common_args), **vars(qmix_config))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 학습 config 설정
    get_env_info = gz_env.get_env_info()
    args.episode_limit = get_env_info["episode_limit"]
    args.obs_shape = get_env_info["obs_shape"][1]
    args.state_shape = get_env_info["state_shape"][0]
    args.n_agents = get_env_info["n_agents"]
    args.n_actions = gz_env.n_actions
    args.device = device
    '''end'''''''''''''''''''''''''''''''''''''''''''''
    
    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # model 저장 및 log 설정
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    log_name = os.path.join(args.project, args.scenario, args.algo_name, timestamp).replace(os.path.sep, '_')
    log_path = os.path.join(args.log_dir, args.project, args.scenario, args.algo_name)
    tensorboard_log_path = get_outdir(log_path, 'tensorboard_log_dir')
    log_file = os.path.join(log_path, log_name + '.qmix_log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')
    model_path = os.getcwd() + '/cooperative_marl_ros/src/robot_control/scripts/qmix_models/'

    print(model_path)
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    logger = TensorboardLogger(writer)
    '''end'''''''''''''''''''''''''''''''''''''''''''''
    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # 학습 buffer 설정
    rpm = ReplayBuffer(
        max_size=args.replay_buffer_size,
        episode_limit=args.episode_limit,
        state_shape=args.state_shape,
        obs_shape=args.obs_shape,
        num_agents=args.n_agents,
        num_actions=args.n_actions,
        batch_size=args.batch_size,
        device=device)
    '''end'''''''''''''''''''''''''''''''''''''''''''''

    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # agent model 설정
    agent_model = RNNActor(
        input_shape=args.obs_shape,
        rnn_hidden_dim=args.rnn_hidden_dim,
        n_actions=args.n_actions,
    )
    # Mixer
    mixer_model = QMixerModel(3, args.state_shape)

    # QMIX
    marl_agent = QMixAgent(
        agent_model=agent_model,
        mixer_model=mixer_model,
        n_agents=args.n_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        exploration_start=args.exploration_start,
        min_exploration=args.min_exploration,
        update_target_interval=args.update_target_interval,
        update_learner_freq=args.update_learner_freq,
        clip_grad_norm=args.clip_grad_norm,
        optim_alpha = 0.99,
        optim_eps = 0.00001,
        device=args.device,
    )
    '''end'''''''''''''''''''''''''''''''''''''''''''''
    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # 학습 과정 prgressBar 설정
    # progress_bar = ProgressBar(args.memory_warmup_size)
    
    # while rpm.size() < args.memory_warmup_size:
    #     run_train_episode(gz_env, marl_agent, rpm, args)
    #     progress_bar.update()

    steps_cnt = 0
    episode_cnt = 0
    # progress_bar = ProgressBar(args.total_steps)
    '''end'''''''''''''''''''''''''''''''''''''''''''''

    '''start'''''''''''''''''''''''''''''''''''''''''''''
    # 학습 시작
    
    while steps_cnt < args.total_steps:
        epi_count = 0
        episode_limit = args.episode_limit
        marl_agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        episode_experience = EpisodeData(
            episode_limit=episode_limit,
            state_shape=args.state_shape,
            obs_shape=args.obs_shape,
            num_actions=args.n_actions,
            num_agents=args.n_agents,
        )

        obs = gz_env.get_obs()
        state = gz_env.get_state()
        while not terminated:
            epi_count += 1
            available_actions = gz_env.get_available_actions()
            actions = marl_agent.sample(obs, available_actions)
            actions_onehot = gz_env._get_actions_one_hot(actions)
            next_state, next_obs, reward, terminated = await gz_env.step(actions)
            episode_reward += reward
            episode_step += 1
            episode_experience.add(state, obs, actions, actions_onehot,
                                available_actions, reward, terminated, 0)
            state = next_state
            obs = next_obs

            await asyncio.sleep(0.02)

            if (epi_count % episode_limit == 0) or terminated:
                #print("reset")
                #print("step", epi_count)
                state, obs = await gz_env.reset()
                break
                


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
                loss, td_error = marl_agent.learn(**batch)
                print(loss, td_error)
                mean_loss.append(loss)
                mean_td_error.append(td_error)

        mean_loss = np.mean(mean_loss) if mean_loss else 0
        mean_td_error = np.mean(mean_td_error) if mean_td_error else 0


        # update episodes and steps
        episode_cnt += 1
        steps_cnt += episode_step

        # 학습 learning rate 설정
        # learning rate decay
        marl_agent.learning_rate = max(
            marl_agent.lr_scheduler.step(episode_step),
            marl_agent.min_learning_rate)

        # train results
        train_results = {
            'env_step': episode_step,
            'rewards': episode_reward,
            'mean_loss': mean_loss,
            'mean_td_error': mean_td_error,
            'exploration': marl_agent.exploration,
            'learning_rate': marl_agent.learning_rate,
            'replay_buffer_size': rpm.size(),
            'target_update_count': marl_agent.target_update_count,
            }
        
        # log 
        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Train], episode: {}, train_reward: {:.2f}'
                .format(episode_cnt, episode_reward))
            logger.log_train_data(train_results, steps_cnt)

        # if episode_cnt % args.test_log_interval == 0:
        # # if episode_cnt % 1 == 0:
        #     gz_env.set_gazebo_env()
        #     await gz_env.setup()

        #     eval_reward_buffer = []
        #     eval_steps_buffer = []

        #     num_eval_episodes = 3
        #     for i in range(num_eval_episodes):
        #         print(f"qtran_runner:evaluation episode{num_eval_episodes}")
        #         marl_agent.reset_agent()
        #         episode_reward = 0.0
        #         episode_step = 0
        #         terminated = False
                
        #         obs = gz_env.get_obs()
        #         state = gz_env.get_state()
        #         while not terminated:
        #             available_actions = gz_env.get_available_actions()
        #             actions = marl_agent.predict(obs, available_actions)
        #             state, obs, reward, terminated = await gz_env.step(actions)
        #             # print(reward)
        #             # print(f"obs : {obs}")
        #             episode_step += 1
        #             episode_reward += reward

        #             if (episode_step % episode_limit == 0) or terminated:
        #                 state, obs = await gz_env.reset()
        #                 break

        #         eval_reward_buffer.append(episode_reward)
        #         eval_steps_buffer.append(episode_step)
                
        #         gz_env.set_gazebo_env()
        #         await gz_env.setup()
                    

            
            # eval_rewards = np.mean(eval_reward_buffer)
            # eval_steps = np.mean(eval_steps_buffer)
            
            # text_logger.info(
            #     '[Eval], episode: {}, eval_rewards: {:.2f}'
            #     .format(episode_cnt, eval_rewards))

            # test_results = {
            #     'env_step': eval_steps,
            #     'rewards': eval_rewards
            # }
            # logger.log_test_data(test_results, steps_cnt)
            marl_agent.save(model_path)

        gz_env.set_gazebo_env()
        await gz_env.setup()


        # progress_bar.update(episode_step)

    # env = MultiAgentEnv(args)

    # await env.setup()

    # step = 0
    # while True:
    #     t1 = time.time()

    #     action = np.random.randint(env.n_actions, size=env.n_agents)  # i.e. array([8, 2, 0])

    #     next_obs, reward, done = await env.step(action)
    #     # print(obs)

    #     print("next_obs", next_obs)
    #     print("reward", reward)
    #     print("done", done)

    #     step = step + 1

    #     await asyncio.sleep(0.01)  # 100 Hz ?

    #     t2 = time.time()
    #     print("loop_time = ", t2 - t1)


    #     if (step % env.episode_limit == 0) or done:
    #         print("reset")
    #         print("step", step)
    #         obs = await env.reset()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_agents', default=3, type=int)
    parser.add_argument('--episode_limit', default=1000, type=int)
    parser.add_argument('--px4_git_dir', default="../PX4-Autopilot", type=str)
    parser.add_argument('--mavserver_dir', default="../GazeboPX4MARL", type=str)
    parser.add_argument('--goal_margin', default=0.6, type=float)
    parser.add_argument('--altitude_margin', default=0.3, type=float)
    parser.add_argument('--rad_per_s_margin', default=1.0, type=float)
    parser.add_argument('--target_altitude', default=4.0, type=float)

    args, _ = parser.parse_known_args()
    print("args:", args)
    
    asyncio.run(main(args))

    




