#!/usr/bin/env python

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

import rospy
import signal
import time 

# import our training environment
from openai_ros.robot_envs import multiagent_drone_env
from openai_ros.task_envs.drone import continous_multiagent_drone_goal
from geometry_msgs.msg import Point

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

def is_in_desired_position(desired_point, current_position, epsilon=0.2):
    """
    It return True if the current position is similar to the desired poistion
    """
    
    is_in_desired_pos = False

    x_pos_plus = desired_point.x + epsilon
    x_pos_minus = desired_point.x - epsilon
    y_pos_plus = desired_point.y + epsilon
    y_pos_minus = desired_point.y - epsilon

    x_current = current_position.x
    y_current = current_position.y

    x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
    y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

    is_in_desired_pos = x_pos_are_close and y_pos_are_close

    return is_in_desired_pos

if __name__ =="__main__":

    rospy.init_node('reset_test', anonymous=True, log_level=rospy.WARN)

    rospy.logwarn("-----mlic-version-----")
    
    # Parameter setting
    args = argparse.Namespace(
    env_name="MultiagentDrone-v1",
    policy="Gaussian",
    eval=True,
    gamma=0.99,
    tau=0.005,
    lr=0.0003,
    alpha=0.2,
    automatic_entropy_tuning=False,
    seed=1,
    batch_size=256,
    num_steps=1000001,
    hidden_size=64,
    updates_per_step=1,
    start_steps=1,  #10000
    target_update_interval=1,
    replay_size=1000000,
    cuda=True
    )
    
    
    device = torch.device("cuda" if args.cuda else "cpu")
    rospy.logwarn("device : {}".format(device))

    # Environment
    rospy.logwarn("Gym environment start")
    env = gym.make("MultiagentDrone-v1")
    env = NormalizedActions(env)
    rospy.logwarn("Gym environment done")

    default_handler = signal.getsignal(signal.SIGINT)
    def handler(num, frame) : 
        env.close()    
        return default_handler(num, frame)
    signal.signal(signal.SIGINT, handler)

    env.seed(args.seed)
    # env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0] // 2
    print("state_dim is:", state_dim)

    action_dim = env.action_space.shape[0]
    print("action_dim is:", action_dim)

    # Agent
    agent = SAC(env.observation_space.shape[0] // 2, env.action_space, args)

    #Tesnorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    print("Training start")

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False    
        # rospy.logwarn("takeoff")
        # env.takeoff()
        print("start reset")
        state = env.reset()
        # rospy.logwarn("takeoff")
        # env.takeoff()
        print("end reset")
        
        print("initial state is:", state)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("epsiode number is:", i_episode)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
                action = action[0]
              
            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            next_state, reward, done, _ = env.step_drone(action) # Step
            done = done[0]
            # print(next_state)
            # print(action)
            # print(reward)
            # print(done)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward[0]
            mask = 1 if episode_steps == 700 else float(not done)

            memory.push(state, action, reward, next_state, mask) 

            state = next_state

            if episode_steps == 700:
                done = True

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                episode_steps = 0
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    action = action[0]
                    next_state, reward, done, _ = env.step_drone(action)
                    done = done[0]
                    episode_steps += 1
                    episode_reward += reward[0]

                    state = next_state

                    if episode_steps == 700:
                        done = True

                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
    # while True : 
    #    rospy.logwarn("reset")
    #    state = env.reset()
    #    #rospy.logwarn("takeoff")
    #    #env.takeoff()
        
        '''
        for _ in range(100) : 
            #actions = [[0.0]*action_dim]*num_agents
            actions = [[0.5, 0.0, 0.0, 0.0]]*num_agents
            next_state, reward, done, info = env.step(actions)
        
        '''
        
    #    time.sleep(5)
        '''
        rospy.logwarn("land")
        for i in range(num_agents) : 
            env.land(i)
            #time.sleep(0.5)
        
        for i in range(num_agents) : 
            env.wait_for_stop(i, update_rate=10) 
        
        time.sleep(30)
        '''
        #state = env.reset()
    env.close()

