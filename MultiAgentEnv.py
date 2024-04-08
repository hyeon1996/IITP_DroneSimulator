#!/usr/bin/env python3


from mavsdk import System
import asyncio as aio
import numpy as np

import subprocess
import time
import random

from src.robot_control.scripts.marltoolkit.utils.transforms import OneHotTransform


class MultiAgentEnv:
    def __init__(self, args):

        self.n_agents = args.n_agents
        self.episode_limit = args.episode_limit

        # self.agents = [System(mavsdk_server_address="127.0.0.1", port=50051+i) for i in range(self.n_agents)]
        # print("self.agents", self.agents)
        self.obs = np.zeros([self.n_agents, 9])
        self.states = np.zeros([self.n_agents, 3])

        self.goal_position = [[7.0, -3.0, 0.0], [7.0, 0.0, 0.0], [7.0, 3.0, 0.0]]
        self.goal_margin = args.goal_margin  # 0.5
        self.altitude_margin = args.altitude_margin  # 0.5
        self.target_altitude = args.target_altitude  # 5
        self.rad_per_s_margin = args.rad_per_s_margin
        self.init_goal_distances = np.zeros([self.n_agents])

        # Test set of manual inputs. Format: [roll, pitch, throttle, yaw]
        self.manual_actions = [
            [ 0.0,  0.0,  0.5,  0.0 ],  # no movement
            [-1.0,  0.0,  0.5,  0.0 ],  # minimum roll
            [ 1.0,  0.0,  0.5,  0.0 ],  # maximum roll
            [ 0.0, -1.0,  0.5,  0.0 ],  # minimum pitch
            [ 0.0,  1.0,  0.5,  0.0 ],  # maximum pitch
            [ 0.0,  0.0,  0.5, -1.0 ],  # minimum yaw
            [ 0.0,  0.0,  0.5,  1.0 ],  # maximum yaw
            [ 0.0,  0.0,  1.0,  0.0 ],  # max throttle
            [ 0.0,  0.0,  0.0,  0.0 ],  # minimum throttle
        ]
        self.n_actions = len(self.manual_actions)

        self.px4_git_dir = args.px4_git_dir  # "/home/mo/drone/PX4-Autopilot"
        self.mavserver_dir = args.mavserver_dir  # "/home/mo/drone/gazebo_gym_ws/src/dronepkg/dronepkg"

        self.gz_processes = []
        self.mavserv_processes = []

        self.actions_one_hot_transform = OneHotTransform(self.n_actions)
        self.gz_processes = []
        self.mavserv_processes = []
        self.subs_odom_tasks = []
        # self.subs_ground_truth_tasks = []
        self.breakover=False


    async def set_gazebo_env(self):
        print("set Gazebo, Px4, MAVSDK_server ...")
        for i in range(self.n_agents):
            pos = '0,' + str(i)
            id = str(i + 1)  # VERBOSE_SIM=0 ?, HEADLESS=1
            gz_cmd_args = "HEADLESS=1 PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE=" + pos + " PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i " + id
            print("gz_cmd_args :", gz_cmd_args)
            self.gz_processes.append(subprocess.Popen(gz_cmd_args, cwd=self.px4_git_dir, shell=True, stdout=subprocess.DEVNULL))
            await aio.sleep(0.01)

            port = str(50051 + i)
            udp = str(14541 + i)
            mavserv_cmd_args = "./mavsdk_server -p " + port + " udp://:" + udp
            print("mavserv_cmd_args :", mavserv_cmd_args)
            self.mavserv_processes.append(subprocess.Popen(mavserv_cmd_args, cwd=self.mavserver_dir, shell=True, stdout=subprocess.DEVNULL))
        await aio.sleep(0.01)
        print("set done.")

    async def terminate_subprocesses(self):
        print("terminate subprocesses..")
        for i in range(len(self.gz_processes)):
            self.gz_processes[i].terminate()
            self.gz_processes[i].wait()
            self.mavserv_processes[i].terminate()
            self.mavserv_processes[i].wait()
            
        self.breakover=True
        for i in self.subs_odom_tasks:
            while not i.done():

                await aio.wait_for(i,timeout=5.0)
                try :
                    await aio.wait_for(i,timeout=5.0)
                except :
                    break
        

        # for i in range(len(self.subs_ground_truth_tasks)):
        #     while not self.subs_odom_tasks[i].done():
        #         await self.subs_odom_tasks[i]
        #         try:
        #             await self.subs_odom_tasks[i]
        #         except aio.CancelledError:
        #             pass
        #     while not self.subs_ground_truth_tasks[i].done():
        #         await self.subs_ground_truth_tasks[i]
        #         try:
        #             await self.subs_ground_truth_tasks[i]
        #         except aio.CancelledError:
        #             pass
        # self.subs_ground_truth_tasks = []
        self.subs_odom_tasks = []
        self.breakover=False
        
        subprocess.run(['pkill', '-9', '-f', 'gz'])
        subprocess.run(['pkill', '-9', '-f', 'px4'])
        subprocess.run(['pkill', '-9', '-f', 'mavsdk'])
        self.agents = [System(mavsdk_server_address="127.0.0.1", port=50051+i) for i in range(self.n_agents)]
        print("terminate done")


    async def setup(self):
        print("setup start")
        for i, agent in enumerate(self.agents):
            await aio.sleep(0.01)
            await aio.wait_for(agent.connect("udp://:{}".format(14541+i)),timeout=10)
            print("start create task")
            self.subs_odom_tasks.append(aio.create_task(self.subscribe_odometry(agent, i)))
            # self.subs_ground_truth_tasks.append(aio.create_task(self.subscribe_ground_truth(agent, i)))
            print("Try no init for state connection")
            tryNo=0
            async for state in agent.core.connection_state():
                if state.is_connected:
                    # print(f"-- Connected to drone!")
                    break
                elif tryNo>100:
                    print("tryNo over 100")
                    break
                else:
                    tryNo+=1
                    print("tryNo increased")
            
            if tryNo>100:
                raise RuntimeError("state is not connected")
            print("Try no init for health connection")
            tryNo=0
            async for health in agent.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    # print("-- Global position state is good enough for flying.")
                    break
                elif tryNo>100:
                    print("tryNo over 100")
                    break
                else:
                    tryNo+=1
                    print("tryNo increased")
            if tryNo>100:
                raise RuntimeError("health is not connected")
            await agent.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)
            print("-- Arming")
            await aio.sleep(0.1)
            await agent.action.arm()
            print("-- Taking off Agent {}".format(i))
            await agent.action.takeoff()
            await aio.sleep(0.1)
        await aio.sleep(0.1)

        self.init_goal_distances = self.calculate_goal_distance(self.obs)
        print("self.init_goal_distances", self.init_goal_distances)

    async def subscribe_odometry(self, agent, id):
        async for odom in agent.telemetry.odometry():
            self.obs[id] = [odom.position_body.x_m, odom.position_body.y_m, odom.position_body.z_m,
                            odom.velocity_body.x_m_s, odom.velocity_body.y_m_s, odom.velocity_body.z_m_s,
                            odom.angular_velocity_body.roll_rad_s, odom.angular_velocity_body.pitch_rad_s,
                            odom.angular_velocity_body.yaw_rad_s]
            self.states[id] = [odom.position_body.x_m, odom.position_body.y_m, odom.position_body.z_m]
            if self.breakover:
                break

    # async def subscribe_ground_truth(self, agent, id):
    #     async for ground_truth in agent.telemetry.ground_truth():
    #         self.states[id] = [ground_truth.latitude_deg, ground_truth.longitude_deg, ground_truth.absolute_altitude_m]
    #         if self.breakover:
    #             break

    async def step(self, actions):

        for i, agent in enumerate(self.agents):
            roll, pitch, throttle, yaw = self.manual_actions[actions[i]]
            await agent.manual_control.set_manual_control_input(roll, pitch, throttle, yaw)

        next_obs = self.get_obs()
        next_state = self.get_state()

        reward, goal_check, collison_check = self.get_reward(next_obs)
        done = goal_check or collison_check

        return next_state, next_obs, reward, done

    def get_reward(self, obs):

        # (1) 골 지점까지의 거리 : 가까울수록 큰 보상
        current_goal_distances = self.calculate_goal_distance(obs)
        goal_distances_ratio = current_goal_distances / self.init_goal_distances
        goal_distances_reward = (1 - goal_distances_ratio).sum()

        # (2) 골 포인트까지 도달하면 큰 보상
        goal_check = np.prod(current_goal_distances < self.goal_margin)  # 모두 다 도달하면,
        goal_reward = 1000 if goal_check else 0 # 5 -> 100

        # (3) 타겟 높이까지 올라가면 큰 보상
        altitude_check_list = [0.1 if self.target_altitude - self.altitude_margin < obs_i[2] < self.target_altitude + self.altitude_margin and obs_i[2] else 0 for obs_i in obs]

        altitude_check_reward = sum(altitude_check_list)

        # (4) 충돌 일어나면 페널티
        collison_check = [True if obs_i[6] > self.rad_per_s_margin or obs_i[7] > self.rad_per_s_margin or obs_i[8] > self.rad_per_s_margin else False for obs_i in obs]
        collison_check = (sum(collison_check) >= 1)
        # collison_check = False
        collison_reward = -1000 if collison_check else 0   # -5 -> -20

        # print(goal_distances_reward, goal_reward, altitude_check_reward, collison_reward)
        reward = goal_distances_reward + goal_reward + altitude_check_reward + collison_reward

        return reward, goal_check, collison_check


    def calculate_goal_distance(self, obs):
        distances = []
        for i in range(self.n_agents):
            d_i = (self.goal_position[i][0] - obs[i][0])**2 + (self.goal_position[i][1] - obs[i][1])**2 + (self.goal_position[i][2] - obs[i][2])**2
            distances.append(np.sqrt(d_i))
        return np.array(distances)

    def get_obs(self):
        return self.obs

    def get_obs_size(self):
        return self.obs.shape

    def get_state(self):
        return self.states.flatten()
    
    def get_state_size(self):
        return self.states.flatten().shape

    def get_action_size(self):
        """Returns the size of the action space."""
        return self.n_actions

    def _get_actions_one_hot(self, actions):
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self):
        available_actions = []
        for agent_id in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(agent_id))
        return np.array(available_actions)
    
    def get_avail_agent_actions(self, agent_id):
        avail_actions = [1,1,1,1,1,1,1,1,1]
        return avail_actions

    async def reset(self):
        tryNo = 0
        while True:
            tryNo+=1
            try:
                print("reset trial start. Trial No : {}".format(tryNo))
                await self.terminate_subprocesses()
                await aio.sleep(0.01)
                await self.set_gazebo_env()
                await self.setup()
                print("reset try succeed.")
                break
            except Exception as e:
                print("Reset failed. We will try arming again. Try number : {}".format(tryNo))
                print("error message : ", e)
                if tryNo > 100:
                    print("Reset failed over 100 times. System will shut down.")
                    self.terminate_subprocesses()
                    exit(1)
        return self.states, self.obs
    
    async def get_states_obs(self):
        return self.states, self.obs


    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_action_size(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info
