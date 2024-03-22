from mavsdk import System
import asyncio as aio
import numpy as np


class MultiAgentEnv:
    def __init__(self, n_agents, episode_limit):
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.agents = [System(mavsdk_server_address="127.0.0.1", port=50051+i) for i in range(self.n_agents)]
        self.obs = np.zeros([n_agents, 9])
        self.states = np.zeros([n_agents, 3])

    async def setup(self):
        for i, agent in enumerate(self.agents):
            await agent.connect()
            aio.create_task(self.subscribe_odometry(agent, i))
            aio.create_task(self.subscribe_ground_truth(agent, i))


    async def subscribe_odometry(self, agent, id):
        async for odometry in agent.telemetry.odometry():
            self.obs[id] = [odometry.position_body.x_m, odometry.position_body.y_m, odometry.position_body.z_m,
                            odometry.velocity_body.x_m_s, odometry.velocity_body.y_m_s, odometry.velocity_body.z_m_s,
                            odometry.angular_velocity_body.roll_rad_s, odometry.angular_velocity_body.pitch_rad_s,
                            odometry.angular_velocity_body.yaw_rad_s]

    async def subscribe_ground_truth(self, agent, id):
        async for ground_truth in agent.telemetry.ground_truth():
            self.states[id] = [ground_truth.latitude_deg, ground_truth.longitude_deg, ground_truth.absolute_altitude_m]

    def step(self, actions):
        """Returns observations (agents x observations), states (agents x states), reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        return self.obs

    def get_obs_size(self):
        return self.obs.shape

    def get_state(self):
        return self.states

    def get_state_size(self):
        return self.states.shape

    def get_action_size(self):
        """Returns the size of the action space."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_action_size(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info
