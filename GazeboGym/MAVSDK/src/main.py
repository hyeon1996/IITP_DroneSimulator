#!/usr/bin/env python3


# https://github.com/m2-farzan/ros2-asyncio

import asyncio
from MultiAgentEnv import MultiAgentEnv

import numpy as np
import random
import argparse
import time


async def main(args):

    env = MultiAgentEnv(args)

    await env.setup()

    step = 0
    while True:
        t1 = time.time()

        action = np.random.randint(env.n_actions, size=env.n_agents)  # i.e. array([8, 2, 0])

        next_obs, reward, done = await env.step(action)
        # print(obs)

        print("next_obs", next_obs)
        print("reward", reward)
        print("done", done)

        step = step + 1

        await asyncio.sleep(0.01)  # 100 Hz ?

        t2 = time.time()
        print("loop_time = ", t2 - t1)


        if (step % env.episode_limit == 0) or done:
            print("reset")
            print("step", step)
            obs = await env.reset()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_agents', default=3, type=int)
    parser.add_argument('--episode_limit', default=1000, type=int)
    parser.add_argument('--px4_git_dir', default="/home/mo/drone/PX4-Autopilot", type=str)
    parser.add_argument('--mavserver_dir', default="/home/mo/drone/GazeboPX4RL", type=str)
    parser.add_argument('--goal_margin', default=0.6, type=float)
    parser.add_argument('--altitude_margin', default=0.3, type=float)
    parser.add_argument('--rad_per_s_margin', default=1.0, type=float)
    parser.add_argument('--target_altitude', default=4.0, type=float)

    args, _ = parser.parse_known_args()
    print("args:", args)


    asyncio.run(main(args))
