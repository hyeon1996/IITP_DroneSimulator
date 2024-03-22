import asyncio
from Env.MultiAgentEnv import MultiAgentEnv
from mavsdk import System

async def f():
    agent = System()
    await agent.connect(system_address=f"udp://:14541")
    async for odometry in agent.telemetry.odometry():
        print(f"z_m {odometry.position_body.z_m}")
        # print([odometry.position_body.x_m, odometry.position_body.y_m, odometry.position_body.z_m,
        #                 odometry.velocity_body.x_m_s, odometry.velocity_body.y_m_s, odometry.velocity_body.z_m_s,
        #                 odometry.angular_velocity_body.roll_rad_s, odometry.angular_velocity_body.pitch_rad_s,
        #                 odometry.angular_velocity_body.yaw_rad_s])

async def main():
    env = MultiAgentEnv(2,2)
    await env.setup()
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())