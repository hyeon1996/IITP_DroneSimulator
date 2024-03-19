# GazeboGym

It is the ROS2 workspace that controls the drone in Gazebo.

## Prerequisites
- Ubuntu 22.02
- [Install PX4](https://docs.px4.io/main/en/ros/ros2_comm.html#install-px4)
- [Install ROS2 (humble)](https://docs.px4.io/main/en/ros/ros2_comm.html#install-ros-2)
- [Setup Micro XRCE-DDS Agent & Client](https://docs.px4.io/main/en/ros/ros2_comm.html#setup-micro-xrce-dds-agent-client)

## How to run
1. Update submodule (only once)
```
git submodule update --init
```
2. Start Micro XRCE-DDS Agent 
```
MicroXRCEAgent udp4 -p 8888
```
3. Start client
```
# for single drone agent simulation
make px4_sitl gz_x500 
```
4. Build ROS2 workspace
  - move to GazeboGym directory and source ROS2 environment
  - ```
    cd GazeboGym
    source /opt/ros/humble/setup.bash
    ```
  -  build workspace
  - ```
    colcon build
    ```
  - source ROS2 environment of current workspace
  - ```
    source install/local_setup.bash
    ```
5. Run
```
ros2 run px4_control drone_control
```
