드론 시뮬레이터 구현 코드 입니다. 

Require : 

- Ubuntu : 20.04 or 22.04
- python : >= 3.8
- install Gazebo Garden : https://gazebosim.org/docs/garden/getstarted
- git clone PX4-Autopilot : https://github.com/PX4/PX4-Autopilot
  git clone하여 repo 활용
- install MAVSDK : https://mavsdk.mavlink.io/main/en/python/quickstart.html
  (현재 코드에서는 ROS2 활용 안함)
  


활용법 : 
- main_qmix.py의 args에 PX4-Autopilot repo dir 입력
- main_qmix.py의 args에 mavsdk_server 파일 경로 입력
- python3 main_qmix.py 
