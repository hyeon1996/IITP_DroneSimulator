## Requirements
```
pip3 install mavsdk==2.1.0
```

## How to run
1. In px4 directory build px4_sitl
    ```
    make px4_sitl
    ```
2. spawn drones 
    ```
    PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0,1" PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 2 
    PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0,2" PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 3
    ...
    ```
3. run mavsdk server located in this directory for each drone
   ```
   ./mavsdk_server -p 50051 udp://:14541
   ./mavsdk_server -p 50052 udp://:14542
   ...
   ```
4. run `main.py`