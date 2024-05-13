import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64, Empty, String
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point
from gazebo_msgs.msg import ContactsState, ContactState, ModelState
from gazebo_msgs.srv import SetModelState
import cv2
from cv_bridge import CvBridge, CvBridgeError

import math
from pyquaternion import Quaternion

from openai_ros.drone_action import DroneActionClient
from openai_ros.msg import DroneResult, DroneGoal
import roslaunch

import asyncio as aio
import subprocess

class MultiagentDroneEnv(robot_gazebo_env.RobotGazeboEnv):
    
    def __init__(self, num_agents):
        """
        Initializes a new multi-DroneEnv environment.
        Turtlebot2 doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings
        
        Actuators Topic List: 
        * /cmd_vel
        * /takeoff
        * /land
         
        
        Args:
        """
        rospy.logwarn("Start DroneEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.num_agents = num_agents
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(MultiagentDroneEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")


        self.iris_init_point = [Point() for i in range(self.num_agents)]
        for i in range(self.num_agents) : 
            self.iris_init_point[i].x = rospy.get_param('/drone/iris_{}/init_pose/x'.format(i))
            self.iris_init_point[i].y = rospy.get_param('/drone/iris_{}/init_pose/y'.format(i))
            self.iris_init_point[i].z = rospy.get_param('/drone/iris_{}/init_pose/z'.format(i))
        
        self.takeoff_height = 1.0
        
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        
        # publish to client
        self._client_pub = [None]*self.num_agents
        
        #px4 instance
        self.px4_dir = "/home/user/PX4_Firmware/"
        self.px4_launch = [None]*self.num_agents
        self.agents_alive = [True]*self.num_agents
        self.agents_done = [False]*self.num_agents
        self.server_ready = [False]*self.num_agents
        self._check_all_sensors_ready()
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # We Start all the ROS related Subscribers and publishers
        # # self.has_crashed = [False]*self.num_agents
        
        # failsafe detection is done by action server
        
        self._cmd_pos_pub = [None]*self.num_agents
        self._cmd_vel_pub = [None]*self.num_agents
        self._cmd_pub = [None]*self.num_agents
        
        
        #self.buff_size = 3
        #self.past_frames = [] * self.buff_size
        
        for i in range(self.num_agents) : 
            rospy.Subscriber("/iris_{}/odom".format(i), Odometry, self._odom_callback, i)
            '''
            rospy.Subscriber("/iris_{}/stereo_camera/left/image_raw/compressed".format(i), CompressedImage, self._camera_rgb_image_left_raw_callback, i)
            rospy.Subscriber("/iris_{}/stereo_camera/right/image_raw/compressed".format(i), CompressedImage, self._camera_rgb_image_right_raw_callback, i)
            '''
            #rospy.Subscriber("/iris_{}/camera/depth/image_raw".format(i), Image, self._camera_depth_image_raw_callback, i) 
            
            rospy.Subscriber("/iris_{}/scan".format(i), LaserScan, self._laser_scan_callback, i)
            rospy.Subscriber("/droneclient/iris_{}/result".format(i), DroneResult, self._action_client_callback, i)
            self._client_pub[i] = rospy.Publisher("/droneclient/iris_{}/goal".format(i), DroneGoal, queue_size=1)
            #rospy.Subscriber("/iris_{}/scan_left".format(i), LaserScan, self._laser_scan_left_callback, i)
            #rospy.Subscriber("/iris_{}/scan_right".format(i), LaserScan, self._laser_scan_right_callback, i)
            self._cmd_vel_pub[i] = rospy.Publisher("/xtdrone/iris_{}/cmd_vel_flu".format(i), Twist, queue_size=10)
            self._cmd_pos_pub[i] = rospy.Publisher("/xtdrone/iris_{}/cmd_pose_enu".format(i), Pose, queue_size=10)
            self._cmd_pub[i] = rospy.Publisher("/xtdrone/iris_{}/cmd".format(i), String, queue_size=3)
            
        self._check_all_publishers_ready()

        self.gazebo.pauseSim()

        
        rospy.logdebug("Finished MultiDroneEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        
        result = self.takeoff() #if takeoff fails, reset

        return result

        

    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
            
        odom = self._check_odom_ready()
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = [None]*self.num_agents
        rospy.logdebug("Waiting for /odom to be READY...")
        while not all(i is not None for i in self.odom) and not rospy.is_shutdown():
            try:
                # self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/odom".format(i)
                    self.odom[i] = rospy.wait_for_message(topic_name, Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")
        #rospy.logdebug(self.odom[0])
        # return self.odom
     
        return self.odom
    
    def _image_to_cv2(self, image) : 
        cv_image = self.bridge.imgmsg_to_cv2(image, "32FC1")
        cv_image_array = np.array(cv_image, dtype=np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        cv_image_resized = cv2.resize(cv_image_norm, self.desired_image_size, interpolation = cv2.INTER_CUBIC)
        return cv_image_resized
        
    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image = [None]*self.num_agents
        rospy.logdebug("Waiting for /camera/depth/image_raw to be READY...")
        while not all(img is not None for img in self.camera_depth_image) and not rospy.is_shutdown():
            try:
                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/camera/depth/image_raw".format(i)
                    tmp = rospy.wait_for_message(topic_name, Image, timeout=5.0)
                    self.camera_depth_image[i] = self._image_to_cv2(tmp)
                    rospy.logdebug("iris_{} ok".format(i))
                rospy.logdebug("Current /camera/depth/image_raw READY=>")
    
            except:
                rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting camera_depth_image_raw")
       
        
        tmp = self.camera_depth_image[0]
        #h, w = tmp.shape
        #self.depth_image_info = {"width" : w, "height" : h} 
        #rospy.logdebug(self.depth_image_info)
        #self.camera_depth_image = [raw.data for raw in self.camera_depth_image]
        return self.camera_depth_image

        
    # def _check_camera_depth_points_ready(self):
    #     self.camera_depth_points = None
    #     rospy.logdebug("Waiting for /camera/depth/points to be READY...")
    #     while self.camera_depth_points is None and not rospy.is_shutdown():
    #         try:
    #             self.camera_depth_points = rospy.wait_for_message("/camera/depth/points", PointCloud2, timeout=10.0)
    #             rospy.logdebug("Current /camera/depth/points READY=>")
    #
    #         except:
    #             rospy.logerr("Current /camera/depth/points not ready yet, retrying for getting camera_depth_points")
    #     return self.camera_depth_points

    
    # from torchrl.envs.transforms import (
    # TransformedEnv,
    # Transform,
    # Compose,
    # FlattenObservation,
    # CatTensors
    # )
    # class DepthImageNorm(Transform):
    # def __init__(
    #     self,
    #     in_keys: Sequence[str],
    #     min_range: float,
    #     max_range: float,
    #     inverse: bool=False
    # ):
    #     super().__init__(in_keys=in_keys)
    #     self.max_range = max_range
    #     self.min_range = min_range
    #     self.inverse = inverse

    # def _apply_transform(self, obs: torch.Tensor) -> None:
    #     obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
    #     obs = obs.clip(self.min_range, self.max_range)
    #     if self.inverse:
    #         obs = (obs - self.min_range) / (self.max_range - self.min_range)
    #     else:
    #         obs = (self.max_range - obs) / (self.max_range - self.min_range)
    #     return obs    
        
    def _check_camera_rgb_image_raw_ready(self):
        #self.camera_rgb_image_raw = [[None, None], [None, None], [None, None], [None, None]]
        self.camera_rgb_image_raw = [None]*self.num_agents
        rospy.logdebug("Waiting for /camera/rgb/image_raw to be READY...")
        while all(i[0] is None and i[1] is None for i in self.camera_rgb_image_raw) and not rospy.is_shutdown():
            try:
                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/stereo_camera/left/image_raw/compressed".format(i)
                    self.camera_rgb_image_raw[i] = rospy.wait_for_message(topic_name, CompressedImage, timeout=5.0)
                    #topic_name = "/iris_" + str(i) + "/stereo_camera/right/image_raw/compressed"
                    #self.camera_rgb_image_raw[i][1] = rospy.wait_for_message(topic_name, CompressedImage, timeout=5.0)
                rospy.logdebug("Current /camera/rgb/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/rgb/image_raw not ready yet, retrying for getting camera_rgb_image_raw")
                
        return self.camera_rgb_image_raw
        

    def _check_laser_scan_ready(self):
        self.laser_scan = [None]*self.num_agents
        #self.laser_scan_left = [None]*self.num_agents
        #self.laser_scan_right = [None]*self.num_agents
        rospy.logdebug("Waiting for /iris/scan to be READY...")

        while all(l is None for l in self.laser_scan) is None and not rospy.is_shutdown():
            
            try:
                # self.laser_scan = rospy.wait_for_message("/kobuki/laser/scan", LaserScan, timeout=5.0)
                # self.laser_scan_marobot1 = rospy.wait_for_message("marobot1/kobuki/laser/scan", LaserScan, timeout=5.0)

                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/scan".format(i)
                    self.laser_scan = rospy.wait_for_message(topic_name, LaserScan, timeout=5.0)
                rospy.logdebug("Current /iris/scan READY=>")

            except:
                rospy.logerr("Current /iris/scan not ready yet, retrying for getting laser_scan")
            '''
            try:
                # self.laser_scan = rospy.wait_for_message("/kobuki/laser/scan", LaserScan, timeout=5.0)
                # self.laser_scan_marobot1 = rospy.wait_for_message("marobot1/kobuki/laser/scan", LaserScan, timeout=5.0)

                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/scan_left".format(i)
                    self.laser_scan_left = rospy.wait_for_message(topic_name, LaserScan, timeout=5.0)
                rospy.logdebug("Current /iris/scan_left READY=>")

            except:
                rospy.logerr("Current /iris/scan_left not ready yet, retrying for getting laser_scan")
            
            try:
                # self.laser_scan = rospy.wait_for_message("/kobuki/laser/scan", LaserScan, timeout=5.0)
                # self.laser_scan_marobot1 = rospy.wait_for_message("marobot1/kobuki/laser/scan", LaserScan, timeout=5.0)

                for i in range(self.num_agents) : 
                    topic_name = "/iris_{}/scan_right".format(i)
                    self.laser_scan_right = rospy.wait_for_message(topic_name, LaserScan, timeout=5.0)
                rospy.logdebug("Current /iris/scan_right READY=>")

            except:
                rospy.logerr("Current /iris/scan_right not ready yet, retrying for getting laser_scan")
            '''
        return self.laser_scan
                
            
    def _action_client_callback(self, result, index) : 
        #result.msg 
        rospy.logwarn("[iris_{}] result_type : {} {}".format(index, result.result_type, result.msg))
        if result.result_type == 3 :
            if result.msg == "aborted" :
                rospy.logwarn("[iris_{}] action client callback : goal aborted".format(index))
                # invalid cmd detected, individual reset
            elif result.msg == "reset" : # reset
                self.server_ready[index] = True
                rospy.logwarn("[iris_{}] action client callback : server reset".format(index))
                return
            
        elif result.result_type != 1 : # when failsafe, time over
            rospy.logwarn("[iris_{}] action client callback : {}, need reset".format(index, result.msg))
            #self.agents_alive[index] = False 
            # individual reset
        self.agents_done[index] = True
    
            
    def _odom_callback(self, data, index):
        self.odom[index] = data

    
    def _camera_depth_image_raw_callback(self, data, index):
        self.camera_depth_image[index] = self._image_to_cv2(data)
        #self.camera_depth_image[index] = data.data

    
    # def _camera_depth_points_callback(self, data):
    #     self.camera_depth_points = data
    #
    def _camera_rgb_image_raw_callback(self, data, index):
        self.camera_rgb_image_raw[index] = data
     
    #def _camera_rgb_image_raw_right_callback(self, data, index):
    #    self.camera_rgb_image_raw[index] = data
        
    def _laser_scan_callback(self, data, index):
        self.laser_scan[index] = data

    def _laser_scan_left_callback(self, data, index):
        self.laser_scan[index] = data
        
    def _laser_scan_right_callback(self, data, index):
        self.laser_scan[index] = data


        
    def _check_all_publishers_ready(self) : 
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("START ALL SENSORS READY")
        self._check_client_pub_connection()
        self._check_cmd_vel_pub_connection()
        self._check_cmd_pos_pub_connection()
        self._check_cmd_pub_connection()
        
        rospy.logdebug("ALL SENSORS READY")
        
    def _check_client_pub_connection(self):
        rate = rospy.Rate(10)  # 10hz
        for i in range(self.num_agents) : 
            while self._client_pub[i].get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug("No susbribers to _client_pub yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
                    
            rospy.logdebug("iris_{}_client_pub Publisher Connected".format(i))

        rospy.logdebug("All Publishers READY")
        
    def _check_cmd_vel_pub_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        # while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        for i in range(self.num_agents) : 
            while self._cmd_vel_pub[i].get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
                    
            rospy.logdebug("iris_{}_cmd_vel_pub Publisher Connected".format(i))

        rospy.logdebug("All Publishers READY")
        
    def _check_cmd_pos_pub_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        # while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        for i in range(self.num_agents) : 
            while self._cmd_pos_pub[i].get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug("No susbribers to _cmd_pos_pub yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
                    
            rospy.logdebug("iris_{}_cmd_pos_pub Publisher Connected".format(i))

        rospy.logdebug("All Publishers READY")
    
    def _check_cmd_pub_connection(self):

        rate = rospy.Rate(10)  # 10hz
        for i in range(self.num_agents) :
            while self._cmd_pub[i].get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug("No susbribers to _takeoff_pub yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
            rospy.logdebug("iris_{}_cmd_pub Publisher Connected".format(i))

        rospy.logdebug("All Publishers READY")
        
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    
    def _individual_reset(self, index) :
        reset_goal = DroneGoal()
        reset_goal.cmd_mode = 3
        self._client_pub[index].publish(reset_goal)# stop action server
        rospy.logwarn("individual reset iris_{}".format(index))
        '''
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() : 
            rospy.logwarn("waiting server ready iris_{}".format(index))
            if self.server_ready[index] :
                break
            rate.sleep()
        '''
        time.sleep(2)
        rospy.logwarn("server ready iris_{} : {}".format(index, self.server_ready[index]))
        self.stop_px4(index)
        self._set_individual_init_pose(index)
        self.start_px4(index) # to stop moving
        self.server_ready[index] = False
        self.agents_done[index] = False
        time.sleep(0.5)
        


    def _set_individual_init_pose(self, index):
        self.gazebo.pauseSim()
        rospy.wait_for_service('/gazebo/set_model_state')
        init_pose = ModelState()
        init_pose.model_name = "iris_{}".format(index)
        init_position = init_pose.pose.position 
        init_position.x = self.iris_init_point[index].x
        init_position.y = self.iris_init_point[index].y
        init_position.z = self.iris_init_point[index].z
        self.set_state(init_pose)
        rospy.logwarn('iris_{} position reset'.format(index))
        
        
        '''
        # publish landing and zero speed
        land_cmd = String()
        land_cmd.data = "AUTO.LAND"
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = 0.0
        cmd_vel_value.linear.y = 0.0
        cmd_vel_value.linear.z = 0.0
        cmd_vel_value.angular.x = 0.0
        cmd_vel_value.angular.y = 0.0
        cmd_vel_value.angular.z = 0.0
        self._cmd_vel_pub[index].publish(cmd_vel_value)
        self._cmd_pub[index].publish(land_cmd)
        '''
        self.gazebo.unpauseSim()
        
    def _set_init_pose(self):
        for i in range(self.num_agents) : 
            self._individual_reset(i)
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    
    # landing, takeoff and waiting should not be blocking
    # implemented with rosaction
    '''
        robot_env : rosaction server
        
        when get takeoff / land order, each client connect to server

        server, client asynchronously check for the goal (non blocking)

        if goal is finished, rosaction connection ends
    
    '''
    
    def start_px4(self, index) : 
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.px4_launch[index] = roslaunch.parent.ROSLaunchParent(uuid, [self.px4_dir + "launch/connect/connect_sitl_iris_{}.launch".format(index)])
        self.px4_launch[index].start()
        rospy.logdebug("iris_{} px4 launched".format(index))        
    
    def stop_px4(self, index) : 
        if self.px4_launch[index] is not None : 
            self.px4_launch[index].shutdown()
        rospy.logdebug("iris_{} px4 shutdown".format(index))        
    
    
    def hover(self,index):
        hover_cmd = String()
        hover_cmd.data = "HOVER"
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = 0.0
        cmd_vel_value.linear.y = 0.0
        cmd_vel_value.linear.z = 0.0
        cmd_vel_value.angular.x = 0.0
        cmd_vel_value.angular.y = 0.0
        cmd_vel_value.angular.z = 0.0
        self._cmd_pub[index].publish(hover_cmd)
        self._cmd_vel_pub[index].publish(cmd_vel_value)
        
    def arm_offboard(self, i) : 
        takeoff_cmd = String()
        takeoff_cmd.data = "ARM"
        self._cmd_pub[i].publish(takeoff_cmd)
    
        takeoff_cmd = String()
        takeoff_cmd.data = "OFFBOARD"
        self._cmd_pub[i].publish(takeoff_cmd)
        
    def takeoff(self): 
        """
        Sends the takeoff command and checks it has taken of
        It unpauses the simulation and pauses again
        to allow it to be a self contained action
        """
        
        self.gazebo.unpauseSim()
        self._check_cmd_pub_connection()
        self._check_client_pub_connection()
        
        for i in range(self.num_agents) :
            init_x = self.iris_init_point[i].x
            init_y = self.iris_init_point[i].y
            init_z = self.iris_init_point[i].z
            target = Pose()
            target.position.x = init_x
            target.position.y = init_y
            target.position.z = init_z + self.takeoff_height
            rospy.logwarn("{} {} {}".format(target.position.x, target.position.y, target.position.z))
            agent_goal = DroneGoal(cmd_mode=0, target_pose = target)
            self._client_pub[i].publish(agent_goal)   
            
            #time.sleep(0.2)         
        # wait until all goals reached, need timeout
        #time.sleep(3)
        '''
        for i in range(self.num_agents) : 
            for _ in range(2) : 
                self.arm_offboard(i)
                #time.sleep(0.2)
        '''
        hz = 10
        rate = rospy.Rate(hz)
        num = 0
        timeover = 20 # 10seconds
        result = False
        while not rospy.is_shutdown():
            rospy.logwarn("takeoff : {}".format(num))
            for i in range(self.num_agents) : 
                if num < 50 : 
                    self.arm_offboard(i)
                    # self.agents_done[0] = True
                self.move_base_vel(i, [0, 0, 0.8], 0)
               #  self.move_base_pos(i,[target.position.x, target.position.y, target.position.z],0)
            rospy.logwarn("agents done : {}".format(self.agents_done))
            if all(self.agents_done) :
                result = True
                break
            
            if num > hz * timeover : 
                break
            
            num += 1
            rate.sleep()
        
        self.gazebo.pauseSim()
        
        return result
        
    def land(self, index):
        """
        Sends the Landing command and checks it has landed
        It unpauses the simulation and pauses again
        to allow it to be a self contained action
        """
        self.gazebo.unpauseSim()
        
        self._check_cmd_pub_connection()
    
        
        # drone should be hovering before landing
        self.wait_for_stop(index, update_rate = 10)
        
        land_cmd = String()
        land_cmd.data = "AUTO.LAND"
        self._cmd_pub[index].publish(land_cmd)
        # waiting should be asynchronous
        self.wait_for_stop(index, update_rate = 10, epsilon = 0.05) #it should be on the floor
        self.gazebo.pauseSim()

    def move_base_pos(self, index, pos, yaw) : 
        cmd_pos_value = Pose()
        cmd_pos_value.position.x = pos[0]
        cmd_pos_value.position.y = pos[1]
        cmd_pos_value.position.z = pos[2]
        cmd_pos_value.orientation = self.yaw2q(yaw)
        
        rospy.logdebug("iris_{} Base Pose Cmd>>{}".format(index, cmd_pos_value))
        #self._check_all_publishers_ready()
        self._cmd_pos_pub[index].publish(cmd_pos_value)
        #time.sleep(0.5)
    
    def move_base_vel(self, index, linear_speed, angular_speed):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed[0]
        cmd_vel_value.linear.y = linear_speed[1]
        cmd_vel_value.linear.z = linear_speed[2]
        cmd_vel_value.angular.x = 0.0
        cmd_vel_value.angular.y = 0.0
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("iris_{} Base Twist Cmd>>{}".format(index, cmd_vel_value))
        #self._check_all_publishers_ready()
        self._cmd_vel_pub[index].publish(cmd_vel_value)
        #time.sleep(0.02)

    def close(self):
        rospy.logdebug("Closing Drone Action Server")
        # close created action server process
        
        
        rospy.logdebug("Closing MultiagentDroneEnvironment")
        rospy.signal_shutdown("Closing MultiagentDroneEnvironment")

    def get_odom(self):
        return self.odom
        
    def get_camera_depth_embedding(self):
        return self.camera_depth_embedding
    
    # def get_camera_depth_points(self):
    #     return self.camera_depth_points
    
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw
        
    def get_laser_scan(self):
        #return [self.laser_scan, self.laser_scan_left, self.laser_scan_right]
        return self.laser_scan

    def q2angle(self, q) : 
        if isinstance(q, Quaternion) : 
            rotate_rad = q.yaw_pitch_roll
        else : 
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_rad = q_.yaw_pitch_roll
        return rotate_rad

    def yaw2q(self, yaw) : #yaw in degree
        q_ = Quaternion(axis=[0.0,0.0,1.0], degrees=yaw)
        return q_

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and 
        
        """
        
        
