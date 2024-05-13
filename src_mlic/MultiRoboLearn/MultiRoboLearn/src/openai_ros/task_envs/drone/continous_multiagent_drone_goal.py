import rospy
import numpy as np
import time
import math
#import tf
from gym import spaces
from openai_ros.robot_envs import multiagent_drone_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from pyquaternion import Quaternion


max_episode_steps_per_episode = 100 # Can be any Value

register(
        id='MultiagentDrone-v1',
        entry_point='openai_ros.task_envs.drone.continous_multiagent_drone_goal:MultiagentDroneEnv',
        max_episode_steps=max_episode_steps_per_episode,
    )

class MultiagentDroneEnv(multiagent_drone_env.MultiagentDroneEnv):
    def __init__(self):
        """
        This Task Env is designed for having the multi drone in some kind of scenarios.
        It will learn how to move around the desired point without crashing into static and dynamic obstacle.
        """
        rospy.logwarn("INIT task enviornment")
        # Only variable needed to be set here
        self.number_actions = rospy.get_param('/drone/n_actions')
        self.num_agents = rospy.get_param('/drone/n_agents')
        high = np.full((self.number_actions), 1.0)
        low = np.full((self.number_actions), -1.0)
        self.action_space = spaces.Box(low, high)

        # Maximum linear velocity (m/s) of Drone
        max_lin_vel = 20
        # Maximum angular velocity (rad/s) of Drone
        max_ang_vel = 3
        self.max_vel = np.array([max_lin_vel, max_ang_vel])
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        
        # Actions and Observations
        self.dec_obs = rospy.get_param("/drone/number_decimals_precision_obs", 3)
        self.linear_forward_speed = rospy.get_param('/drone/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/drone/linear_turn_speed')
        self.angular_speed = rospy.get_param('/drone/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/drone/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/drone/init_linear_turn_speed')
        
        
        self.n_observations = rospy.get_param('/drone/n_observations')
        self.n_scan_obs = rospy.get_param('/drone/n_scan_obs')
        self.min_range = rospy.get_param('/drone/min_range')
        # self.new_ranges = rospy.get_param('/drone/new_ranges')
        self.max_laser_value = rospy.get_param('/drone/max_laser_value')
        self.min_laser_value = rospy.get_param('/drone/min_laser_value')

        # Get Desired Point to Get for different robots

        self.iris_desired_point = [Point() for i in range(self.num_agents)]
        for i in range(self.num_agents) :
            self.iris_desired_point[i].x = rospy.get_param("/drone/iris_{}/desired_pose/x".format(i))
            self.iris_desired_point[i].y = rospy.get_param("/drone/iris_{}/desired_pose/y".format(i))
            self.iris_desired_point[i].z = rospy.get_param("/drone/iris_{}/desired_pose/z".format(i))

        super(MultiagentDroneEnv, self).__init__(self.num_agents)
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.

        laser_scans = self.get_laser_scan()
        rospy.logwarn("laser_scan len===>"+str(len(laser_scans[0].ranges)))
        
        # Laser data for different robots
        #self.laser_scan_frame = [ls.header.frame_id for ls in laser_scans]
        
        # Number of laser reading jumped
        interval = int(math.ceil(float(len(laser_scans[0].ranges)) / float(self.n_scan_obs)))
        self.new_ranges = {"min_angle" : math.pi*2 ,"max_angle" : math.pi *2, "interval" :interval}
        #int_front = int(math.ceil(float(len(laser_scans[0][0].ranges)) / float(self.n_front_obs)))
        #int_left = int(math.ceil(float(len(laser_scans[1][0].ranges)) / float(self.n_left_obs)))
        #int_right = int(math.ceil(float(len(laser_scans[2][0].ranges)) / float(self.n_right_obs)))
        #front_new_ranges = {"min_angle" : math.pi * 3/4, "max_angle" : math.pi * 5/4,"interval":int_front}
        #left_new_ranges = {"min_angle" : math.pi * 1/3, "max_angle" : math.pi * 5/3,"interval":int_left}
        #right_new_ranges = {"min_angle" : math.pi * 1/3, "max_angle" : math.pi * 5/3,"interval":int_right}
        # self.new_ranges = 1

        #rospy.logdebug("n_observations===>"+str(self.n_observations))
        #rospy.logdebug("new_ranges, jumping laser readings===>"+str(self.new_ranges))
        
        
        high = np.full((self.n_observations), self.max_laser_value)
        #in order to validate the observation data, we modify the min_laser_value into -self.max_laser_value as low
        low = np.full((self.n_observations), -1*self.max_laser_value)
        # low = np.full((self.n_observations), self.min_laser_value)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        
        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        #done for all robots
        # self._episode_dones = []
        
        # Rewards
        # self.forwards_reward = rospy.get_param("/drone/forwards_reward")
        # self.turn_reward = rospy.get_param("/drone/turn_reward")
        # self.end_episode_points = rospy.get_param("/drone/end_episode_points")

        self.cumulated_steps = 0.0
        self.laser_filtered_pub = [None]*self.num_agents
        for i in range(self.num_agents) : 
            self.laser_filtered_pub[i] = rospy.Publisher('/iris_{}/drone/laser/scan_filtered'.format(i), LaserScan, queue_size=10)
        rospy.logwarn("Task Env INIT finished")    
     
    '''
    def _set_init_pose(self, i):
        """Sets the Robot in its init pose
        """
    ''' 
        

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        
        '''
        status 
        idle : before takeoff, after landing
        takeoff : during takeoff
        running : actual flight
        landing : during landing
        '''
        
        # For Info Purposes,and total reward for all robots
        self.cumulated_reward = 0.0 #This only is put here, in fact, it is less useful.
        # self.cumulated_episode_reward = [0, 0, 0]

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self._episode_dones = [False]*self.num_agents
        self._if_dones_label = [False]*self.num_agents

        
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)
        
        # TODO: Add reset of published filtered laser readings
        #add
        #laser_scan = self.get_laser_scan()
        #print("laser for real robots", laser_scans)
        #discretized_ranges = [l.ranges for l in laser_scans]

        #add
        odometrys = self.get_odom()
        #print("odom for real robots", odometrys)
        print("odometrys is:", odometrys)
        self.previous_distance_from_des_points = [self.get_distance_from_desired_point(odometry.pose.pose.position, i) for i, odometry in enumerate(odometrys)]

    def _set_action(self, actions):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        # for i in range(self.num_agents) : 
        #     action = actions[i]
        #     # action = [x,y,z,yaw,land]    
        #     action = np.array(action)
        #     rospy.logdebug("Start Set Action for iris"+str(i)+"=>"+str(action))
        #     print(action)
        #     print(self.max_vel)
        #     linear_action = action[:2]*self.max_vel[0]
        #     angular_action = action[3]*self.max_vel[1]
        #     rospy.logdebug("agent{} action is:{}",i, [linear_action,angular_action])     
        #     # We tell drone the linear and angular speed to set to execute
        #     self.move_base_vel(i, linear_speed = linear_action.tolist(), angular_speed = angular_action)
        action = np.array(actions)
        rospy.logdebug("Start Set Action for iris"+str(0)+"=>"+str(action))
        linear_action = action[:3]*self.max_vel[0]
        angular_action = action[3]*self.max_vel[1]
        rospy.logdebug("agent{} action is:{}",0, [linear_action,angular_action])     
        # We tell drone the linear and angular speed to set to execute
        
        self.move_base_vel(0, linear_speed = linear_action.tolist(), angular_speed = angular_action)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data for all robots
        # pose, rot, lin_vel, ang_vel, image
        observations = []

        #add
        odometrys = self.get_odom()
        # odometrys = self.get_odom_spark()

        odometry_array = []
        for odom in odometrys :
            x_pos = odom.pose.pose.position.x
            y_pos = odom.pose.pose.position.y
            z_pos = odom.pose.pose.position.z
            
            quaternion = odom.pose.pose.orientation
            #qx = quaternion.x
            #qy = quaternion.y
            #qz = quaternion.z 
            #qw = quaternion.w
            #roll, pitch, yaw = tf.transformations.euler_from_quaternion([qx,qy,qz,qw])
            yaw, pitch, roll = self.q2angle(quaternion)
            
            x_vel = odom.twist.twist.linear.x
            y_vel = odom.twist.twist.linear.y
            z_vel = odom.twist.twist.linear.z
            yaw_vel = odom.twist.twist.angular.z
            odometry_array.append([x_pos, y_pos, z_pos, yaw, x_vel, y_vel, z_vel, yaw_vel])
        
        # observations = observations.append(discretized_laser_scan)
        # observations = observations.append(odometry_array)
        
        laser_scans = self.get_laser_scan()
        filtered_laser_scan = []
        #discretize laser date for different robots:
        for i in range(self.num_agents) : 
            discretized_obs = self.discretize_observation(laser_scans[i], self.new_ranges,i)
            # rospy.logwarn("discretized : "+str((np.array(discretized_obs)).shape))
            filtered_laser_scan.append(discretized_obs)
        
        # obtain laser data for all robots 
        
        '''
        depth_image_array = self.get_camera_depth_image_raw()
        '''
        for odom, filtered in zip(odometry_array, filtered_laser_scan) : 
            observations.append(np.array([*odom, *filtered]))

        rospy.logdebug("END Get Observation ==>"+str(np.array(observations).shape))

        # rospy.logdebug("Observations==>"+str(discretized_observations))
        # rospy.logdebug("AFTER DISCRET_episode_done==>"+str(self._episode_done))
        # rospy.logdebug("END Get Observation ==>")
        return np.array(observations)
        

    '''
    return done, info
    done : Episode ended from crashing, timeout, all tasks finished
    info : {success : [T/F] * num_agents, logs : ["agent 2 crash",]}
    '''
    
    def _is_position_valid(self, val, min_val, max_val) : 
        return val >= min_val and val <= max_val
        
    
    def _is_done(self, observations): # additional check on basis of action server (self.agent_done)
        #deciede per agent done and store in list
        info = {"success" : [False]*self.num_agents, "logs" : []}
        
        
        if self._episode_done : 
            print("All drone robots are Too Close or has crashed==>"+str(self._episode_dones))
            info["success"] = self._episode_dones
            return self._episode_dones, info
        else:
            # rospy.logerr("All drone robots are Ok ==>")

            # MAX_X = 10.0
            # MIN_X = -10.0
            # MAX_Y = 10.0
            # MIN_Y = -10.0
            # MAX_Z = 10
            # MIN_Z = -10.0
            MAX_X = 10.0
            MIN_X = -10.0
            MAX_Y = 10.0
            MIN_Y = -10.0
            MAX_Z = 10.0
            MIN_Z = -10.0
            
            current_position = [Point() for _ in range(self.num_agents)]
            # for i in range(self.num_agents) : 
            #     current_position[i].x = observations[i][0][0]
            #     current_position[i].y = observations[i][0][1]
            #     current_position[i].z = observations[i][0][2]
            current_position[0].x = observations[0][0]
            current_position[0].z = observations[0][1]
            current_position[0].y = observations[0][2]
                # We see if we are outside the Learning Space or get into desired points        
            for num, curr_pos in enumerate(current_position) :
                if self._episode_dones[num] is False:
                    is_valid = self._is_position_valid(current_position[0].x, MIN_X, MAX_X) and \
                        self._is_position_valid(current_position[0].y, MIN_Y, MAX_Y) and \
                        self._is_position_valid(current_position[0].z, MIN_Z, MAX_Z)
                    if is_valid : 
                        # We see if it got to the desired point
                        # if self.is_in_desired_position(self.desired_current_position[str(current_position)], current_position):
                        if self.is_in_desired_position(self.iris_desired_point[num], current_position[num]):
                            self._episode_done = True
                        else:
                            self._episode_done = False
                        # elif self.has_crashed[i] : 
                        #     self._episode_done = True
                        # else:
                        #     self._episode_done = False
                    else : 
                        rospy.logerr(f"iris {num} in invalid position")
                        self._episode_done = True
                        info["success"][num] = False
                    #print("goal_Env_done is:", self._episode_done)
                # sub_episode_done = sub_episode_done.append(self._episode_done)
                #     sub_episode_done.append(self._episode_done)
                    self._episode_dones[num] = self._episode_done
                else:
                    self._episode_dones[num] = True
            # self._episode_dones = sub_episode_done[:]
            # print("all robot dones are", self._episode_dones)

            #add
            # self._episode_dones[1] = True
            # self._episode_dones[2] = True

            return self._episode_dones, info


    # define reward for all robots through distance between each robot and desired point or has crashed into each other

    def _compute_reward(self, observations, dones):
        # define and store all reward for different robots
        reward_all = [0]*self.num_agents
        distance_from_des_points = []
        distance_differences = []
        
        current_position = [Point() for _ in range(self.num_agents)]
        for i in range(self.num_agents) : 
            current_position[i].x = observations[0][0]
            current_position[i].y = observations[0][1]
            current_position[i].z = observations[0][2]

            #obtain all robots given to the desired points
            #Agents are rewarded based on minimum agent distance to each desired point, penalized for collisions
            #establish reward for each robot and there are three conditions: each distance to desired point, 
            #all reached desired point and each crashed
            distance_from_des_point = self.get_distance_from_desired_point(current_position[i], i)
            distance_difference = distance_from_des_point - self.previous_distance_from_des_points[i]
            distance_from_des_points.append(distance_from_des_point)
            distance_differences.append(distance_difference)

        self.previous_distance_from_des_points = distance_from_des_points[:]
        # distance_difference = distance_from_des_point - self.previous_distance_from_des_point
                
        return reward_all


    # Internal TaskEnv Methods
    def discretize_observation(self,data,new_ranges,index):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        discretized_ranges = []
        min_angle = new_ranges["min_angle"]
        max_angle = new_ranges["max_angle"]
        interval = new_ranges["interval"]
        #mod = interval
        # mod = new_ranges # In the term of simulation
        
        max_laser_value = data.range_max
        min_laser_value = data.range_min
        
        #rospy.logdebug("data=" + str(data))
        #rospy.logwarn("mod=" + str(mod))
        
        # 0 ~ min, max ~ len(data.ranges)
        min_index = int(math.floor((min_angle / (math.pi * 2)) * len(data.ranges)))
        max_index = int(math.ceil((max_angle / (math.pi * 2)) * len(data.ranges)))
        
        for i, item in enumerate(data.ranges):
            if i > min_index and i < max_index : 
                continue
            else : 
                if (i%interval==0):
                    if item == float ('Inf') or np.isinf(item):
                        #discretized_ranges.append(self.max_laser_value)
                        discretized_ranges.append(round(max_laser_value,self.dec_obs))
                    elif np.isnan(item):
                        #discretized_ranges.append(self.min_laser_value)
                        discretized_ranges.append(round(min_laser_value,self.dec_obs))
                    else:
                        #discretized_ranges.append(int(item))
                        discretized_ranges.append(round(item,self.dec_obs))

        rospy.logdebug("Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))
        
        
        self.publish_filtered_laser_scan(laser_original_data=data, 
                                        new_filtered_laser_range=discretized_ranges,
                                        index=index)
        
        return discretized_ranges
        
    
    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range, index):
        
        rospy.logdebug("new_filtered_laser_range==>"+str(new_filtered_laser_range))
        
        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
        h.frame_id = laser_original_data.header.frame_id
        
        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        
        new_angle_incr = abs(laser_original_data.angle_max - laser_original_data.angle_min) / len(new_filtered_laser_range)
        
        #laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max
        
        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            # if item == 0.0:
            #     # laser_distance = 0.1
            #     laser_distance = 0.0
            # else:
            laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)
        
        
        self.laser_filtered_pub[index].publish(laser_filtered_object)

    

    def get_distance_from_desired_point(self, current_position, index): 
        distance = self.get_distance_from_point(current_position, self.iris_desired_point[index])
        return distance
        
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((float(pstart.x), float(pstart.y), float(pstart.z)))
        b = np.array((float(p_end.x), float(p_end.y), float(p_end.z)))

        distance = np.linalg.norm(a - b)

        return distance

    def is_in_desired_position(self, desired_point, current_position, epsilon=0.2):
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
