#!/usr/bin/env python
import rospy
import actionlib
from openai_ros.msg import DroneAction, DroneFeedback, DroneResult
import sys

'''
Action server - task env

failsafe check
goal condition check

feedback : distance from goal 
other information that can be used for calculating reward

'''
from std_msgs.msg import Float64, Empty, String
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, ContactState, ModelState

import math
from pyquaternion import Quaternion


class DroneActionServer() : 
    #_feedback = DroneActionFeedback()
    #_result = DroneActionResult()
    
    def __init__(self, name, id, rate) : 
        self._action_name = name
        self.name = name
        self.id = id
        self._as = actionlib.SimpleActionServer(self._action_name, DroneAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        
        self.rate =rate
        self.odom = None
        #self.time_elapsed = 0
        self.goal_radius = 0.2 # if distance is smaller than 0.2, task succeess
        rospy.Subscriber("/{}/odom".format(self.name), Odometry, self._odom_callback)
        self.failsafe_angle = 45 # 45 degree
        self.takeoff_height = 1.3 # 1.3m
        # self.MAX_X = 10.0
        # self.MIN_X = -10.0
        # self.MAX_Y = 10
        # self.MIN_Y = -10.0
        # self.MAX_Z = 10.0
        # self.MIN_Z = -10.0
        self.MAX_X = 10.0
        self.MIN_X = -10.0
        self.MAX_Y = 10
        self.MIN_Y = -10.0
        self.MAX_Z = 2.5
        self.MIN_Z = 0
        self.has_crashed = False
        self.status = "idle"
        # colision subscribe
        rospy.Subscriber("/collision", ContactsState, self._collision_callback)
        
    def init_variable(self) :
        self.has_crashed = False
        self.status = "idle"
        
    def execute_cb(self, goal):
        r = rospy.Rate(self.rate)
        #success = False
        
        #execute the action
        success = False
        msg = ""
        result_type = 0

        target_pose = goal.target_pose
        cmd_mode = goal.cmd_mode
        self._result = DroneResult(result_type=3, msg="default")
        # 1. check invalid condition, abort before running goal
        if cmd_mode == 0 : 
            if self.status != "idle" : 
                self._as.set_aborted(self._result)
                return 
        
        if cmd_mode == 1 : 
            if self.status == "takeoff" :     
                self.status = "running"
            elif self.status == "running" : 
                pass
            else : # agent needs to takeoff
                self._as.set_aborted(self._result)
                return
        
        if cmd_mode == 2 :
            if self.status == "running" : 
                self.status = "landing"
            else : # takeoff or idle
                self._as.set_aborted(self._result)
                return
            
        if cmd_mode == 3 : 
            self.init_variable()
            self._result = DroneResult(result_type=3, msg="reset")
            self._as.set_aborted(self._result)
            return 
        
        # 2. feedback, result loop 
        while True:
            # if over max step, set server abort
            # when client preempt the request
            if self._as.is_preempt_requested() : 
                rospy.loginfo('{}: Preempted'.format(self._action_name))
                self._as.set_preempted(self._result)
                break
            
            if self.odom is None : 
                # it is checked before in env
                rospy.loginfo('[ActionServer] {} odom not published'.format(self.name))
                r.sleep()
                continue
            curr_pose = self.odom.pose.pose.position
            
            if cmd_mode == 0 : #takeoff detected
                if self.status == "idle" and curr_pose.z > 0.8 : 
                    self.status = "takeoff"
            if cmd_mode == 0 : 
                distance_to_target = target_pose.position.z - curr_pose.z
            else : 
                distance_to_target = self.distance(curr_pose, target_pose.position)
            yaw, pitch, roll = self.q2angle(self.odom.pose.pose.orientation)
            
            is_valid = self._is_position_valid(curr_pose.x, self.MIN_X, self.MAX_X) and \
                        self._is_position_valid(curr_pose.y, self.MIN_Y, self.MAX_Y) and \
                        self._is_position_valid(curr_pose.z, self.MIN_Z, self.MAX_Z)
            # 3. failsafe condition
            ## 3.0 check valid position
            if not is_valid : 
                success = True
                result_type = 2 
                msg = "failsafe_invalid_pos"
                break   

            ## 3.1 too high pitch, angle
            if abs(pitch) > self.failsafe_angle or abs(roll) > self.failsafe_angle : 
                success = True
                result_type = 2 
                msg = "failsafe_angle"
                break
            
            ## 3.2 crashed
            if self.has_crashed : 
                success = True
                result_type = 2
                msg = "failsafe_crashed"
                break

            if distance_to_target < self.goal_radius : 
                success = True
                result_type = 1 
                msg = "target_reached"
                break
            
            # generate feedback
            self._feedback = DroneFeedback(status=self.status, target_dist=distance_to_target)
            self._as.publish_feedback(self._feedback)
            
            r.sleep()
    
        if success : 
            #generate result
            self._result = DroneResult(result_type = result_type, msg = msg)
            if result_type == 2 :
                rospy.loginfo('{}: {} activated'.format(self._action_name,msg))
            elif result_type == 1:
                rospy.loginfo('{}: task completed'.format(self._action_name))
                if cmd_mode == 0 : # takeoff finished
                    self.status = "running"
                # running finished : still running 
                # landing finished : check by ground collision 
                
            self._as.set_succeeded(self._result)
        else : #preempted finished from timeover
            rospy.loginfo('{}: timeover'.format(self._action_name))
    
    def _collision_callback(self, data):
        states = data.states
        
        if len(states) > 0 : 
            collision_list = []
            for state in states : 
                assert("iris_" in state.collision1_name)
                collision_list.append(state.collision2_name)
            agent_id = int(states[0].collision1_name[5])    
            
            if self.id == agent_id and not self.has_crashed : # when it was first contact
                tmp = 0
                for cl in collision_list : 
                    #if exists other than ground 
                    if "ground_plane" in cl :
                       tmp+=1 
                if tmp == len(collision_list) : 
                    # filter collision after landing, before takeoff
                    if self.status == 'idle': # before takeoff
                        self.has_crashed = False
                        return
                    if self.status == 'landing': # after landing
                        rospy.logdebug("[ActionServer] {} landed".format(self.name))
                        self.has_crashed = False
                        self.status = 'idle'
                        return
                self.has_crashed = True
                        
            
                
        
    def _odom_callback(self, data):
        self.odom = data
        
    def _is_position_valid(self, val, min_val, max_val) : 
        return val >= min_val and val <= max_val
    
    def distance(self, p1, p2) : 
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def q2angle(self, q) : 
        if isinstance(q, Quaternion) : 
            rotate_rad = q.yaw_pitch_roll
        else : 
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_rad = q_.yaw_pitch_roll
        return rotate_rad
    
if __name__ == '__main__' : 
    #get vehicle, id as argument 
    vehicle = sys.argv[1]
    id = sys.argv[2]
    rospy.init_node('drone_action_server_' + vehicle + "_" + id, log_level=rospy.DEBUG)
    name = vehicle + "_" + id
    server = DroneActionServer(name, int(id), 10)
    rospy.spin()
