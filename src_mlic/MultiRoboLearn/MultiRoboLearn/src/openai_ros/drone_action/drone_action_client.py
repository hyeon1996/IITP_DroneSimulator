#!/usr/bin/env python
import rospy
import actionlib
from openai_ros.msg import DroneAction, DroneGoal, DroneResult
#from actionlib_msgs.msg import GoalStatus

import sys
import roslaunch
import asyncio as aio

from std_msgs.msg import Float64, Empty, String
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, ContactState, ModelState
from openai_ros.gazebo_connection import GazeboConnection
'''
Action client

non blocking execution while server checks failsafe / goal condition


'''

class DroneActionClient() : 
    def __init__(self, name) :         
        rospy.logdebug("INIT Drone Client {}".format(name))
        self.client = actionlib.SimpleActionClient(name, DroneAction)
        self.px4_dir = "/home/user/PX4_Firmware/"
        #self.gazebo = gazebo
        self.name = name
        #spawn vehicle 
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.px4_dir + "launch/spawn/spawn_{}.launch".format(self.name)])
        launch.start()
        
        if self.client.wait_for_server() : 
            rospy.logdebug("Drone Client {} Connected".format(name))
        else : 
            rospy.logdebug("Drone Client {} Failed to connect action server".format(name))
        
        x = rospy.get_param('/drone/{}/desired_pose/x'.format(name))
        y = rospy.get_param('/drone/{}/desired_pose/y'.format(name))
        z = rospy.get_param('/drone/{}/desired_pose/z'.format(name))
        self.goal_position = [x, y, z]
        x = rospy.get_param('/drone/{}/init_pose/x'.format(name))
        y = rospy.get_param('/drone/{}/init_pose/y'.format(name))
        z = rospy.get_param('/drone/{}/init_pose/z'.format(name))
        self.start_position = [x, y, z]
        self.result = None
    
        self.done_pub = rospy.Publisher("/droneclient/{}/result".format(self.name), DroneResult, queue_size=1)
        rospy.Subscriber("/droneclient/{}/goal".format(self.name),DroneGoal,self._robot_env_callback)     

    
    def run(self, cmd_mode, target, timeout=None) : 
        
        target_pose = Pose()
        if target is not None : 
            target_pose.position.x = target[0]
            target_pose.position.y = target[1]
            target_pose.position.z = target[2]

        goal = DroneGoal(cmd_mode = cmd_mode, target_pose = target_pose)
        self.client.send_goal(goal, active_cb =self.callback_active, feedback_cb=self.callback_feedback, done_cb = self.callback_done)
        if timeout is not None : 
            self.client.wait_for_result(timeout=rospy.Duration(timeout))
        else : 
            self.client.wait_for_result()
        result = self.client.get_result()
        rospy.logdebug("[ActionClient {}] result received".format(self.name))
        return result
    
    def callback_active(self) : 
        rospy.logdebug("[ActionClient {}] Action server start processing".format(self.name))
    
    def callback_done(self, state, result) : 
        rospy.logdebug("[ActionClient {}] Action server done. State : {}, Result : {}".format(self.name, state, result))
        done_msg = DroneResult()
        if state == 4 : # aborted 
            done_msg.result_type = 3
            if result.msg == "reset" : 
                done_msg.msg = "reset"
            else : 
                done_msg.msg = "aborted"
            self.done_pub.publish(done_msg)
        elif state == 3 : #succeeded 
            done_msg.result_type = result.result_type
            done_msg.msg = result.msg
            self.done_pub.publish(done_msg)
        
    def callback_feedback(self, feedback) : 
        # pass feedback to reward
        rospy.logdebug("[ActionClient {}] Feedback : {}".format(self.name, feedback))
    
    def _robot_env_callback(self, goal) : 
        timeout = None
        '''
        if goal.cmd_mode == 0 : 
            timeout = rospy.Duration(10) #takeoff timeout 10
        '''                
        target = None 
        if goal.target_pose is not None : 
            target = [goal.target_pose.position.x, goal.target_pose.position.y, goal.target_pose.position.z]
        
        result = self.run(goal.cmd_mode, target, timeout)
        #self.done_pub.publish(result)
        
        '''
        if goal.cmd_mode == 3 : 
            self.client.cancel_all_goals() #after reset server, cancel all goals
            rospy.logdebug("[ActionClient {}] Goal all canceled".format(self.name))
        '''
        
        
        
if __name__ == '__main__':
    vehicle = sys.argv[1]
    id = int(sys.argv[2])
    name = vehicle + "_" +str(id)
    rospy.init_node('drone_action_client_'+name, log_level=rospy.DEBUG)
    
    #gazebo = GazeboConnection(False,"WORLD")
    #da = DroneActionClient(name=name,gazebo=gazebo)
    da = DroneActionClient(name=name)
    # Initializes a rospy node so that the SimpleActionClient can publish and subscribe over ROS.
    
    rospy.spin()
