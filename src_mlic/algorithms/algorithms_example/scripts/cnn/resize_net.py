#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
from torch import optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
class DepthImageSubscriber:
    def __init__(self, robot_id, width, height):
        self.robot_id = robot_id
        self.desired_image_size = (width, height)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/iris_{}/camera/depth/image_raw".format(self.robot_id), Image, self._camera_depth_image_raw_callback, i) 
        
        rospy.logwarn("[DepthImageSubscriber] Check depth image iris_{}".format(robot_id))
        self._check_camera_depth_image_raw_ready()

    
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
    
    def _camera_depth_image_raw_callback(self, data, index):
        self.camera_depth_image[index] = self._image_to_cv2(data)
        #self.camera_depth_image[index] = data.data
        
'''
class ResizeNet(nn.Module):
    def __init__(self, in_channels, out_channels, new_width, new_height):
        super(ResizeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(8960,512)
        self.fc2 = nn.Linear(512,out_channels)
        self.relu = nn.ReLU()
        self.new_width = new_width
        self.new_height = new_height
        #self.optimizer = optim.Adam(self.parameters(),lr = 0.001)
        '''
        self.num_agents = num_agents       
        
        self.image_subscriber = [None]*self.num_agents
        for i in range(self.num_agents) : 
            self.image_subscriber[i] = DepthImageSubscriber(i)
        self.embedding = [None]*self.num_agents
        '''
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(self.new_width, self.new_height), mode='bilinear', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Resize the output to match the desired width and height
        
        return x
    '''
    def get_embedding(self) : 
        # synchronization
        return self.embedding
    '''

if __name__ == "__main__" : 
    x = np.random.normal(0,1,(160, 120)) 
    x = x.reshape((-1, 1, 160, 120))
    x = torch.Tensor(x).to(device)
    print(x.shape)
    model = ResizeNet(1, 48, 160, 120).to(device)
    
    for i in model.state_dict():
        print(i, '/t', model.state_dict()[i].size())
    y = model(x)
    print(y.shape)
    print(y)