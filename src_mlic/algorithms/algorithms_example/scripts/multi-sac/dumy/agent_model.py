import rospy
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

from SAC_DUAL_Q_net import SAC
from cnn import ResizeNet



class AgentModel() : 
    def __init__(self, state_dim, action_dim, min_Val, Transition, learning_rate, capacity, gradient_steps,
                       batch_size, gamma, max_action, tau, device, agent_id, encoder):
        
        self.SAC = SAC(state_dim, action_dim, min_Val, Transition, learning_rate, capacity, gradient_steps,
                       batch_size, gamma, max_action, tau, device, agent_id)
        
        # 
        self.encoder = encoder
        
    # get_embedding : synchronous image embedding
    # state from _get_obs()
    # concatenate 
    def select_action(self, state):
        # concatenate state with img_embedding (Tensor)
        img_embedding = self.encoder.get_embedding(self.agent_id)
        state = torch.Tensor(state).to(self.device)
        modified_state = torch.cat((state, img_embedding), dim=1)
        return self.SAC.select_action(modified_state)


    def store(self, s, a, r, s_, d):
        index = self.num_transition % self.capacity
        transition = self.Transition(s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma*z.to(self.device))
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(self.device)) - torch.log(1 - action.pow(2) + self.min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        self.SAC.update()
        
    def save(self):
        self.SAC.save()


    def load(self):
        self.SAC.load()

        
if __name__ == "__main__" : 
    cnn_encoder = ResizeNet()
    model = AgentModel(cnn_encoder = cnn_encoder)
    
    