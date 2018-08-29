import tensorflow as tf
import numpy as np
import gym
from random import randint
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as linalg
import scipy.io as sio
import scipy
from Agent import PG
from Agent import VIME
from Agent import TUC
import os
import math

# Device settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_mem_ratio = 0.2

# Experiment settings
MAX_TIMESTEP = 200
EPISODE = 130
WINDOW_SIZE = 40
TUC_EPOCHS = 3
CRITIC_EPOCHS = 3
EP_WINDOW_SIZE = 5

# Environment settings
#tf.set_random_seed(1)
np.random.seed(1)
env = gym.make('CartPole-v0')
#env.seed(1)     
env = env.unwrapped

# TUC returns normalization 
def get_return(reward_list,decay):
      discounted_values = np.zeros_like(reward_list)
      running_add = 0.
      for t in reversed(range(0, len(reward_list))):
           running_add = running_add * decay + reward_list[t]
           discounted_values[t] = running_add
      discounted_values -= 0.5*(np.max(discounted_values)+np.min(discounted_values))
      discounted_values /= np.max(discounted_values)-np.min(discounted_values)
      discounted_values += 0.5
      return discounted_values

      
# Experiment results moving average
def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l)
 
    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)
 
    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N
 
    return result    
      
      
      
# Session settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create agents and networks
PG_agent = PG(sess,"PG",4,2,0.02,0.95)
PG_agent.load_model("./model_final/PG1_KL")




state = env.reset()
done = 0
t = 0
EP_reward_sum = 0.
   
    
while not done: 

      env.render()
    
      # Choose action    
      action = PG_agent.agent_choose_action(state)
            
      # Get state and reward
      next_state, reward, done, info = env.step(action)
 
      # Enter next state
      state = next_state