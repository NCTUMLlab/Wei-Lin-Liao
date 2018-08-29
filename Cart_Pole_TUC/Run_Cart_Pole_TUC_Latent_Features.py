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

    
# Wasserstein-2 distance
def W2_distance(mean_1,std_1,mean_2,std_2):
      cov_1 = np.matmul(np.transpose(std_1),std_1)
      cov_2 = np.matmul(np.transpose(std_2),std_2)
      bias = 0*np.ones(np.shape(cov_1))
      term_1 = np.matmul(mean_1-mean_2,np.transpose(mean_1-mean_2))
      term_2 = std_1 + std_2 - 2*np.sqrt(np.multiply(np.multiply(np.sqrt(std_2),std_1),np.sqrt(std_2)))
      distance = abs(np.squeeze(term_1)) + abs(np.sum(term_2))
      distance = np.sqrt(distance)
      return distance

      
# KL divergence
def KL_divergence(mean_1,log_std_1,mean_2,log_std_2):    

      term_1 = np.sum(np.square(np.divide(log_std_1,log_std_2))) 
      term_2 = np.sum(2*log_std_2-2*log_std_1)
      term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(log_std_2)))
          
      return 0.5*(term_1+term_2+term_2-1)   
      
      
# Session settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create agents and networks
PG_agent = PG(sess,"PG",4,2,0.02,0.95)
PG_agent.load_model("./model_final/PG1_KL")


# Create recomposed transition critic
state_dim = 4
hidden_dim = 3
critic_hidden_dim = 2
action_dim = 2
tuc = TUC(sess,"TUC",state_dim,hidden_dim,critic_hidden_dim,action_dim,0.003) 
tuc.load_model("./model_final/TUC1_KL")

EPs_total_reward = []

states = []
next_states = []
actions = []
values = []

#EP = EPISODE 
for EP in range(1,2):

    state = env.reset()
    done = 0
    t = 0
    EP_reward_sum = 0.
    
    while not done: 

          # Choose action    
          action = PG_agent.agent_choose_action(state)
            
          # Get state and reward
          next_state, reward, done, info = env.step(action)
          EP_reward_sum += reward

          # Store states 
          states.append(state) 

          # Store actions
          temp_act = np.zeros([2])
          temp_act[action] = 1.
          actions.append(temp_act)         

          # Time step
          t = t + 1
 
          if t == MAX_TIMESTEP :
            done = 1
             
          if done:
            
            # Record total reward
            EPs_total_reward.append(EP_reward_sum)            

            
            states = np.vstack(states)
            actions = np.vstack(actions)
 
           
            TUC_z = tuc.dump_z(states, actions)
            with open("./Latent_Features/TUC_z_features","wb") as fp:
                 pickle.dump(TUC_z, fp)
            with open("./Latent_Features/TUC_z_actions","wb") as fp:
                 pickle.dump(actions, fp)
            with open("./Latent_Features/TUC_states","wb") as fp:
                 pickle.dump(states, fp)     
            states = []
            actions = []
          
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
           
            break
            
          # Enter next state
          state = next_state
