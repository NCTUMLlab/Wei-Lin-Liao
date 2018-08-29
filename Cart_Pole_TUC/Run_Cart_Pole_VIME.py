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


      
# Session settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create agents and networks
PG_agent = PG(sess,"PG",4,2,0.02,0.95)
#PG_agent.save_model("./model_init/PG1")
PG_agent.load_model("./model_init/PG1")


# Create recomposed transition critic
state_dim = 4
action_dim = 2
vime = VIME(sess,"VIME",state_dim,action_dim,10,0.003)
#vime.save_model("./model_init/VIME1")
vime.load_model("./model_init/VIME1")
mem_size = 5

# Run W/O TUC 
exp_results_filename = "PG_VIME"
EPs_total_reward = []
ratio_init_1 = 0.1

 
for EP in range(1,EPISODE+1):

    state = env.reset()
    done = 0
    t = 0
    EP_reward_sum = 0.

    ratio_1 = ratio_init_1*(EPISODE-EP)/EPISODE   
    
    states_actions_mem = []
    next_states_mem = []  
    
    while not done: 

          # Choose action    
          action = PG_agent.agent_choose_action(state)
            
          # Get state and reward
          next_state, reward, done, info = env.step(action)
          EP_reward_sum += reward
          
          # Store state and action
          temp_state_action = np.zeros([6])
          temp_state_action[0:state_dim] = state
          temp_state_action[state_dim:state_dim+action] = 1.
          
          # Time step
          t = t + 1
          
          # Get intrinsic reward
          if t <= mem_size :
             states_actions_mem.append(temp_state_action)
             next_states_mem.append(next_state)
             if t == mem_size :
                pre_hyper_parameters = vime.get_hyper_parameters()
             intrinsic_reward_1 = 0
          else : 
             vime.train(np.vstack(states_actions_mem), np.vstack(next_states_mem), 2)  
             hyper_parameters = vime.get_hyper_parameters()
             intrinsic_reward_1 = vime.get_info_gain(hyper_parameters, pre_hyper_parameters)   
             pre_hyper_parameters = hyper_parameters
             # Update memory
             states_actions_mem.append(temp_state_action)
             next_states_mem.append(next_state)
             states_actions_mem.pop(0)
             next_states_mem.pop(0)
             
             
          # Store actions and states        
          PG_agent.agent_store_transition(state, action, reward + intrinsic_reward_1*ratio_1)
        
          if t == MAX_TIMESTEP :
            done = 1
             
          if done:
            
            # Record total reward
            EPs_total_reward.append(EP_reward_sum)            
            
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
            # Stop to train agent
            PG_agent.agent_REINFORCE()
           
            break
            
          # Enter next state
          state = next_state

#'''
with open("./Exp_Total_Reward/"+exp_results_filename,"wb") as fp:
        pickle.dump(EPs_total_reward,fp)
#'''
PG_agent.save_model("./model_final/PG1_VIME")
vime.save_model("./model_final/VIME1")
