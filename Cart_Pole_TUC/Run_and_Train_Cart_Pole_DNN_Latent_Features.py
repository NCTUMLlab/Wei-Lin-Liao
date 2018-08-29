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
from Agent import DNN
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
np.random.seed(1)
env = gym.make('CartPole-v0')   
env = env.unwrapped

# Session settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create agents and networks
PG_agent = PG(sess,"PG",4,2,0.02,0.95)
PG_agent.load_model("./model_init/PG1")

dnn = DNN(sess,"DNN",[6,5,3,4,4],tf.nn.tanh,0.001)
dnn.save_model("./model_init/DNN")

EPISODE = 100
mem_size = 5
'''
# Train PG and DNN
for EP in range(1,EPISODE+1):

    state = env.reset()
    done = 0
    t = 0
    EP_reward_sum = 0.
    
    states_mem = []
    actions_mem = []
    next_states_mem = []
    
    while not done: 

          # Choose action    
          action = PG_agent.agent_choose_action(state)
            
          # Get state and reward
          next_state, reward, done, info = env.step(action)
          EP_reward_sum += reward


          # Store actions
          temp_act = np.zeros([2])
          temp_act[action] = 1.
      

          # Time step
          t = t + 1
          
          # Get intrinsic reward
          if t <= mem_size :
             states_mem.append(state) 
             actions_mem.append(temp_act)
             next_states_mem.append(next_state)
          else : 
             dnn.train(np.concatenate((np.vstack(states_mem),np.vstack(actions_mem)), axis = 1), np.vstack(next_states_mem))  
             # Update memory
             states_mem.append(state)
             actions_mem.append(temp_act)
             next_states_mem.append(next_state)
             states_mem.pop(0)
             actions_mem.pop(0)
             next_states_mem.pop(0)
             
         
             
          # Store actions and states          
          PG_agent.agent_store_transition(state, action, reward)
        
          if t == MAX_TIMESTEP :
            done = 1
             
          if done:
            
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
   
            # Stop to train agent
            PG_agent.agent_REINFORCE()
           
            break
            
          # Enter next state
          state = next_state

dnn.save_model("./model_final/DNN")
'''


# ==================================================================
# Create recomposed transition critic
dnn.load_model("./model_final/DNN")
PG_agent.load_model("./model_final/PG1_pure")
EPs_total_reward = []

states = []
next_states = []
actions = []
values = []

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
 
           
            features = dnn.dump_latent_features(np.concatenate((states,actions), axis = 1), 2)
            with open("./Latent_Features/DNN_features","wb") as fp:
                 pickle.dump(features, fp)
            with open("./Latent_Features/DNN_actions","wb") as fp:
                 pickle.dump(actions, fp)
            with open("./Latent_Features/DNN_states","wb") as fp:
                 pickle.dump(states, fp)     
            states = []
            actions = []
          
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
           
            break
            
          # Enter next state
          state = next_state
