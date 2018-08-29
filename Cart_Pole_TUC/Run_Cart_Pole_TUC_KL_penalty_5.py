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
#env = env.unwrapped

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
      
# KL divergence
def KL_divergence(mean_1,log_std_1,mean_2,log_std_2):    

      term_1 = np.sum(np.square(np.divide(np.exp(log_std_1),np.exp(log_std_2)))) 
      term_2 = np.sum(2*log_std_2-2*log_std_1)
      term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(np.exp(log_std_2))))
          
      return np.maximum(0,0.5*(term_1+term_2+term_2-1))   
      
      
# Session settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create agents and networks
PG_agent = PG(sess,"PG",4,2,0.02,0.95)
#PG_agent.save_model("./model_init/PG1")
PG_agent.load_model("./model_init/PG1")


# Create transition uncertainty critic
state_dim = 4
hidden_dim = 3
critic_hidden_dim = 2
action_dim = 2
tuc = TUC(sess,"TUC",state_dim,hidden_dim,critic_hidden_dim,action_dim,0.003) 
#tuc.save_model("./model_init/TUC1")
tuc.load_model("./model_init/TUC1")


# Run W/O TUC 
exp_results_filename = "PG_TUC_KL_penalty_5"
#exp_results_filename = "PG"
EPs_total_reward = []
ratio_init_1 = 1.9
ratio_init_2 = 5

mem_size = 5

states = []
next_states = []
actions = []
values = []

 
for EP in range(1,EPISODE+1):

    state = env.reset()
    done = 0
    t = 0
    EP_reward_sum = 0.

    ratio_1 = ratio_init_1*(EPISODE-EP)/EPISODE    
    ratio_2 = ratio_init_2*(EP)/EPISODE    
    
    states_mem = []
    actions_mem = []
    next_states_mem = []
    
    while not done: 

          #env.render()
          
    
          # Choose action    
          action = PG_agent.agent_choose_action(state)
            
          # Get state and reward
          next_state, reward, done, info = env.step(action)
          EP_reward_sum += reward

          # Store states 
          states.append(state) 

          # Store next states
          next_states.append(next_state)

          # Store actions
          temp_act = np.zeros([2])
          temp_act[action] = 1.
          actions.append(temp_act)  

          # Store reward
          values.append(reward)          

          # Time step
          t = t + 1
          
         
          # Get intrinsic reward
          if t <= mem_size :
             states_mem.append(state) 
             actions_mem.append(temp_act)
             next_states_mem.append(next_state)
             if t == mem_size :
                pre_z_mean, pre_z_std = tuc.dump_z_mean_std(state[np.newaxis,:],temp_act[np.newaxis,:])
             intrinsic_reward_1 = 0
             intrinsic_reward_2 = 0
          else : 
             tuc.train_tuc(np.vstack(states_mem), np.vstack(next_states_mem), np.vstack(actions_mem))  
             z_mean, z_std = tuc.dump_z_mean_std(state[np.newaxis,:],temp_act[np.newaxis,:])
             intrinsic_reward_1 = KL_divergence(pre_z_mean,pre_z_std,z_mean,z_std)/mem_size
             intrinsic_reward_2 = tuc.dump_regret(state,action)
             pre_z_mean = z_mean
             pre_z_std = z_std
             # Update memory
             states_mem.append(state)
             actions_mem.append(temp_act)
             next_states_mem.append(next_state)
             states_mem.pop(0)
             actions_mem.pop(0)
             next_states_mem.pop(0)
             
         
             
          # Store actions and states   
          #print(intrinsic_reward_1,intrinsic_reward_2)         
          PG_agent.agent_store_transition(state, action, reward+intrinsic_reward_1*ratio_1-intrinsic_reward_2*ratio_2)
        
          if t == MAX_TIMESTEP :
            done = 1
             
          if done:
            
            # Record total reward
            EPs_total_reward.append(EP_reward_sum)            

            
            states = np.vstack(states)
            next_states = np.vstack(next_states)
            actions = np.vstack(actions)
            values = get_return(values,0.95)
            values = np.vstack(values)
            
            for ve in range(CRITIC_EPOCHS):                  
                critic_loss = tuc.train_critic(states, values, actions)     
 
            states = []
            next_states = []
            actions = []
            values = []
            
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
   
            # Stop to train agent
            PG_agent.agent_REINFORCE()
           
            break
            
          # Enter next state
          state = next_state


with open("./Exp_Total_Reward/"+exp_results_filename,"wb") as fp:
        pickle.dump(EPs_total_reward,fp)

#PG_agent.save_model("./model_final/PG1_pure")
#PG_agent.save_model("./model_final/PG1_KL")
#tuc.save_model("./model_final/TUC1_KL")