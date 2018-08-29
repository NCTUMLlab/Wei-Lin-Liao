import copy
import pylab
import numpy as np
import tensorflow as tf
from Environment import Env
from Agent import PG
from Agent import TUC
import pickle

np.random.seed(0)
EPISODES = 50

env = Env()
agent = PG()


EP_reward_sums, episodes = [], []
agent.load_model("./model_init/PG1")

# Session settings
GPU_mem_ratio = 0.2
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create recomposed transition critic
state_dim = 22
hidden_dim = 3
critic_hidden_dim = 2
action_dim = 5
tuc = TUC(sess,"TUC",state_dim,action_dim,0.003) 
tuc.load_model("./model_final/TUC1")

# KL divergence
def KL_divergence(mean_1,log_std_1,mean_2,log_std_2):    

      term_1 = np.sum(np.square(np.divide(np.exp(log_std_1),np.exp(log_std_2)))) 
      term_2 = np.sum(2*log_std_2-2*log_std_1)
      term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(np.exp(log_std_2))))
          
      return np.maximum(0,0.5*(term_1+term_2+term_2-1))   

exp_results_filename = "PG1_TUC"
EPs_total_reward = []
ratio_init_1 = 0.9
ratio_init_2 = 100

mem_size = 5

states = []
next_states = []
actions = []
values = []

#EP = EPISODE 
for EP in range(1,2):

     done = False
     EP_reward_sum = 0
     state = env.reset()
     state = np.reshape(state, [1, 22])
     t = 0
     
     ratio_1 = ratio_init_1*(EPISODES-EP)/EPISODES    
     ratio_2 = ratio_init_2*(EP)/EPISODES    
    
     states_mem = []
     actions_mem = []
     next_states_mem = []
    
     while not done: 

          # Choose action    
          action = agent.get_action(state)
            
          # Get state and reward
          next_state, reward, done = env.step(action)
          next_state = np.reshape(next_state, [1, 22])
          EP_reward_sum += reward

          # Store states 
          states.append(state) 

          # Store actions
          temp_act = np.zeros([5])
          temp_act[action] = 1.
          actions.append(temp_act)         

          # Time step
          t = t + 1
             
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
