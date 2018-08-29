import copy
import pylab
import numpy as np
import tensorflow as tf
from Environment import Env
from Agent import PG
from Agent import VAE
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
vae = VAE(sess,"VAE",22,5,0.001)
vae.save_model("./model_init/VAE")

EPISODE = 50
action_dim = 5
mem_size = 5
'''
for EP in range(1,EPISODES+1):

     done = False
     EP_reward_sum = 0
     state = env.reset()
     state = np.reshape(state, [1, 22])
     t = 0
    
     states_mem = []
     actions_mem = []
     next_states_mem = []

     while not done:

         # RL choose action based on observation and go one step
         action = agent.get_action(state)
         next_state, reward, done = env.step(action)
         next_state = np.reshape(next_state, [1, 22])
         EP_reward_sum += reward

         # Store actions
         temp_act = np.zeros([action_dim])
         temp_act[action] = 1.

         # Time step
         t += 1               
 
         # Get intrinsic reward
         if t <= mem_size :
            states_mem.append(state) 
            actions_mem.append(temp_act)
            next_states_mem.append(next_state)

         else : 
            vae.train(np.concatenate((np.vstack(states_mem),np.vstack(actions_mem)), axis = 1),np.vstack(next_states_mem))

            # Update memory
            states_mem.append(state)
            actions_mem.append(temp_act)
            next_states_mem.append(next_state)
            states_mem.pop(0)
            actions_mem.pop(0)
            next_states_mem.pop(0)

         #print(intrinsic_reward_1,intrinsic_reward_2)
         agent.memory(state, action, reward)

         if done:
            agent.train_episodes()
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
         state = next_state


vae.save_model("./model_final/VAE")

'''
# =======================================================


agent.load_model("./model_init/PG1")
vae.load_model("./model_final/VAE")

EPs_total_reward = []

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
 
           
            #TUC_z = tuc.dump_z(states, actions)
            features = vae.dump_z(np.concatenate((states,actions), axis =1))
            with open("./Latent_Features/VAE_features","wb") as fp:
                 pickle.dump(features, fp)
            with open("./Latent_Features/VAE_actions","wb") as fp:
                 pickle.dump(actions, fp)
            with open("./Latent_Features/VAE_states","wb") as fp:
                 pickle.dump(states, fp)     

            states = []
            actions = []
          
            # Print total reward          
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            
           
            break
            
          # Enter next state
          state = next_state
