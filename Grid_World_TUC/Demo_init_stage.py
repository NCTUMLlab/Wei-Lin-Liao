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

exp_results_filename = "PG1_TUC"
EPs_total_reward = []
ratio_init_1 = 0.9
ratio_init_2 = 100

mem_size = 5

states = []
next_states = []
actions = []
values = []



for EP in range(EPISODES):

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

         # RL choose action based on observation and go one step
         action = agent.get_action(state)
         next_state, reward, done = env.step(action)
         next_state = np.reshape(next_state, [1, 22])

         # Store states 
         states.append(state) 

         # Store next states
         next_states.append(next_state)

         # Store actions
         temp_act = np.zeros([5])
         temp_act[action] = 1.
         actions.append(temp_act)  

         # Store reward
         values.append(reward)    

         # Time step
         t += 1         

         state = next_state

# end of game