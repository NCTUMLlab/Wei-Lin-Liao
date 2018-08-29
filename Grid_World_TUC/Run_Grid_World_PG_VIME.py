import copy
import pylab
import numpy as np
import tensorflow as tf
from Environment import Env
from Agent import PG
from Agent import TUC
from Agent import VIME
import pickle

np.random.seed(0)
EPISODES = 50

env = Env()
agent = PG()


EP_reward_sums, episodes = [], []
#agent.save_model("./model_init/PG1")
agent.load_model("./model_init/PG1")

# Session settings
GPU_mem_ratio = 0.2
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_mem_ratio)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

# Create recomposed transition critic
state_dim = 22
action_dim = 5
vime = VIME(sess,"VIME",state_dim,action_dim,20,0.003)
vime.save_model("./model_init/VIME1")
#vime.load_model("./model_init/VIME1")
mem_size = 5


exp_results_filename = "PG1_VIME"
EPs_total_reward = []
ratio_init_1 = 0.1

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
    
     states_actions_mem = []
     next_states_mem = []  

     while not done:

         # RL choose action based on observation and go one step
         action = agent.get_action(state)
         next_state, reward, done = env.step(action)
         next_state = np.reshape(next_state, [1, 22])
         EP_reward_sum += reward

         # Store state and action
         temp_state_action = np.zeros([22+5])
         temp_state_action[0:state_dim] = state
         temp_state_action[state_dim:state_dim+action] = 1.

         # Store next states
         next_states.append(next_state)

         # Time step
         t += 1               
 
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
            intrinsic_reward_1 = vime.get_info_gain(hyper_parameters, pre_hyper_parameters)/mem_size    
            pre_hyper_parameters = hyper_parameters
            # Update memory
            states_actions_mem.append(temp_state_action)
            next_states_mem.append(next_state)
            states_actions_mem.pop(0)
            next_states_mem.pop(0)

         #print(intrinsic_reward_1)
         agent.memory(state, action, reward+intrinsic_reward_1*ratio_1)

         if done:
            agent.train_episodes()
            EPs_total_reward.append(EP_reward_sum)
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            if EP == int(EPISODES/2):
               agent.save_model("./model_final/PG1_VIME_EP_"+str(int(EPISODES/2)))

         state = next_state

# end of game
#'''
agent.save_model("./model_final/PG1_VIME")
with open("./Exp_Total_Reward/"+exp_results_filename,"wb") as fp:
     pickle.dump(EPs_total_reward,fp)
#'''
print('Game over')
env.destroy()