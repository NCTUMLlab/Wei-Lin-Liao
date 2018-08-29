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
#agent.save_model("./model_init/PG1")
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
#tuc.save_model("./model_init/TUC1")
tuc.load_model("./model_init/TUC1")

# KL divergence
def KL_divergence(mean_1,log_std_1,mean_2,log_std_2):    

      term_1 = np.sum(np.square(np.divide(np.exp(log_std_1),np.exp(log_std_2)))) 
      term_2 = np.sum(2*log_std_2-2*log_std_1)
      term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(np.exp(log_std_2))))
          
      return np.maximum(0,0.5*(term_1+term_2+term_2-1))   

exp_results_filename = "PG1_TUC_exploration_1_2"
EPs_total_reward = []
ratio_init_1 = 1.2
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
         EP_reward_sum += reward

         # Store states 
         states.append(state) 

         # Store next states
         next_states.append(next_state)

         # Store actions
         temp_act = np.zeros([action_dim])
         temp_act[action] = 1.
         actions.append(temp_act)  

         # Store reward
         values.append(reward)    

         # Time step
         t += 1               
 
         # Get intrinsic reward
         if t <= mem_size :
            states_mem.append(state) 
            actions_mem.append(temp_act)
            next_states_mem.append(next_state)
            if t == mem_size :
               pre_z_mean, pre_z_std = tuc.dump_z_mean_std(state,temp_act[np.newaxis,:])
            intrinsic_reward_1 = 0
            intrinsic_reward_2 = 0
         else : 
            tuc.train_tuc(np.vstack(states_mem), np.vstack(next_states_mem), np.vstack(actions_mem))  
            z_mean, z_std = tuc.dump_z_mean_std(state,temp_act[np.newaxis,:])
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

         #print(intrinsic_reward_1,intrinsic_reward_2)
         agent.memory(state, action, reward+intrinsic_reward_1*ratio_1-intrinsic_reward_2*ratio_2)

         if done:
            agent.train_episodes()
            EPs_total_reward.append(EP_reward_sum)
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            if EP == int(EPISODES/2):
               agent.save_model("./model_final/PG1_TUC_EP_"+str(int(EPISODES/2)))

         state = next_state

# end of game
#'''
#agent.save_model("./model_final/PG1_TUC")
with open("./Exp_Total_Reward/"+exp_results_filename,"wb") as fp:
       pickle.dump(EPs_total_reward,fp)
#'''
#tuc.save_model("./model_final/TUC1")
print('Game over')
env.destroy()