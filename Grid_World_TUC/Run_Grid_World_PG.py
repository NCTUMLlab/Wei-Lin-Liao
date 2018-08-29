import copy
import pylab
import numpy as np
from Environment import Env
from Agent import PG
import pickle

np.random.seed(0)
EPISODES = 50

env = Env()
agent = PG()


EP_reward_sums, episodes = [], []
agent.save_model("./model_init/PG1")
#agent.load_model("./model_init/PG1")

exp_results_filename = "PG1"
EPs_total_reward = []

for EP in range(EPISODES):

     done = False
     EP_reward_sum = 0
     state = env.reset()
     state = np.reshape(state, [1, 22])
     t = 0
     
     while not done:
            
         t += 1

         # RL choose action based on observation and go one step
         action = agent.get_action(state)
         next_state, reward, done = env.step(action)
         next_state = np.reshape(next_state, [1, 22])
         agent.memory(state, action, reward)

         EP_reward_sum += reward
         state = next_state

         if done:
            agent.train_episodes()
            EPs_total_reward.append(EP_reward_sum)
            print("PG episode : {0: <5} , total reward : {1: <5}".format(EP,EP_reward_sum))
            if EP == int(EPISODES/2):
               agent.save_model("./model_final/PG1_EP_"+str(int(EPISODES/2)))

# end of game
agent.save_model("./model_final/PG1")
with open("./Exp_Total_Reward/"+exp_results_filename,"wb") as fp:
     pickle.dump(EPs_total_reward,fp)
print('Game over')
env.destroy()