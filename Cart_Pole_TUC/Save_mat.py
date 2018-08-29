import pickle
import scipy.io as sio
import numpy as np

# Load VAE latent features 
with open("./Latent_Features/VAE_features","rb") as fp:
        VAE_features = pickle.load(fp)
     
with open("./Latent_Features/VAE_actions","rb") as fp:
        VAE_actions = pickle.load(fp)

with open("./Latent_Features/VAE_PCA","rb") as fp:
        VAE_features_PCA = pickle.load(fp)      

with open("./Latent_Features/VAE_tSNE","rb") as fp:
        VAE_features_tSNE = pickle.load(fp) 
        
# Convert VAE data to mat
sio.savemat('./VAE_ori.mat', dict([('z_ori', VAE_features), ('act', VAE_actions)]))         
sio.savemat('./VAE_PCA.mat', dict([('z_PCA', VAE_features_PCA), ('act', VAE_actions)]))  
sio.savemat('./VAE_tSNE.mat', dict([('z_tSNE', VAE_features_tSNE), ('act', VAE_actions)]))  

# Load TUC latent features
with open("./Latent_Features/TUC_z_features","rb") as fp:
        TUC_features = pickle.load(fp)
     
with open("./Latent_Features/TUC_z_actions","rb") as fp:
        TUC_actions = pickle.load(fp)

with open("./Latent_Features/TUC_z_PCA","rb") as fp:
        TUC_PCA = pickle.load(fp)      

with open("./Latent_Features/TUC_z_tSNE","rb") as fp:
        TUC_tSNE = pickle.load(fp) 
        
# Convert TUC data to mat
sio.savemat('./TUC_ori.mat', dict([('z_ori', TUC_features), ('act', TUC_actions)]))         
sio.savemat('./TUC_PCA.mat', dict([('z_PCA', TUC_PCA), ('act', TUC_actions)]))  
sio.savemat('./TUC_tSNE.mat', dict([('z_tSNE', TUC_tSNE), ('act', TUC_actions)]))  


# Convert total reward
with open("./Exp_Total_Reward/PG","rb") as fp:
        PG = pickle.load(fp)
        PG = np.array(PG)

with open("./Exp_Total_Reward/PG_VIME","rb") as fp:
        PG_VIME = pickle.load(fp)
        PG_VIME = np.array(PG_VIME)
        
with open("./Exp_Total_Reward/PG_TUC_KL_0","rb") as fp:
        PG_TUC = pickle.load(fp)
        PG_TUC = np.array(PG_TUC)

total_reward = np.array([PG,PG_VIME,PG_TUC])
model = ['PG','PG_VIME','PG_TUC']
        
# Convert total reward data to mat
sio.savemat('./Total_reward.mat', dict([('total_reward', total_reward), ('model', model)]))      

# Convert the comparison
with open("./Exp_Total_Reward/PG_TUC_KL_exploration_0_9","rb") as fp:
        PG_TUC_KL_exploration_0_9 = pickle.load(fp)
        PG_TUC_KL_exploration_0_9= np.array(PG_TUC_KL_exploration_0_9)   

with open("./Exp_Total_Reward/PG_TUC_KL_exploration_2_9","rb") as fp:
        PG_TUC_KL_exploration_2_9 = pickle.load(fp)
        PG_TUC_KL_exploration_2_9= np.array(PG_TUC_KL_exploration_2_9)   

with open("./Exp_Total_Reward/PG_TUC_KL_penalty_20","rb") as fp:
        PG_TUC_KL_penalty_20 = pickle.load(fp)
        PG_TUC_KL_penalty_20= np.array(PG_TUC_KL_penalty_20)  

with open("./Exp_Total_Reward/PG_TUC_KL_penalty_5","rb") as fp:
        PG_TUC_KL_penalty_5 = pickle.load(fp)
        PG_TUC_KL_penalty_5= np.array(PG_TUC_KL_penalty_5)  

exploration_comparison = np.array([PG_TUC_KL_exploration_0_9,PG_TUC_KL_exploration_2_9])
model = ['Exp_0_9','Exp_2_9']
sio.savemat('./Exploration_comparison.mat', dict([('reward', exploration_comparison), ('model', model)]))      

penalty_comparison = np.array([PG_TUC_KL_penalty_5,PG_TUC_KL_penalty_20])
model = ['Pen_5','Pen_20']
sio.savemat('./Penalty_comparison.mat', dict([('reward', penalty_comparison), ('model', model)]))    