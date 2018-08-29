import pickle
import scipy.io as sio
import numpy as np

# Load VAE latent features      
with open("./Latent_Features/VAE_actions","rb") as fp:
        VAE_actions = pickle.load(fp)

with open("./Latent_Features/VAE_features","rb") as fp:
        VAE_ori = pickle.load(fp)    

with open("./Latent_Features/VAE_PCA_3","rb") as fp:
        VAE_PCA_3 = pickle.load(fp)      

with open("./Latent_Features/VAE_PCA","rb") as fp:
        VAE_PCA = pickle.load(fp)   

with open("./Latent_Features/VAE_tSNE_3","rb") as fp:
        VAE_tSNE_3 = pickle.load(fp) 

with open("./Latent_Features/VAE_tSNE","rb") as fp:
        VAE_tSNE = pickle.load(fp) 
        
# Convert VAE data to mat
sio.savemat('./VAE_ori.mat', dict([('z_ori',  VAE_ori), ('act', VAE_actions)]))  

sio.savemat('./VAE_PCA_3.mat', dict([('z_PCA_3', VAE_PCA_3), ('act', VAE_actions)]))  
sio.savemat('./VAE_PCA.mat', dict([('z_PCA', VAE_PCA), ('act', VAE_actions)]))  

sio.savemat('./VAE_tSNE_3.mat', dict([('z_tSNE_3', VAE_tSNE_3), ('act', VAE_actions)]))  
sio.savemat('./VAE_tSNE.mat', dict([('z_tSNE', VAE_tSNE), ('act', VAE_actions)]))  

# Load TUC latent features   
with open("./Latent_Features/TUC_z_actions","rb") as fp:
        TUC_actions = pickle.load(fp)

with open("./Latent_Features/TUC_z_features","rb") as fp:
        TUC_ori = pickle.load(fp)   

with open("./Latent_Features/TUC_z_PCA_3","rb") as fp:
        TUC_PCA_3 = pickle.load(fp)   

with open("./Latent_Features/TUC_z_PCA","rb") as fp:
        TUC_PCA = pickle.load(fp)      

with open("./Latent_Features/TUC_z_tSNE_3","rb") as fp:
        TUC_tSNE_3 = pickle.load(fp) 

with open("./Latent_Features/TUC_z_tSNE","rb") as fp:
        TUC_tSNE = pickle.load(fp)
        
# Convert TUC data to mat
sio.savemat('./TUC_ori.mat', dict([('z_ori',  TUC_ori), ('act', VAE_actions)]))  

sio.savemat('./TUC_PCA_3.mat', dict([('z_PCA_3', TUC_PCA_3), ('act', VAE_actions)]))  
sio.savemat('./TUC_PCA.mat', dict([('z_PCA', TUC_PCA), ('act', VAE_actions)]))  

sio.savemat('./TUC_tSNE_3.mat', dict([('z_tSNE_3', TUC_tSNE_3), ('act', VAE_actions)]))  
sio.savemat('./TUC_tSNE.mat', dict([('z_tSNE', TUC_tSNE), ('act', VAE_actions)]))  


# Convert total reward
with open("./Exp_Total_Reward/PG1","rb") as fp:
        PG = pickle.load(fp)
        PG = np.array(PG)

with open("./Exp_Total_Reward/PG1_VIME","rb") as fp:
        PG_VIME = pickle.load(fp)
        PG_VIME = np.array(PG_VIME)
        
with open("./Exp_Total_Reward/PG1_TUC","rb") as fp:
        PG_TUC = pickle.load(fp)
        PG_TUC = np.array(PG_TUC)

total_reward = np.array([PG,PG_VIME,PG_TUC])
model = ['PG','PG_VIME','PG_TUC']
        
# Convert total reward data to mat
sio.savemat('./Total_reward.mat', dict([('total_reward', total_reward), ('model', model)]))         

# Save all data
with open("./Exp_Total_Reward/PG1_TUC_exploration_0_6","rb") as fp:
        PG1_TUC_exploration_0_6 = pickle.load(fp)

with open("./Exp_Total_Reward/PG1_TUC_exploration_1_2","rb") as fp:
        PG1_TUC_exploration_1_2 = pickle.load(fp)

with open("./Exp_Total_Reward/PG1_TUC_penalty_150","rb") as fp:
        PG1_TUC_penalty_150 = pickle.load(fp)

with open("./Exp_Total_Reward/PG1_TUC_penalty_50","rb") as fp:
        PG1_TUC_penalty_50 = pickle.load(fp)


total_reward = np.array([PG1_TUC_exploration_0_6,PG1_TUC_exploration_1_2,PG1_TUC_penalty_150,PG1_TUC_penalty_50])
model = ['Exp_0_6','Exp_1_2','Penalty_150','Penalty_50']

sio.savemat('./Comparison_exploration_penalty.mat', dict([('total_reward', total_reward), ('model', model)]))         
