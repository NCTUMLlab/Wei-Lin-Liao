import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

with open("./Latent_Features/VAE_features","rb") as fp:
        VAE_features = pickle.load(fp)
        print("VAE shape : ",VAE_features.shape)

with open("./Latent_Features/VAE_actions","rb") as fp:
        VAE_actions = pickle.load(fp)
        print("VAE action shape : ",VAE_actions.shape)

VAE_data_num = VAE_features.shape[0]   


VAE_PCA = PCA(n_components=2).fit_transform(VAE_features)
VAE_tSNE = TSNE(n_components=2).fit_transform(VAE_features)


with open("./Latent_Features/VAE_PCA","wb") as fp:
          pickle.dump(VAE_PCA, fp)
          
with open("./Latent_Features/VAE_tSNE","wb") as fp:
          pickle.dump(VAE_tSNE, fp)     

with open("./Latent_Features/VAE_PCA","rb") as fp:
        VAE_PCA = pickle.load(fp)
        print("VAE PCA shape : ",VAE_actions.shape)
      
with open("./Latent_Features/VAE_tSNE","rb") as fp:
        VAE_tSNE = pickle.load(fp)
        print("VAE tSNE shape : ",VAE_actions.shape)


label = ["Action 0","Action 1"]
color = ["b","r"]

legend_size = 15
img_dpi = 600

# 3D
fig = plt.figure()
ax = Axes3D(fig)
flag = [0,0]
ax.view_init(30, 30)
for d in range(VAE_data_num):
     z_x = VAE_features[d,:]
     z_p = VAE_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx],label = label[idx])
       flag[idx] = 1
     else:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx])
       
plt.title("Latent features of VAE")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_VAE", dpi = img_dpi)
plt.show()


# PCA 2D
plt.figure()
flag = [0,0]
for d in range(VAE_data_num):
     z_x = VAE_PCA[d,:]
     z_p = VAE_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])

plt.title("Latent features of VAE (PCA)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_VAE_PCA",dpi=img_dpi)
plt.show()

# tSNE 2D
plt.figure()
flag = [0,0]
for d in range(VAE_data_num):
     z_x = VAE_tSNE[d,:]
     z_p = VAE_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])
        
plt.title("Latent features of VAE (tSNE)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_VAE_tSNE",dpi=img_dpi)
plt.show()
