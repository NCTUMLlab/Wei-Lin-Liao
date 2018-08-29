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

VAE_PCA_3 = PCA(n_components=3).fit_transform(VAE_features)
VAE_tSNE_3 = TSNE(n_components=3).fit_transform(VAE_features)

VAE_PCA = PCA(n_components=2).fit_transform(VAE_features)
VAE_tSNE = TSNE(n_components=2).fit_transform(VAE_features)


with open("./Latent_Features/VAE_PCA_3","wb") as fp:
          pickle.dump(VAE_PCA_3, fp)
          
with open("./Latent_Features/VAE_tSNE_3","wb") as fp:
          pickle.dump(VAE_tSNE_3, fp)     

with open("./Latent_Features/VAE_PCA_3","rb") as fp:
        VAE_PCA_3 = pickle.load(fp)
        print("VAE PCA shape : ",VAE_actions.shape)
      
with open("./Latent_Features/VAE_tSNE_3","rb") as fp:
        VAE_tSNE_3 = pickle.load(fp)
        print("VAE tSNE shape : ",VAE_actions.shape)


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


label = ["Action 0","Action 1","Action 2","Action 3","Action 4"]
color = ["b","r","g","c","m"]

legend_size = 15
img_dpi = 600

# PCA 3D
fig = plt.figure()
ax = Axes3D(fig)
flag = [0,0,0,0,0]
ax.view_init(30, 30)
for d in range(VAE_data_num):
     z_x = VAE_PCA_3[d,:]
     z_p = VAE_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx],label = label[idx])
       flag[idx] = 1
     else:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx])
       
plt.title("Latent features of VAE (PCA)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_VAE_PCA_3", dpi = img_dpi)
plt.show()


# tSNE 3D
fig = plt.figure()
ax = Axes3D(fig)
flag = [0,0,0,0,0]
ax.view_init(30, 30)
for d in range(VAE_data_num):
     z_x = VAE_tSNE_3[d,:]
     z_p = VAE_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx],label = label[idx])
       flag[idx] = 1
     else:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx])
       
plt.title("Latent features of VAE (tSNE)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_VAE_tSNE_3", dpi = img_dpi)
plt.show()

# PCA 2D
plt.figure()
flag = [0,0,0,0,0]
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
flag = [0,0,0,0,0]
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
