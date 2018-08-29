import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

with open("./Latent_Features/TUC_z_features","rb") as fp:
        TUC_z_features = pickle.load(fp)
        print("TUC z shape : ",TUC_z_features.shape)

with open("./Latent_Features/TUC_z_actions","rb") as fp:
        TUC_z_actions = pickle.load(fp)
        print("TUC z action shape : ",TUC_z_actions.shape)

TUC_data_num = TUC_z_features.shape[0]   


TUC_z_PCA = PCA(n_components=2).fit_transform(TUC_z_features)
TUC_z_tSNE = TSNE(n_components=2).fit_transform(TUC_z_features)


with open("./Latent_Features/TUC_z_PCA","wb") as fp:
          pickle.dump(TUC_z_PCA, fp)
          
with open("./Latent_Features/TUC_z_tSNE","wb") as fp:
          pickle.dump(TUC_z_tSNE, fp)     

with open("./Latent_Features/TUC_z_PCA","rb") as fp:
        TUC_z_PCA = pickle.load(fp)
        print("TUC z PCA shape : ",TUC_z_actions.shape)
      
with open("./Latent_Features/TUC_z_tSNE","rb") as fp:
        TUC_z_tSNE = pickle.load(fp)
        print("TUC z tSNE shape : ",TUC_z_actions.shape)


label = ["Action 0","Action 1"]
color = ["b","r"]

legend_size = 15
img_dpi = 600

# PCA 3D
fig = plt.figure()
ax = Axes3D(fig)
flag = [0,0]
ax.view_init(30, 30)
for d in range(TUC_data_num):
     z_x = TUC_z_features[d,:]
     z_p = TUC_z_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx],label = label[idx])
       flag[idx] = 1
     else:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx])
       
plt.title("Latent features of TUC")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_TUC", dpi = img_dpi)
plt.show()


# PCA 2D
plt.figure()
flag = [0,0]
for d in range(TUC_data_num):
     z_x = TUC_z_PCA[d,:]
     z_p = TUC_z_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])

plt.title("Latent features of TUC (PCA)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_TUC_PCA",dpi=img_dpi)
plt.show()

# tSNE 2D
plt.figure()
flag = [0,0]
for d in range(TUC_data_num):
     z_x = TUC_z_tSNE[d,:]
     z_p = TUC_z_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])
        
plt.title("Latent features of TUC (tSNE)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_TUC_tSNE",dpi=img_dpi)
plt.show()
