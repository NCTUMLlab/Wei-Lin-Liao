import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

with open("./Latent_Features/DNN_features","rb") as fp:
        DNN_features = pickle.load(fp)
        print("DNN shape : ",DNN_features.shape)

with open("./Latent_Features/DNN_actions","rb") as fp:
        DNN_actions = pickle.load(fp)
        print("DNN action shape : ",DNN_actions.shape)

DNN_data_num = DNN_features.shape[0]   


DNN_PCA = PCA(n_components=2).fit_transform(DNN_features)
DNN_tSNE = TSNE(n_components=2).fit_transform(DNN_features)


with open("./Latent_Features/DNN_PCA","wb") as fp:
          pickle.dump(DNN_PCA, fp)
          
with open("./Latent_Features/DNN_tSNE","wb") as fp:
          pickle.dump(DNN_tSNE, fp)     

with open("./Latent_Features/DNN_PCA","rb") as fp:
        DNN_PCA = pickle.load(fp)
        print("DNN PCA shape : ",DNN_actions.shape)
      
with open("./Latent_Features/DNN_tSNE","rb") as fp:
        DNN_tSNE = pickle.load(fp)
        print("DNN tSNE shape : ",DNN_actions.shape)


label = ["Action 0","Action 1"]
color = ["b","r"]

legend_size = 15
img_dpi = 600

# 3D
fig = plt.figure()
ax = Axes3D(fig)
flag = [0,0]
ax.view_init(30, 30)
for d in range(DNN_data_num):
     z_x = DNN_features[d,:]
     z_p = DNN_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx],label = label[idx])
       flag[idx] = 1
     else:
       ax.scatter(z_x[0],z_x[1],z_x[2],c = color[idx])
       
plt.title("Latent features of DNN")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_DNN", dpi = img_dpi)
plt.show()


# PCA 2D
plt.figure()
flag = [0,0]
for d in range(DNN_data_num):
     z_x = DNN_PCA[d,:]
     z_p = DNN_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])

plt.title("Latent features of DNN (PCA)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_DNN_PCA",dpi=img_dpi)
plt.show()

# tSNE 2D
plt.figure()
flag = [0,0]
for d in range(DNN_data_num):
     z_x = DNN_tSNE[d,:]
     z_p = DNN_actions[d,:]
     idx = np.argmax(z_p)
     if flag[idx] == 0:
        plt.scatter(z_x[0],z_x[1],c = color[idx],label = label[idx])
        flag[idx] = 1
     else:
        plt.scatter(z_x[0],z_x[1],c = color[idx])
        
plt.title("Latent features of DNN (tSNE)")
plt.legend(prop={'size': legend_size})
plt.savefig("./Latent_Features/Latent_features_of_DNN_tSNE",dpi=img_dpi)
plt.show()
