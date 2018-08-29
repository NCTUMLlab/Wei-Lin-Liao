import pickle
import numpy as np
import matplotlib.pyplot as plt

def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l )
 
    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)
 
    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N
 
    return result    
    
def running_std(l, N):

    sum = 0
    result = list( 0 for x in l )
    
    window = []
    
    for i in range( 0, N ):
        window.append(l[i])
        result[i] = np.std(np.array(window))
 
    for i in range( N, len(l) ):
        window.append(l[i])
        window.pop(0)
        result[i] = np.std(np.array(window))
 
    return np.array(result)  


with open("./Exp_Total_Reward/PG_TUC_KL_0","rb") as fp:
        PG_TUC_KL = pickle.load(fp)

with open("./Exp_Total_Reward/PG_TUC_KL_exploration_2_9","rb") as fp:
        PG_TUC_KL_2_9 = pickle.load(fp)

with open("./Exp_Total_Reward/PG_TUC_KL_exploration_0_9","rb") as fp:
        PG_TUC_KL_0_9 = pickle.load(fp)

        
window_size = 13
linewidth = 3.0
legend_size = 12
alpha = 0.3

PG_TUC_mean = running_mean(PG_TUC_KL,window_size)
PG_TUC_2_9_mean = running_mean(PG_TUC_KL_2_9,window_size)
PG_TUC_0_9_mean = running_mean(PG_TUC_KL_0_9,window_size)

#PG_std = running_std(PG,window_size)
#PG_TUC_KL_std = running_std(PG_TUC_KL,window_size)
#PG_VIME_std = running_std(PG_VIME,window_size)

EPs_X = np.arange(1,1+len(PG_TUC_mean))

plt.figure()
#plt.fill_between(EPs_X, PG_mean + PG_std, PG_mean - PG_std,  where = PG_mean + PG_std >= PG_mean - PG_std, facecolor = (153/255,204/255,255/255), alpha = alpha)
#plt.fill_between(EPs_X, PG_TUC_KL_mean + PG_TUC_KL_std, PG_TUC_KL_mean - PG_TUC_KL_std,  where = PG_TUC_KL_mean + PG_TUC_KL_std >= PG_TUC_KL_mean - PG_TUC_KL_std, facecolor = (255/255,204/255,204/255), alpha = alpha)
#plt.fill_between(EPs_X, PG_VIME_mean + PG_VIME_std, PG_VIME_mean - PG_VIME_std,  where = PG_VIME_mean + PG_VIME_std >= PG_VIME_mean - PG_VIME_std, facecolor = (204/255,255/255,229/255), alpha = alpha)


plt.plot(EPs_X, PG_TUC_mean, "r",label = "PG + TUC original", linewidth = linewidth)
plt.plot(EPs_X, PG_TUC_2_9_mean  , "g",label = "PG + plus", linewidth = linewidth)
plt.plot(EPs_X, PG_TUC_0_9_mean, "b",label = "PG + minus", linewidth = linewidth)

plt.xlim((1,130))
plt.title("Cart Pole")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend(prop={'size': legend_size})
#plt.savefig("./Exp_Total_Reward/Cart Pole Total Reward with standard deviation", dpi = 700)
plt.show()