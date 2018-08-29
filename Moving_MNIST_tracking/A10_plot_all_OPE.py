import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt


with open("./tracking_results/SL_OPE","rb") as fp:
       SL_OPE = pickle.load(fp)

with open("./tracking_results/RL_OPE","rb") as fp:
       RL_OPE = pickle.load(fp)

with open("./tracking_results/RL_VIME_OPE","rb") as fp:
       RL_VIME_OPE = pickle.load(fp)

with open("./tracking_results/RL_TUC_OPE","rb") as fp:
       RL_TUC_OPE = pickle.load(fp)
 
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


window_size = 50
threshold_res = 0.001
linewidth = 4
legend_size = 15
precision_threshold_x = np.arange(0,1+threshold_res,threshold_res)

SL_OPE = running_mean(SL_OPE, window_size)
RL_OPE = running_mean(RL_OPE, window_size)
RL_VIME_OPE = running_mean(RL_VIME_OPE, window_size)
RL_TUC_OPE = running_mean(RL_TUC_OPE, window_size)

plt.figure()
plt.title('Tracking MNIST success plots of OPE')
plt.plot(precision_threshold_x,RL_TUC_OPE,'g',label='SL + PG (TUC)',linewidth=linewidth)
plt.plot(precision_threshold_x,RL_OPE,'b',label='SL + PG',linewidth=linewidth)
plt.plot(precision_threshold_x,RL_VIME_OPE,'r',label='SL + PG (VIME)',linewidth=linewidth)
plt.plot(precision_threshold_x,SL_OPE,'m',label='SL',linewidth=linewidth)
plt.xlabel('Overlap threshold')
plt.ylabel('Success rate')
plt.xlim(0,1)
plt.legend(prop={'size': legend_size})
plt.savefig('./tracking_results/Total_OPE_success', dpi = 500)
plt.show()