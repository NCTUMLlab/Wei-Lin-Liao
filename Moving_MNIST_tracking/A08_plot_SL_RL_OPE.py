import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt
import os

with open("./tracking_results/RL_OPE","rb") as fp:
        RL_ope_curve_y = pickle.load(fp)

with open("./tracking_results/SL_OPE","rb") as fp:
        SL_ope_curve_y = pickle.load(fp)

threshold_res = 0.001
precision_threshold_x = np.arange(0,1+threshold_res,threshold_res)
linewidth = 3.0


plt.figure(1)
plt.title('Tracking MNIST')
plt.plot(precision_threshold_x,SL_ope_curve_y,'r',label='SL',linewidth=linewidth)
plt.plot(precision_threshold_x,RL_ope_curve_y,'b',label='RL',linewidth=linewidth)
plt.xlabel('Overlap threshold')
plt.ylabel('Success rate')
plt.xlim(0,1)
plt.legend(prop={'size': 12})
plt.savefig('./tracking_results/RL_SL_OPE')
plt.show()