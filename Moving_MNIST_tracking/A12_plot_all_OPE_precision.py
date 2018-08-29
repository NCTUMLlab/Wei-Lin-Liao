import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt
import math
import pickle
import scipy.io as sio
import numpy as np

search_step = 3
threshold_res = 1

# Precision
# SL
with open("./tracking_results/SL_box_pos","rb") as fp:
       SL_box_pos = pickle.load(fp)

with open("./tracking_results/SL_gd_pos","rb") as fp:
       SL_gd_pos = pickle.load(fp)

# RL
with open("./tracking_results/RL_box_pos","rb") as fp:
       RL_box_pos = pickle.load(fp)

with open("./tracking_results/RL_gd_pos","rb") as fp:
       RL_gd_pos = pickle.load(fp)
 
# VIME
with open("./tracking_results/RL_VIME_box_pos","rb") as fp:
       RL_VIME_box_pos = pickle.load(fp)

with open("./tracking_results/RL_VIME_gd_pos","rb") as fp:
       RL_VIME_gd_pos = pickle.load(fp)

# TUC
with open("./tracking_results/RL_TUC_box_pos","rb") as fp:
       RL_TUC_box_pos = pickle.load(fp)

with open("./tracking_results/RL_TUC_gd_pos","rb") as fp:
       RL_TUC_gd_pos = pickle.load(fp)

#print(len(RL_TUC_box_pos))
#print(len(RL_TUC_gd_pos))

def distance(x1,y1,x2,y2):

      return math.sqrt((x1-x2)**2+(y1-y2)**2)

def RL_compute_precision(box_pos, gd_pos):
      avg_errors = []
      for gf in range(int(len(box_pos)/3)):
           temp_error = 0
           for ff in range(3):
                 f = int(gf*3 + ff)
                 d = distance(box_pos[f][0],box_pos[f][1],gd_pos[gf][0],gd_pos[gf][1])
                 temp_error += d
           temp_error /= 3
           avg_errors.append(temp_error)

      return avg_errors
                   

def SL_compute_precision(box_pos, gd_pos):
      avg_errors = []
      for f in range(len(box_pos)):
           d = distance(box_pos[f][0],box_pos[f][1],gd_pos[f][0],gd_pos[f][1])
           avg_errors.append(d)

      return avg_errors



def compute_result(errors, max_error):
      precision_threshold_x = np.arange(0,1+max_error,threshold_res)
      #print(precision_threshold_x)
      precision = []
      total_num = len(errors)
      for i in range(len(precision_threshold_x)):
           temp_count = 0
           for j in range(total_num):
                if errors[j] < precision_threshold_x[i]:
                   temp_count += 1
           #print(float(temp_count/total_num))
           precision.append(float(temp_count/total_num))
      #print(precision)
      return precision

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



window_size = 5

linewidth = 4
legend_size = 15


SL_error = SL_compute_precision(SL_box_pos, SL_gd_pos)
RL_error = RL_compute_precision(RL_box_pos, RL_gd_pos)
RL_VIME_error = RL_compute_precision(RL_VIME_box_pos, RL_VIME_box_pos)
RL_TUC_error = RL_compute_precision(RL_TUC_box_pos, RL_TUC_gd_pos)

print(len(SL_box_pos), len(SL_gd_pos))
print(len(RL_box_pos), len(RL_gd_pos))
print(len(RL_VIME_box_pos), len(RL_VIME_gd_pos))
print(len(RL_TUC_box_pos), len(RL_TUC_gd_pos))

print(len(SL_error))
print(len(RL_error))
print(len(RL_VIME_error))
print(len(RL_TUC_error))

print(min(SL_error), max(SL_error))
print(min(RL_error), max(RL_error))
print(min(RL_VIME_error), max(RL_VIME_error))
print(min(RL_TUC_error), max(RL_TUC_error))


max_error = int(max([max(SL_error), max(RL_error), max(RL_VIME_error), max(RL_TUC_error)]))

SL_OPE = compute_result(SL_error, max_error)
RL_OPE = compute_result(RL_error, max_error)
RL_VIME_OPE = compute_result(RL_VIME_error, max_error)
RL_TUC_OPE = compute_result(RL_TUC_error, max_error)

precision_threshold_x = np.arange(0,1+max_error,threshold_res)

SL_OPE = running_mean(SL_OPE, window_size)
RL_OPE = running_mean(RL_OPE, window_size)
RL_VIME_OPE = running_mean(RL_VIME_OPE, window_size)
RL_TUC_OPE = running_mean(RL_TUC_OPE, window_size)

model = ['SL','RL','RL_VIME','RL_TUC']      
total_precision = np.array([SL_OPE, RL_OPE, RL_VIME_OPE, RL_TUC_OPE])

with open("./tracking_results/All_precision","wb") as fp:
        pickle.dump([SL_OPE, RL_OPE, RL_VIME_OPE, RL_TUC_OPE], fp)

sio.savemat('./All_precision.mat', dict([('prcision', total_precision), ('model', model)]))  

plt.figure()
plt.title('Tracking MNIST precision plots of OPE')
plt.plot(precision_threshold_x,RL_TUC_OPE,'g',label='SL + PG (TUC)',linewidth=linewidth)
plt.plot(precision_threshold_x,RL_OPE,'b',label='SL + PG',linewidth=linewidth)
plt.plot(precision_threshold_x,SL_OPE,'m',label='SL',linewidth=linewidth)
plt.plot(precision_threshold_x,RL_VIME_OPE,'r',label='SL + PG (VIME)',linewidth=linewidth)

plt.xlabel('Location error threshold')
plt.ylabel('Precision')
plt.xlim(0,max_error)
plt.legend(prop={'size': legend_size})
plt.savefig('./tracking_results/Total_OPE_precision', dpi = 500)
plt.show()
#'''