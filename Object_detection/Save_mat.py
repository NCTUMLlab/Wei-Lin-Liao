import pickle
import scipy.io as sio
import numpy as np



def process(precision, recall):
      precision = precision.tolist()
      recall = recall.tolist()
      new_recall_list = []
      new_precision_list = []
      new_recall = []
      new_precision = []
      current_recall = recall[0]
      temp_recall = []
      temp_precision = []
           
      for i in range(0, len(recall)):
           if recall[i] == current_recall:
              temp_recall.append(recall[i])       
              temp_precision.append(precision[i])    
           else:
              new_recall_list.append(temp_recall)
              new_precision_list.append(temp_precision)
              temp_recall = [recall[i]]
              temp_precision = [precision[i]]
              current_recall = recall[i]

      for i in range(0, len(new_recall_list)):  
           print(new_recall_list[i])   
           new_recall.append(new_recall_list[i][0])      
           new_precision.append(max(new_precision_list[i]))


      return np.array(new_precision), np.array(new_recall)


with open("./experiment_data/PG_2_PR_curve","rb") as fp:
        PG_PR_curve = pickle.load(fp)
        print(PG_PR_curve)    

with open("./experiment_data/PG_VIME_2_PR_curve","rb") as fp:
        PG_VIME_PR_curve = pickle.load(fp)
        print(PG_PR_curve)    

with open("./experiment_data/PG_TUC_2_PR_curve","rb") as fp:
        PG_TUC_PR_curve = pickle.load(fp)
        print(PG_TUC_PR_curve) 


#PG_PR_curve[0] = running_mean(PG_PR_curve[0], 3)
#PG_VIME_PR_curve[0] = running_mean(PG_VIME_PR_curve[0], 3)
#PG_TUC_PR_curve[0] = running_mean(PG_TUC_PR_curve[0], 3)

PG_PR_curve[0], PG_PR_curve[1] = process(PG_PR_curve[0], PG_PR_curve[1])
PG_VIME_PR_curve[0], PG_VIME_PR_curve[1] = process(PG_VIME_PR_curve[0], PG_VIME_PR_curve[1])
PG_TUC_PR_curve[0], PG_TUC_PR_curve[1] = process(PG_TUC_PR_curve[0], PG_TUC_PR_curve[1])

sio.savemat('./PG_PR_curve', dict([('precision',  PG_PR_curve[0]),('recall', PG_PR_curve[1])])) 
sio.savemat('./PG_VIME_PR_curve', dict([('precision',  PG_VIME_PR_curve[0]),('recall', PG_VIME_PR_curve[1])])) 
sio.savemat('./PG_TUC_PR_curve', dict([('precision',  PG_TUC_PR_curve[0]),('recall', PG_TUC_PR_curve[1])])) 