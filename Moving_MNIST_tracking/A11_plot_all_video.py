import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt

with open('new_MNIST_RL_data_digit_7', 'rb') as fp:
       RL_img, RL_gd, _,  _ = pickle.load(fp)  

with open("./tracking_results/SL_box_pos","rb") as fp:
       SL_box_pos = pickle.load(fp)

with open("./tracking_results/RL_box_pos","rb") as fp:
       RL_box_pos = pickle.load(fp)

with open("./tracking_results/RL_VIME_box_pos","rb") as fp:
       RL_VIME_box_pos = pickle.load(fp)

with open("./tracking_results/RL_TUC_box_pos","rb") as fp:
       RL_TUC_box_pos = pickle.load(fp)


frame_num = 80
search_step = 3
stage_list = ["SL","RL","RL_VIME","RL_TUC"]
box_pos_list = [SL_box_pos,RL_box_pos,RL_VIME_box_pos,RL_TUC_box_pos]

# Record the tracking procedure
def save_tracking_process(idx,ori_img,box,gd,stage):
      img = ori_img[0,:,:,0]

      # Plot box
      img[box[1],box[0]:box[0]+28]= 0.5
      img[box[1]+28,box[0]:box[0]+28]= 0.5
      img[box[1]:box[1]+28,box[0]]= 0.5
      img[box[1]:box[1]+28,box[0]+28]= 0.5

      # Plot groundtruth
      img[gd[1],gd[0]:gd[0]+28]= 1
      img[gd[1]+28,gd[0]:gd[0]+28]= 1
      img[gd[1]:gd[1]+28,gd[0]]= 1
      img[gd[1]:gd[1]+28,gd[0]+28]= 1

      # Save tracking images
      if idx<10:
         filename = 'MNIST_00'+str(idx)+'.png'
      elif idx<100:
         filename = 'MNIST_0'+str(idx)+'.png'
      elif idx<1000:
         filename = 'MNIST_'+str(idx)+'.png'
      plt.figure(idx)
      plt.imshow(img)
      #plt.savefig('./new_'+stage+'_tracking_outcome_imgs/'+filename)
      plt.savefig("./tracking_videos/"+stage+"/"+filename)
      plt.clf()
      plt.close()
 


def save_figures(box_pos,stage):
      idx = 1
      print(stage," !!")
      for f in range(frame_num):
           frame_gd_x, frame_gd_y = RL_gd[f][0], RL_gd[f][1]
           frame_img = RL_img[f]
           for s in range(search_step):
                box_x, box_y = box_pos[f*search_step+s][0], box_pos[f*search_step+s][1]
                save_tracking_process(idx,np.copy(frame_img),[box_x, box_y ],[frame_gd_x, frame_gd_y],stage)
                idx += 1

for d in range(len(stage_list)):
     save_figures(box_pos_list[d],stage_list[d])