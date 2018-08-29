from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randint
import numpy as np
import pickle


batch_size = 1

# Image size
img_H = 100
img_W = 100

# Frame num
train_num = 2000
test_num = 500
step_size_range = 14

# Get mnist data
sample_x, sample_y = mnist.train.next_batch(1)
sample_x = np.reshape(sample_x, (28,28))

# Initialization of lacation
x = randint(0, img_W-1-28)
y = randint(0, img_H-1-28)

# Imgs list
train_imgs = []
test_imgs = []

# Groundtruth list
train_gds = [] 
test_gds = []

for t in range(train_num):
      frame_img = np.zeros((1,img_H,img_W,1))
      frame_img[0,y: y+28, x: x+28,0] = sample_x
      train_imgs.append(frame_img)
      train_gds.append([x,y])
      
      while 1 :  
               dx = randint(-step_size_range,step_size_range)
               dy = randint(-step_size_range,step_size_range)

               if x+dx  >= 0 and x+dx < img_W-28-1  and y+dy >= 0 and y+dy < img_H-28-1 :
                  
                  x += dx 
                  y += dy  
                  break

for t in range(test_num):
      fest_img = np.zeros((1,img_H,img_W,1))
      fest_img[0,y: y+28, x: x+28,0] = sample_x
      test_imgs.append(frame_img)
      test_gds.append([x,y])
      
      while 1 :  
               dx = randint(-step_size_range,step_size_range)
               dy = randint(-step_size_range,step_size_range)
               
               if x+dx  >= 0 and x+dx < img_W-28-1  and y+dy >= 0 and y+dy < img_H-28-1 :
                 
                  x += dx 
                  y += dy  
                  break

all_data = [train_imgs, train_gds, test_imgs, test_gds]
                  
with open('new_MNIST_RL_data_digit_'+str(np.argmax(sample_y)), 'wb') as fp:
     pickle.dump(all_data , fp)     


# Generation of SL training data
     
template_x = np.zeros(((1,56,56,1)))
template_x[0,15:43,15:43,0] = sample_x
ori_x = 14
ori_y = 14

SL_img = []
SL_action = []

SL_train_num = 2000
shift_range = 14

stop_threshold = 3
normal_threshold = 8
double_threshold = 14

for t in range(SL_train_num):
  
   
   dx = randint(-shift_range,shift_range)
   dy = randint(-shift_range,shift_range)   

   clip_x = ori_x + dx
   clip_y = ori_y + dy

   sample_img = template_x[:,clip_y:clip_y+28,clip_x:clip_x+28,:]   
   
   if abs(dx) <= stop_threshold and abs(dy) <= stop_threshold :
       act = 0 # to stop
   else :
       if abs(dy) >= abs(dx) :
           if dy >= 0 :  # to up
              if abs(dy) > stop_threshold and abs(dy) <= normal_threshold:
                 act = 1 # 1x up
              else:
                 act = 2 # 2x up
           else :        # to down
              if abs(dy) > stop_threshold and abs(dy) <= normal_threshold:
                 act = 3 # 1x down
              else:
                 act = 4 # 2x down
       elif abs(dx) > abs(dy) :
           if dx >= 0 :  # to left
              if abs(dx) > stop_threshold and abs(dx) <= normal_threshold:
                 act = 5 # 1x left
              else:
                 act = 6 # 2x left
           else :        # to right
              if abs(dx) > stop_threshold and abs(dx) <= normal_threshold:
                 act = 7 # 1x right
              else:
                 act = 8 # 2x right

   sample_action = np.zeros((1,9))
   sample_action[0,act] = 1.
   
   SL_img.append(sample_img)
   SL_action.append(sample_action)

with open('new_MNIST_SL_data_digit_'+str(np.argmax(sample_y)), 'wb') as fp:
     pickle.dump([SL_img, SL_action] , fp)      