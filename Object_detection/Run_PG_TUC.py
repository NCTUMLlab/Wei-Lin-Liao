#import tensorflow as tf
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from image_helper import *
from parse_xml_annotations import *
from features import *
from reinforcement import *
from metrics import *
from collections import namedtuple
import time
import os
import numpy as np
import random
from Agent import PG
from Agent import TUC


# Device settings for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path_voc = "../datas/VOCdevkit/VOC2007"

# get models 
print("load models")
model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cuda()

agent = PG(0.0002,0.90)
TUC_dynamic = TUC(0.001)


# get image datas
path_voc_1 = "../datas/VOCdevkit/VOC2007"
class_object = '1'
image_names_1, images_1 = load_image_data(path_voc_1, class_object)
image_names = image_names_1 
images = images_1

print("aeroplane_trainval image:%d" % len(image_names))


# define the Pytorch Tensor
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# define the super parameter
CLASS_OBJECT = 1
steps = 5
epochs = 20


# train procedure
print('Train the policy network')
print("Image num : ",len(image_names))

# save init model 
#TUC_dynamic.save_model("./model_init/pg_TUC_2")

# load init model
agent.load_model("./model_init/pg_agent_2")
TUC_dynamic.load_model("./model_init/pg_TUC_2")

# exp settings
ratio_1 = 0.01
ratio_2 = 0.3

#'''
for epoch in range(epochs):
    print('epoch: %d' %epoch)
    now = time.time()
    for i in range(len(image_names)):
        # the image part
        image_name = image_names[i]
        image = images[i]
       
        annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_1)
                  
        classes_gt_objects = get_ids_objects_from_annotation(annotation)
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape) 
         
        # the iou part
        original_shape = (image.shape[0], image.shape[1])
        region_mask = np.ones((image.shape[0], image.shape[1]))
        #choose the max bouding box
        iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
        
        # the initial part
        region_image = image
        size_mask = original_shape
        offset = (0, 0)
        history_vector = torch.zeros((4,6))
        state = get_state(region_image, history_vector, model_vgg)

        #==================== loop of training procedure ==========================================#
        done = False
        actions_matrix = []
        for step in range(steps):

            # Select action, the author force terminal action if case actual IoU is higher than 0.5
            if iou > 0.5 or step == steps-1:
                action = 6
            else:
                action = agent.select_action(state)+1 # select_action(state) 
            
            
            intrinsic_reward = 0
            penalty = 0
            
            # Perform the action and observe new state
            if action == 6:
                next_state = None
                if iou > 0.5:
                   reward = 6
                else:
                   reward = -6 
                done = True
                
                action_np_vec = np.zeros([1,6])
                action_np_vec[0,action-1] = 1.
                action_vec = torch.from_numpy(action_np_vec).float().cuda()
                actions_matrix.append(action_vec)
                
                
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                   region_image, size_mask, action)
                # update history vector and get next state
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)
                
                # find the max bounding box in the region image
                new_iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
                reward = get_reward_movement(iou, new_iou)
                iou = new_iou
                
                
                
                
                # update model-based module
                action_np_vec = np.zeros([1,6])
                action_np_vec[0,action-1] = 1.
                action_vec = torch.from_numpy(action_np_vec).float().cuda()
                actions_matrix.append(action_vec)
                
                
                if step == 0:
                   pre_mean, pre_std = TUC_dynamic.dump_z_mean_std(state, action_vec)
                
                TUC_dynamic.train_enc_dec(state, next_state, action_vec)
                
                if step > 0:                
                   mean, std = TUC_dynamic.dump_z_mean_std(state, action_vec)
                   intrinsic_reward = TUC_dynamic.dump_exploration_reward(pre_mean, pre_std, mean, std)
             
            if i > 0 :      
              penalty = TUC_dynamic.dump_regret(state, action-1)
              
            
            #print(intrinsic_reward,penalty)
            #print(ratio_1*((epochs-epoch)/epochs)*intrinsic_reward - ratio_2*(epoch/epochs)*penalty)
            
            # Store the transition in memory   
            agent.store_transition(state, action-1, reward + ratio_1*(epochs-epoch/epochs)*intrinsic_reward - ratio_2*(epoch/epochs)*penalty)

            print('epoch: %d, image: %d, step: %d, reward: %d' %(epoch ,i, step, reward))    

            
            # Move to the next state
            state = next_state

            # Perform the optimization 
            if done:
                
               states = torch.cat(agent.states)
               values = torch.tensor(np.expand_dims(np.array(agent.get_values()), axis = 1), dtype=torch.float32)
               
               actions_matrix = torch.cat(actions_matrix,0).cuda()
               #print(len(actions_matrix))
               for e in range(3):
                   TUC_dynamic.train_critic(states.cuda(), values.cuda(), actions_matrix.cuda())
            
            
               print("updating agent !")
               agent.REINFORCE()
              
               break

        #==================== loop of training procedure ==========================================#


    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' %(epoch, time_cost))
    
# save the whole model
agent.save_model("./model_final/pg_TUC_agent_2")
TUC_dynamic.save_model("./model_final/pg_TUC_2")
print('Complete')
#'''