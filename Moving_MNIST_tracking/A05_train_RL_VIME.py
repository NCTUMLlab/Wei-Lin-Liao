import tensorflow as tf
import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt
import os
from A09_Nets import VIME

with open('new_MNIST_SL_data_digit_7', 'rb') as fp:
        SL_img, SL_action = pickle.load(fp)    

with open('new_MNIST_RL_data_digit_7', 'rb') as fp:
        RL_img, RL_gd, _,  _ = pickle.load(fp)  


# Save and load model parameter
def save_model(sess,saver,path,model_name):
    path_and_model_name = path + '/' + model_name + '.ckpt'
    _ = saver.save(sess, path_and_model_name)
    print("Model is saved!")

def load_model(sess,saver,path,model_name):
    path_and_model_name = path + '/' + model_name + '.ckpt'
    saver.restore(sess, path_and_model_name)
    print("Model is loaded!")

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
      plt.savefig('./new_'+stage+'_tracking_outcome_imgs/'+filename)
      plt.close()
 

# Function to compute the intersection-of-union
def IoU(box_pos,gd):

    x1i = gd[0]
    y1i = gd[1]
    x2i = gd[0]+gd[2]
    y2i = gd[1]+gd[3]

    x1j = box_pos[0]
    y1j = box_pos[1]
    x2j = box_pos[0]+box_pos[2]              
    y2j = box_pos[1]+box_pos[3]

    areai = (x2i-x1i+1)*(y2i-y1i+1)
    areaj = (x2j-x1j+1)*(y2j-y1j+1)

    xx1 = max(x1i, x1j)
    yy1 = max(y1i, y1j)
    xx2 = min(x2i, x2j)
    yy2 = min(y2i, y2j)

    h = max(0, yy2-yy1+1)
    w = max(0, xx2-xx1+1)

    intersection = w * h
    iou = intersection / (areai + areaj - intersection)
              
    return iou

# KL divergence
def KL_divergence(mean_1,log_std_1,mean_2,log_std_2):    

      term_1 = np.sum(np.square(np.divide(np.exp(log_std_1),np.exp(log_std_2)))) 
      term_2 = np.sum(2*log_std_2-2*log_std_1)
      term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(np.exp(log_std_2))))
          
      return np.maximum(0,0.5*(term_1+term_2+term_2-1))   

# Function to move box
def act_move_box(act,box_x,box_y,shift_size):

    if act == 0:
       next_x, next_y = box_x, box_y
    elif act == 1:
       next_x, next_y = box_x, box_y-shift_size
    elif act == 2:
       next_x, next_y = box_x, box_y-2*shift_size
    elif act == 3:
       next_x, next_y = box_x, box_y+shift_size
    elif act == 4:
       next_x, next_y = box_x, box_y+2*shift_size
    elif act == 5:
       next_x, next_y = box_x-shift_size, box_y
    elif act == 6:
       next_x, next_y = box_x-2*shift_size, box_y   
    elif act == 7:
       next_x, next_y = box_x+shift_size, box_y
    elif act == 8:
       next_x, next_y = box_x+2*shift_size, box_y
       
    # Check x range (100-28=72)
    if next_x > 71 :
       box_x = 71
    elif next_x < 0 :
       box_x = 0
    else :
       box_x = next_x
                 
    # Check y range
    if next_y > 71 :
       box_y = 71
    elif next_y < 0 :
       box_y = 0
    else :
       box_y = next_y
       
    return box_x, box_y   

# Function to get return
def get_returns(ep_rewards):    
    ep_returns = np.zeros_like(ep_rewards)
    running_add = 0
    for t in reversed(range(0, len(ep_rewards))):
        running_add = running_add * reward_decay + ep_rewards[t]
        ep_returns[t] = running_add

    #ep_returns -= np.mean(ep_returns)
    #ep_returns /= np.std(ep_returns)

    ep_returns -= 0.5*(np.max(ep_returns)+np.min(ep_returns))
    ep_returns /= (np.max(ep_returns)-np.min(ep_returns))

    return ep_returns

# Get parameters of network
def get_parameters(parameter_name, arch_list, initialier):
    all_parameters = []
    layer_num = len(arch_list)
    with tf.variable_scope(parameter_name,reuse=False):
         for layer in range(layer_num):
             temp_parameter = tf.get_variable(parameter_name+'_parameter_'+str(layer),
                                              shape=arch_list[layer],
                                              dtype=tf.float32,
                                              initializer=initialier)
             all_parameters.append(temp_parameter)
             
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=parameter_name)
    return all_parameters, vars

# Define the convolutional operation
def conv_layer(x,W,b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.max_pool(tf.nn.relu(x), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

# Get parameters of convolutional neural network
def get_CNN_parameters(name):
    all_Ws, W_vars = get_parameters(name+'_W_', 
                                    [[5,5,1,32],
                                     [5,5,32,64],
                                     [7*7*64, 1024],
                                     [1024,512]],
                                    tf.contrib.layers.xavier_initializer())
    all_bs, b_vars = get_parameters(name+'_b_', 
                                    [[32],[64],[1024],[512]],
                                    tf.contrib.layers.xavier_initializer())

    all_parameters = [all_Ws,all_bs]
    all_vars = W_vars+b_vars
    
    return all_parameters, all_vars
 
# Get parameters of gated recurrent unit   
def get_GRU_parameters(name):
   
    all_Ws, W_vars = get_parameters(name+'_W_', 
                                    [[512, 256],
                                     [512, 256],
                                     [512, 256]],
                                    tf.contrib.layers.xavier_initializer())
    all_Us, U_vars = get_parameters(name+'_U_', 
                                    [[256, 256],
                                     [256, 256],
                                     [256, 256]],
                                    tf.contrib.layers.xavier_initializer())
    
    all_bs, b_vars = get_parameters(name+'_b_', 
                                    [[256],
                                     [256],
                                     [256]],
                                    tf.contrib.layers.xavier_initializer())
    
    all_OW, OW_vars = get_parameters(name+'_O_w_', 
                                     [[256,9]],
                                     tf.contrib.layers.xavier_initializer())
    
    all_Ob, Ob_vars = get_parameters(name+'_O_b_', 
                                     [[9]],
                                     tf.contrib.layers.xavier_initializer())
                                     
    all_parameters = [all_Ws, all_Us, all_bs, all_OW, all_Ob]
    all_vars = W_vars + U_vars + b_vars + OW_vars + Ob_vars

    return all_parameters, all_vars    
   
# Define convolutional neural network                                    
def CNN(name,input_tensor,CNN_all_paremeters):
    
    all_Ws = CNN_all_paremeters[0]
    all_bs = CNN_all_paremeters[1]
    
    # Convolution
    conv1 = conv_layer(input_tensor,all_Ws[0], all_bs[0])
    conv2 = conv_layer(conv1,all_Ws[1], all_bs[1])

    # Dense
    fc1 = tf.reshape(conv2, [-1, all_Ws[2].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, all_Ws[2]), all_bs[2])
    fc1 = tf.nn.sigmoid(fc1)
    fc2 = tf.add(tf.matmul(fc1, all_Ws[3]), all_bs[3])
    
    return fc2 
    
# Define gated recurrent unit    
def GRU(name,input_tensor,h,GRU_all_parameters):    
    
    all_Ws = GRU_all_parameters[0]
    all_Us = GRU_all_parameters[1]
    all_bs = GRU_all_parameters[2]
    all_OW = GRU_all_parameters[3]
    all_Ob = GRU_all_parameters[4]
    
    z = tf.nn.sigmoid(tf.add(tf.matmul(input_tensor,all_Ws[0]),tf.add(tf.matmul(h,all_Us[0]),all_bs[0])))
    r = tf.nn.sigmoid(tf.add(tf.matmul(input_tensor,all_Ws[1]),tf.add(tf.matmul(h,all_Us[1]),all_bs[1])))
    h = tf.add(tf.multiply(z,h),tf.multiply((1-z),tf.nn.sigmoid(tf.add(tf.matmul(input_tensor,all_Ws[2]),tf.add(tf.matmul(tf.multiply(r,h),all_Us[2]),all_bs[2])))))
    return tf.nn.softmax(tf.add(tf.matmul(h,all_OW[0]),all_Ob[0]))


      
    
# Get parameters of CNN and GRU
CNN_all_paremeters, CNN_vars = get_CNN_parameters('CNN_parameters_')
GRU_all_parameters, GRU_vars = get_GRU_parameters('GRU_parameters_')

# Define and build placeholder of network
img_H = 28
img_W = 28
img_C = 1

X = tf.placeholder('float', [None,img_H,img_W,img_C])
Y = tf.placeholder("float", [None,9])
action = tf.placeholder(tf.int32,[None,])
value = tf.placeholder(tf.float32,[None,])

search_step = 3
shift_size = 5

global reward_decay
reward_decay = 0.9

# Initialization of hidden variables of GRU
GRU_h = tf.Variable(tf.zeros([1,256]),trainable=False)    

# Build networks and graph
CNN_output = CNN('CNN',X,CNN_all_paremeters)
pred = GRU('GRU',CNN_output,GRU_h,GRU_all_parameters)
neg_log_pred = tf.reduce_sum(-tf.log(pred)*tf.one_hot(action, 9),axis=1)

# Define and build loss function and optimizer
SL_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y)) 
SL_optimizer = tf.train.AdamOptimizer(learning_rate=0.00004).minimize(SL_loss, var_list = CNN_vars+GRU_vars)    

RL_loss = tf.reduce_mean(neg_log_pred*value)    
RL_optimizer = tf.train.AdamOptimizer(learning_rate=0.00004).minimize(RL_loss, var_list = GRU_vars)

# Accuracy 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))     
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   

# Initialization of all variables in graph
init = tf.global_variables_initializer()

# Graph saver
saver = tf.train.Saver()

# Settings of training procedure
episodes = 20
RL_train_num = 100
frame_idx = 1
mem_size = 5
critic_epoch = 5
ratio_init_1 = 0.01
        
with tf.Session() as sess:

     sess.run(init) 

     # Epoch IoU
     episode_IoU = []     
     
     # Overlap threshold resolution
     threshold_res = 0.001
     
     # One-Pass Evaluation ( OPE )
     RL_ope_curve = []
     RL_box_pos = []
     RL_gd_pos = []

     frame_idx = 1

     # Initialization of position of tracking box 
     box_x, box_y = RL_gd[0][0]+randint(-3,3), RL_gd[0][1]+randint(-3,3)

     # Reinforcement learning
     load_model(sess,saver,'./model_final','SL_01')

     # ** Build VIME
     vime = VIME(sess,saver,"VIME",0.001)
     vime.save_model("./model_init/VIME1")

     for e in range(episodes):   
         frame_idx = 1
         temp_total_reward = 0.
         total_IoU = 0.
         box_x, box_y = RL_gd[0][0]+randint(-3,3), RL_gd[0][1]+randint(-3,3)
         ep_states = []
         ep_actions = []
         ep_rewards = []   
         ep_in_reward = []   
         # **    
         ratio_1 = ratio_init_1*(episodes-e)/episodes  
         
         # ** Memory for latent features
         states_actions_mem = []
         next_states_mem = []

         # **Memory for critic
         states = []
         actions = []
         values = []
 
         t = 0

         for f in range(RL_train_num):
           frame_gd_x, frame_gd_y = RL_gd[f][0], RL_gd[f][1]
           RL_gd_pos.append([frame_gd_x, frame_gd_y])
           frame_img = RL_img[f]
           frame_IoU = 0.
           for s in range(search_step):
                 step_img = frame_img[:,box_y:box_y+28,box_x:box_x+28,:]
                 GRU_pred = sess.run(pred,feed_dict={X:step_img})
 
                 # Dense features
                 if t == 0:
                    state = sess.run(CNN_output,feed_dict={X:step_img})
                 else:
                    next_state = sess.run(CNN_output,feed_dict={X:step_img})

                 # Select action : Categorical sample
                 GRU_pred = np.squeeze(GRU_pred)
                 GRU_pred = GRU_pred/GRU_pred.sum().astype(float)
                 act = np.random.choice(range(9), p=GRU_pred)    
                 
                 # Move box
                 box_x, box_y = act_move_box(act, box_x, box_y, shift_size)
                 RL_box_pos.append([box_x, box_y])

                 # Store states 
                 if t > 0:
                    states.append(state) 
                    
                 # Store actions
                 if t > 0:
                    temp_state_action = np.zeros([1,512+9])
                    temp_state_action[0,512+act] = 1.
                    actions.append(temp_state_action)  

                 

                 # Get intrinsic reward
                 if t > 0:
                    # Get intrinsic reward
                    if t <= mem_size :
                      states_actions_mem.append(temp_state_action)
                      next_states_mem.append(next_state)
                      if t == mem_size :
                        pre_hyper_parameters = vime.get_hyper_parameters()
                        intrinsic_reward_1 = 0
                    else : 
                      vime.train(np.vstack(states_actions_mem), np.vstack(next_states_mem))  
                      hyper_parameters = vime.get_hyper_parameters()
                      intrinsic_reward_1 = vime.get_info_gain(hyper_parameters, pre_hyper_parameters)   
                      pre_hyper_parameters = hyper_parameters
                      # Update memory
                      states_actions_mem.append(temp_state_action)
                      next_states_mem.append(next_state)
                      states_actions_mem.pop(0)
                      next_states_mem.pop(0)
                 else:
                    intrinsic_reward_1 = 0
                    
                 #print(intrinsic_reward_1)                 

                 

                 # Get reward
                 if s < search_step-1 :
                    ext_reward = 0.
                    intrinsic_reward_1 = 0
                 else :
                    if IoU([box_x,box_y,28,28],[frame_gd_x,frame_gd_y,28,28]) > 0.7 :
                       ext_reward = 1.  
                    else:
                       ext_reward = -0.5
                    temp_total_reward += ext_reward
                 
                 temp_IoU = IoU([box_x,box_y,28,28],[frame_gd_x,frame_gd_y,28,28]) 
                 #print(temp_IoU)  
                 frame_IoU += temp_IoU
                 total_IoU += temp_IoU

                 # Store states, actions, rewards
                 ep_states.append(step_img)
                 ep_actions.append(act)
                 ep_in_reward.append(ratio_1*intrinsic_reward_1)
                 ep_rewards.append(ext_reward)

                 # Store reward
                 if t > 0:
                    values.append(ext_reward) 
                 
                 if t > 0:
                   state = next_state

                 # Update time step
                 t += 1
                 
                 # Save outcome
                 #if e == episodes-1:
                    #save_tracking_process(frame_idx,frame_img,[box_x,box_y],[frame_gd_x,frame_gd_y],'RL')
                              
                    # Next image
                    #frame_idx +=1
                 
           #print("Episode = ",e+1," , frame = ", f+1," , frame IoU = ",frame_IoU/search_step)     
                   
           if e == episodes-1:
              episode_IoU.append(frame_IoU/search_step)
              
         print("Episode = ",e+1,' , total IoU = ',total_IoU) 
         #RL_total_reward.append(temp_total_reward)     

         ep_in_reward = np.array(ep_in_reward)
         ep_in_reward/= np.max(ep_in_reward)
         ep_in_reward*= 0.5
         ep_in_reward = ep_in_reward.tolist()

         pratical_returns = get_returns(ep_rewards) + get_returns(ep_in_reward)
         
         

         sess.run(RL_optimizer,
                  feed_dict={X: np.vstack(ep_states),
                             action: np.array(ep_actions), 
                             value: pratical_returns})                  
         
         ep_states = []
         ep_actions = []
         ep_rewards = []  
         ep_in_reward = []


           
     
     vime.save_model("./model_final/VIME1")
     
     # Plot OPE
     for thresold in np.arange(0,1+threshold_res,threshold_res):
         temp_precision = 0.
         for n in range(RL_train_num):
             if episode_IoU[n] >= thresold :
                temp_precision += 1
             
         temp_precision /= RL_train_num
         RL_ope_curve.append(temp_precision)
     
   
     # RL OPE X and Y of figure
     precision_threshold_x = np.arange(0,1+threshold_res,threshold_res)
     RL_ope_curve_y = RL_ope_curve

     with open("./tracking_results/RL_VIME_OPE","wb") as fp:
            pickle.dump(RL_ope_curve,fp)

     with open("./tracking_results/RL_VIME_box_pos","wb") as fp:
            pickle.dump(RL_box_pos,fp)

     with open("./tracking_results/RL_VIME_gd_pos","wb") as fp:
            pickle.dump(RL_gd_pos,fp)

     with open("./tracking_results/SL_OPE","rb") as fp:
            SL_ope_curve_y = pickle.load(fp)

     with open("./tracking_results/RL_OPE","rb") as fp:
            pure_RL_ope_curve_y = pickle.load(fp) 

     with open("./tracking_results/RL_TUC_OPE","rb") as fp:
            tuc_RL_ope_curve_y = pickle.load(fp) 

     plt.figure(5)
     plt.title('Tracking MNIST')
     plt.plot(precision_threshold_x,SL_ope_curve_y,'r',label='SL',linewidth=3)
     plt.plot(precision_threshold_x,pure_RL_ope_curve_y,'b',label='SL+RL',linewidth=3)
     plt.plot(precision_threshold_x,tuc_RL_ope_curve_y,'g',label='SL+RL(TUC)',linewidth=3)
     plt.plot(precision_threshold_x,RL_ope_curve_y,'m',label='SL+RL(VIME)',linewidth=3)
     plt.xlabel('Overlap threshold')
     plt.ylabel('Success rate')
     plt.xlim(0,1)
     plt.legend()
     plt.savefig('./tracking_results/RL_VIME_OPE')
     plt.show()
     
     save_model(sess,saver,'./model_final','RL_VIME_01')
     
     