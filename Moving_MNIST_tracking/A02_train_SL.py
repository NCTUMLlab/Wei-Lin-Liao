import tensorflow as tf
import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt
import os

with open('new_MNIST_SL_data_digit_7', 'rb') as fp:
        SL_img, SL_action = pickle.load(fp)    
        #print(SL_img.shape,SL_gd.shape) 

with open('new_MNIST_RL_data_digit_7', 'rb') as fp:
        RL_img, RL_gd, _,  _ = pickle.load(fp)  
        #print(RL_img[0].shape,RL_gd.shape)


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

    ep_returns -= np.mean(ep_returns)
    ep_returns /= np.std(ep_returns)
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
SL_optimizer = tf.train.AdamOptimizer(learning_rate=0.00002).minimize(SL_loss, var_list = CNN_vars+GRU_vars)    

RL_loss = tf.reduce_mean(neg_log_pred*value)    
RL_optimizer = tf.train.AdamOptimizer(learning_rate=0.00004).minimize(RL_loss, var_list = GRU_vars+GRU_vars)

# Accuracy 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))     
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   

# Initialization of all variables in graph
init = tf.global_variables_initializer()

# Graph saver
saver = tf.train.Saver()

# Settings of training procedure
epochs = 3
batch_num = len(SL_img)
SL_test_num = 100

frame_idx = 1

    
        
with tf.Session() as sess:

     sess.run(init) 

     # Supervised training , batch size is always 1
     save_model(sess,saver,'./model_init','SL_01')
     for e in range(epochs):
         temp_train_loss = 0.
         temp_train_acc = 0.
         for b in range(batch_num):
             batch_x, batch_y = SL_img[b], SL_action[b] #mnist.train.next_batch(1)
             _,temp_loss = sess.run((SL_optimizer,SL_loss),feed_dict={X:batch_x,Y:batch_y})
             temp_acc = sess.run(accuracy,feed_dict={X:batch_x,Y:batch_y})
             temp_train_loss+=temp_loss
             temp_train_acc+=temp_acc
         temp_train_loss/=batch_num
         temp_train_acc/=batch_num
         print("Epoch = ",e+1," , avg loss = ",temp_train_loss,' , avg acc = ',temp_train_acc)  
         
     # Save and load model 
     save_model(sess,saver,'./model_final','SL_01')
     
     
     
     # Epoch IoU
     epoch_IoU = []     
     
     # Overlap threshold resolution
     threshold_res = 0.001
     
     # One-Pass Evaluation ( OPE )
     SL_ope_curve = []

     # Supervised testing
     frame_idx = 1

     # Initialization of position of tracking box 
     box_x, box_y = RL_gd[0][0]+randint(-3,3), RL_gd[0][1]+randint(-3,3)

     # Supervised learning box position
     SL_box_pos = [[box_x, box_y]]
     SL_gd_pos = [[RL_gd[0][0],RL_gd[0][1]]]
     
     total_IoU = 0.
     for f in range(SL_test_num):
           frame_gd_x, frame_gd_y = RL_gd[f][0], RL_gd[f][1]
           frame_img = RL_img[f]
           frame_IoU = 0.
           for s in range(search_step):
                 step_img = frame_img[:,box_y:box_y+28,box_x:box_x+28,:]
                 GRU_pred = sess.run(pred,feed_dict={X:step_img})
                 
                 # Select action : Categorical sample
                 GRU_pred = np.squeeze(GRU_pred)
                 GRU_pred = GRU_pred/GRU_pred.sum().astype(float)
                 act = np.random.choice(range(9), p=GRU_pred)    
                 
                 # Move box
                 box_x, box_y = act_move_box(act, box_x, box_y, shift_size)
                 SL_box_pos.append([box_x, box_y])
                 SL_gd_pos.append([frame_gd_x,frame_gd_y])

                 # Score is IoU , larger is better
                 score = IoU([box_x,box_y,28,28],[frame_gd_x,frame_gd_y,28,28])
                 
                 # Score per frame
                 frame_IoU+= score
                 total_IoU += score

                 # Save outcome
                 #save_tracking_process(frame_idx,frame_img,[box_x,box_y],[frame_gd_x,frame_gd_y],'SL')
                
                 # Next image
                 #frame_idx +=1

           print("Frame = ", f+1," , IoU = ",frame_IoU/search_step)
           epoch_IoU.append(frame_IoU/search_step)
     
     print("Total IoU = ",total_IoU)
     
     # Plot OPE
     for thresold in np.arange(0,1+threshold_res,threshold_res):
         temp_precision = 0.
         for n in range(SL_test_num):
             if epoch_IoU[n] >= thresold :
                temp_precision += 1
             
         temp_precision /= SL_test_num
         SL_ope_curve.append(temp_precision)
     
   
     # SL OPE X and Y of figure
     precision_threshold_x = np.arange(0,1+threshold_res,threshold_res)
     SL_ope_curve_y = SL_ope_curve

     with open("./tracking_results/SL_OPE","wb") as fp:
            pickle.dump(SL_ope_curve,fp)

     with open("./tracking_results/SL_box_pos","wb") as fp:
            pickle.dump(SL_box_pos,fp)

     with open("./tracking_results/SL_gd_pos","wb") as fp:
            pickle.dump(SL_gd_pos,fp)

     plt.figure(5)
     plt.title('Tracking MNIST')
     plt.plot(precision_threshold_x,SL_ope_curve_y,'b',label='SL')
     plt.xlabel('Overlap threshold')
     plt.ylabel('Success rate')
     plt.xlim(0,1)
     plt.legend()
     plt.savefig('./tracking_results/SL_OPE')
     plt.show()
     
     
     
     