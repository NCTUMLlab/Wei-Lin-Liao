import numpy as np
import tensorflow as tf


############################################################################
#==========================================================================#
# Input :                                                      
#
#  - net_name : scope name of DNN ( type : string )
#
#  - arch_list : network architecture ( type : list of int )
#
#               [ input_width , layer_1_width , ... , final_layer_width ]  
#
#  - input_tensor : input for DNN ( tyep : tensor or placeholder )   
#
#  - act_func : activation function of DNN  ( type : tf.nn.XXX )
#
#               tf.nn.relu - relu ( default )
#
#  - var_init : initialization for variables 
#
#               tf.contrib.layers.xavier_initializer() ( default )
#
#               tf.random_normal_initializer() 
#
#
#  - reuse : initialization for scope ( type : True or False )
#
#            False ( default )
#
#  - dtype : data type of variables ( type : tf.float32 or .. )
#
#            tf.float32 ( default )
#
# Member :
#
#  - net_Ws : weights of DNN ( type : list of tensor variable )
#
#  - net_bs : biases of DNN ( type : list of tensor variable )
#  
#  - net_hs : hidden states of DNN ( type : list of tensor )
#
#  - net_vars : collect all variables of network  
#
#  - pred : output of DNN ( type : tensor without act_func ) 
#
#  - softmax_pred : softmax output of DNN 
#
#                   ( type : tensor without act_func )
#
# 
#==========================================================================#
  

class Standard_DNN:
          
      # Constructor of DNN    
      def __init__(self,
                         net_name,
                         arch_list,
                         input_tensor,
                         act_func = tf.nn.relu,
                         var_init = tf.contrib.layers.xavier_initializer(),
                         reuse = False,
                         dtype = tf.float32):
      
            self.net_name = net_name   # scope name of DNN
            self.arch_list = arch_list # architecture list
            self.input_tensor = input_tensor # input tensor
            self.act_func = act_func   # activation function
            self.var_init = var_init   # initialization of variables
            self.reuse = reuse # initialize the scope or reuse
            self.dtype = dtype # data type of varialbes
          
            self.layer_num = len(self.arch_list) # from input to output
            self.net_Ws = [] # initialization of list to store weights of DNN
            self.net_bs = [] # initialization of list to store biases of DNN
            self.net_hs = [] # initialization of list to store hidden states of DNN
          
            # Create weights and biases under network scope
            with tf.variable_scope(self.net_name,reuse=self.reuse):  
                 
                 print("Start construct ",self.net_name," network !!")
                 # Initialization layer by layer    
                 for layer in range(1,self.layer_num,1):
               
                     W_temp = tf.get_variable(self.net_name+'_W'+str(layer),
                                                           shape=[self.arch_list[layer-1],
                                                           self.arch_list[layer]],
                                                           dtype=self.dtype,
                                                           initializer=self.var_init)
                                            
                     b_temp = tf.get_variable(self.net_name+'_b'+str(layer),
                                                          shape=[self.arch_list[layer]],
                                                          dtype=self.dtype,
                                                          initializer=self.var_init)
                                            
                     # Save weights and biases to list
                     self.net_Ws.append(W_temp)
                     self.net_bs.append(b_temp)

                     # Connet the layers    
                     if layer == 1:   # first layer 
                   
                        h_temp = tf.add(tf.matmul(self.input_tensor,W_temp),b_temp)
                        self.net_hs.append(self.act_func(h_temp))
                 
                     elif layer > 1 and layer < self.layer_num-1 :   # middle layers
                   
                        h_temp = tf.add(tf.matmul(h_temp,W_temp),b_temp)
                        self.net_hs.append(self.act_func(h_temp))
                 
                     elif layer == self.layer_num-1 :   # final layer
                   
                        h_temp = tf.add(tf.matmul(h_temp,W_temp),b_temp)
                        self.net_hs.append(h_temp) 
                   
                     # Show layer inforamation                       
                     print("Dense layer "+str(layer+1)+" , weight shape = ",
                             W_temp.get_shape(),
                             " , biase shape = ",
                             b_temp.get_shape(),
                             " , feature shape = ",
                             h_temp.get_shape())
                        
                 # Configure output  
                 self.pred = self.net_hs[-1]    
                 self.softmax_pred = tf.nn.softmax(self.pred)

            # Collect network variables
            self.net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net_name)

#################################################################################################################################

#===== Convolutional Neural Network ======================================================================================================#
# Input :                                                      
#
#  - net_name : scope name of CNN ( type : string )
#
#  - conv_arch_list : convolution layer architecture ( type : list of list int )
#
#    [[ layer_1_filter_h , layer_1_filter_w , layer_1_in_channel , layer_1_filter_num , layer_1_strides , layer_1_pool_size ],
#     [ layer_2_filter_h , layer_2_filter_w , layer_2_in_channel , layer_2_filter_num , layer_2_strides , layer_2_pool_size ],  
#       ...       
#     [ layer_c_filter_h , layer_c_filter_w , layer_c_in_channel , layer_c_filter_num , layer_c_strides , layer_c_pool_size ]]
#
#  - dense_arch_list : dense layer architecture ( list of int )  
#
#    [ dense_layer_1 , dense_layer_2 ... , dense_layer_d ]
#
#  - input_tensor : input for CNN ( tyep : tensor or placeholder )   
#
#  - act_func : activation function of CNN  ( type : tf.nn.XXX )
#
#               tf.nn.relu - relu ( default )
#
#  - pool_func : pooling function of convolutional part ( type : tf.nn.XXXX_pool )
#
#                tf.nn.max_pool ( default )
#
#  - var_init : initialization for variables 
#
#               tf.contrib.layers.xavier_initializer_conv2d() ( default )            
#
#               tf.contrib.layers.xavier_initializer() 
#
#               tf.random_normal_initializer() 
#
#
#  - reuse : initialization for scope ( type : True or False )
#
#            False ( default )
#
#  - dtype : data type of variables ( type : tf.float32 or .. )
#
#            tf.float32 ( default )
#
# Member :
#
#  - net_conv_Ws : weights of CNN conv part( type : list of tensor variable )
#
#  - net_conv_bs : biases of CNN conv part ( type : list of tensor variable )
#  
#  - net_conv_hs : hidden states of CNN conv part ( type : list of tensor )
#
#  - net_dense_Ws : weights of CNN dense part( type : list of tensor variable )
#
#  - net_dense_bs : biases of CNN dense part ( type : list of tensor variable )
#  
#  - net_dense_hs : hidden states of CNN dense part ( type : list of tensor )
#
#  - net_vars : collect all variables of network  
#
#  - pred : output of CNN dense layer ( type : tensor without act_func ) 
#
#  - softmax_pred : softmax output of CNN dense layer 
#
#                   ( type : tensor without act_func )
#
# 
#===============================================================================================================================#

class Standard_CNN:
          
      # Constructor of CNN    
      def __init__(self,
                         net_name,
                         conv_arch_list,
                         dense_arch_list,
                         input_tensor,
                         act_func = tf.nn.relu,
                         pool_func = tf.nn.max_pool,
                         var_init = tf.contrib.layers.xavier_initializer_conv2d(),
                   
                         reuse = False,
                         dtype = tf.float32):
      
            self.net_name = net_name   # scope name of DNN
            self.conv_arch_list = conv_arch_list # conv architecture list
            self.dense_arch_list = dense_arch_list # dense architecture list
            self.input_tensor = input_tensor # input tensor
            self.act_func = act_func   # activation function
            self.pool_func = pool_func # pooling function
            self.var_init = var_init   # initialization of variables
          
            self.reuse = reuse # initialize the scope or reuse
            self.dtype = dtype # data type of varialbes
          
            self.conv_layer_num = len(self.conv_arch_list) # from input to end of conv part
            self.dense_layer_num = len(self.dense_arch_list) # from input to end of conv part
          
            self.net_conv_Ws = [] # initialization of list to store weights of CNN conv part
            self.net_conv_bs = [] # initialization of list to store biases of CNN conv part
            self.net_conv_hs = [] # initialization of list to store hidden states of CNN conv part
          
            self.net_dense_Ws = [] # initialization of list to store weights of CNN dense part
            self.net_dense_bs = [] # initialization of list to store biases of CNN dense part
            self.net_dense_hs = [] # initialization of list to store hidden states of CNN dense part
          
            # Create weights and biases under network scope
            with tf.variable_scope(self.net_name,reuse=self.reuse):  
               
                 print("Start construct ",self.net_name," network !!")
                 # Initialization " convolutional layer part " layer by layer    
                 for conv_layer in range(0,self.conv_layer_num,1):
               
                     conv_W_temp = tf.get_variable(self.net_name+'_conv_W'+str(conv_layer+1),
                                                                    shape=[self.conv_arch_list[conv_layer][0],  # filter height
                                                                    self.conv_arch_list[conv_layer][1],  # filter width
                                                                    self.conv_arch_list[conv_layer][2],  # filter input channel       
                                                                    self.conv_arch_list[conv_layer][3]], # filter num        
                                                                    dtype=self.dtype,
                                                                    initializer=self.var_init)
                                            
                     conv_b_temp = tf.get_variable(self.net_name+'_conv_b'+str(conv_layer+1),
                                                                  shape=[self.conv_arch_list[conv_layer][3]],  # filter num    
                                                                  dtype=self.dtype,
                                                                  initializer=self.var_init)
                                            
                     # Save weights and biases to list
                     self.net_conv_Ws.append(conv_W_temp)
                     self.net_conv_bs.append(conv_b_temp)

                     # Connet the layers    
                     if conv_layer == 0:   # first layer 
                   
                        conv_h_temp = tf.nn.conv2d(self.input_tensor, 
                                                                  conv_W_temp, 
                                                                  strides=[1,self.conv_arch_list[conv_layer][4],self.conv_arch_list[conv_layer][4], 1], 
                                                                  padding='SAME')
                                                 
                                                 
                        conv_h_temp = tf.nn.bias_add(conv_h_temp, conv_b_temp)
                        conv_h_temp = self.act_func(conv_h_temp)
                        conv_h_temp = self.pool_func(conv_h_temp, 
                                                                     ksize=[1, self.conv_arch_list[conv_layer][5], self.conv_arch_list[conv_layer][5], 1], 
                                                                     strides=[1, self.conv_arch_list[conv_layer][5], self.conv_arch_list[conv_layer][5], 1],
                                                                     padding='SAME')
                      
                        self.net_conv_hs.append(conv_h_temp)
                 
                     else :   # middle to final layers
                   
                        conv_h_temp = tf.nn.conv2d(conv_h_temp, 
                                                                  conv_W_temp, 
                                                                  strides=[1,self.conv_arch_list[conv_layer][4],self.conv_arch_list[conv_layer][4], 1], 
                                                                  padding='SAME')
                                                 
                                                 
                        conv_h_temp = tf.nn.bias_add(conv_h_temp, conv_b_temp)
                        conv_h_temp = self.act_func(conv_h_temp)
                        conv_h_temp = self.pool_func(conv_h_temp, 
                                                                     ksize=[1, self.conv_arch_list[conv_layer][5], self.conv_arch_list[conv_layer][5], 1], 
                                                                     strides=[1, self.conv_arch_list[conv_layer][5], self.conv_arch_list[conv_layer][5], 1],
                                                                     padding='SAME')
                      
                        self.net_conv_hs.append(conv_h_temp)
               
                     # Show convolutional layer information
                     print("Conv layer "+str(conv_layer+1)+" , filter weight shape = ",
                              conv_W_temp.get_shape(),
                              " , filter biase shape = ",
                              conv_b_temp.get_shape(),
                              " , feature map shape = ",
                              conv_h_temp.get_shape())
                   
               
                 # Final conv layer property 
                 self.final_conv_layer = self.net_conv_hs[-1]
                 self.final_conv_layer_shape = self.final_conv_layer.get_shape().as_list()
                 self.final_conv_dense_length = self.final_conv_layer_shape[1]*self.final_conv_layer_shape[2]*self.final_conv_layer_shape[3] 
               
                 # Reshape conv layer       
                 self.first_dense_layer = tf.reshape(self.final_conv_layer, [-1, self.final_conv_dense_length])       
 
                 # Initialization " dense layer part " layer by layer    
                 for dense_layer in range(0,self.dense_layer_num,1):  
                   
                     # Initialization " dense layer part " layer by layer   
                     if dense_layer == 0:   # first layer 
                        pre_layer_width = self.final_conv_dense_length
                     else :                 # second layer to final layer
                        pre_layer_width = self.dense_arch_list[dense_layer-1]    
                   
                     # Create variables
                     dense_W_temp = tf.get_variable(self.net_name+'_dense_W'+str(dense_layer+1),
                                                                     shape=[pre_layer_width,       
                                                                     self.dense_arch_list[dense_layer]],         
                                                                     dtype=self.dtype,
                                                                     initializer=self.var_init)
                                            
                     dense_b_temp = tf.get_variable(self.net_name+'_dense_b'+str(dense_layer+1),
                                                                    shape=[self.dense_arch_list[dense_layer]],      
                                                                    dtype=self.dtype,
                                                                    initializer=self.var_init)
                                            
                     # Save weights and biases to list
                     self.net_dense_Ws.append(dense_W_temp)
                     self.net_dense_bs.append(dense_b_temp)
                   
                     # Connet the layers    
                     if dense_layer == 0:   # first layer 
                   
                        dense_h_temp = tf.add(tf.matmul(self.first_dense_layer,dense_W_temp),dense_b_temp)
                        self.net_dense_hs.append(self.act_func(dense_h_temp))
                 
                     elif dense_layer > 0 and dense_layer < self.dense_layer_num-1 :   # middle layers
                   
                        dense_h_temp = tf.add(tf.matmul(dense_h_temp,dense_W_temp),dense_b_temp)
                        self.net_dense_hs.append(self.act_func(dense_h_temp))
                 
                     elif dense_layer == self.dense_layer_num-1 :   # final layer
                   
                        dense_h_temp = tf.add(tf.matmul(dense_h_temp,dense_W_temp),dense_b_temp)
                        self.net_dense_hs.append(dense_h_temp) 
                      
                     # Show dense layer information
                     print("Dense layer "+str(dense_layer+1)+" , filter weight shape = ",
                              dense_W_temp.get_shape(),
                              " , filter biase shape = ",
                              dense_b_temp.get_shape(),
                              " , feature shape = ",
                              dense_h_temp.get_shape())                 
                   
                 # Configure output  
                 self.pred = self.net_dense_hs[-1]    
                 self.softmax_pred = tf.nn.softmax(self.pred)

            # Collect network variables
            self.net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net_name)    
             
             
############################################################################
#===== Bayes Backprop Neural Network ===============================================#
# Input :                                                      
#
#  - net_name : scope name of BBN ( type : string )
#
#  - arch_list : network architecture ( type : list of int )
#
#               [ input_width , layer_1_width , ... , final_layer_width ]  
#
#  - input_tensor : input for BBN ( tyep : tensor or placeholder )   
#
#  - act_func : activation function of DNN  ( type : tf.nn.XXX )
#
#                    tf.nn.relu - relu ( default )
#
#
#  - reuse : initialization for scope ( type : True or False )
#
#                False ( default )
#
#  - dtype : data type of variables ( type : tf.float32 or .. )
#
#               tf.float32 ( default )
#
# Member :
#
#  - net_W_means : means of weights of BBN ( type : list of tensor variable )
#
#  - net_W_logstds : logstds of weights of BBN ( type : list of tensor variable )
#
#  - net_Ws : weights of BBN ( type : list of tensor variable )
#  
#  - net_b_means : means of biases of BBN ( type : list of tensor variable )
#
#  - net_b_logstds : logstds of biases of BBN ( type : list of tensor variable )
#
#  - net_bs : biases of BBN ( type : list of tensor variable )
#
#  - net_hs : hidden states of BBN ( type : list of tensor )
#
#  - net_vars : collect all variables of network  
#
#  - pred : output of BBN ( type : tensor without act_func ) 
#
#  - softmax_pred : softmax output of BBN 
#
#                           ( type : tensor without act_func )
#
# 
#==========================================================================#
         
class Standard_BBN:
        
      # Constructor of BBN
      def __init__(self,
                         sess,
                         net_name,
                         arch_list,
                         input_tensor_shape,
						 target_tensor_shape,
						 epoch_num,
						 regression_or_classification = False,  # False : regression / True : classification
                         act_func = tf.nn.relu,
                         reuse = False,
                         dtype = 'float',
                         stddev = 0.1,
                         stddev_prior = tf.exp(-3.0)):
            
                    
            self.net_name = net_name 
            self.arch_list = arch_list
            self.input_tensor_shape = input_tensor_shape
            self.target_tensor_shape = target_tensor_shape
            self.epoch_num = epoch_num
            self.regression_or_classification = regression_or_classification
            self.act_func = act_func
            self.reuse = reuse
            self.dtype = dtype
            self.stddev = stddev
            self.stddev_prior = stddev_prior
            
            self.input_tensor = tf.placeholder(tf.float32,shape = input_tensor_shape)
            self.target_tensor = tf.placeholder(tf.float32,shape = target_tensor_shape)
            self.batch_idx = tf.placeholder(tf.float32,shape = None)
            
            self.layer_num = len(self.arch_list)
            self.net_W_means = []
            self.net_W_logstds = []
            self.net_Ws = []
            self.net_b_means = []
            self.net_b_logstds = []
            self.net_bs = []
			
            self.net_hs_noisy = []
            self.net_hs_mean = []

            self.sample_log_pw, self.sample_log_qw, self.sample_log_likelihood = 0. ,0. ,0.
            
            with tf.variable_scope(self.net_name,reuse=self.reuse):  
                 
                 print("Start construct ",self.net_name," network !!")
                 # Initialization layer by layer   
                 for layer in range(1,self.layer_num,1):
                  
                     W_mean_temp = tf.get_variable(self.net_name+'_W'+str(layer)+'_mean',
                                                                     dtype = self.dtype,
                                                                     initializer = tf.truncated_normal([self.arch_list[layer-1],self.arch_list[layer]],stddev = self.stddev))
                                                                     
                     W_logstd_temp = tf.get_variable(self.net_name+'_Wlogstd'+str(layer)+'_',
                                                                     dtype = self.dtype,
                                                                     initializer = tf.truncated_normal([self.arch_list[layer-1],self.arch_list[layer]],stddev = self.stddev))
                                                                     
                     b_mean_temp = tf.get_variable(self.net_name+'_b'+str(layer)+'_mean',
                                                                    dtype = self.dtype,
                                                                    shape = [self.arch_list[layer]],
                                                                    initializer = tf.zeros_initializer())
                                                                     
                     b_logstd_temp = tf.get_variable(self.net_name+'_b'+str(layer)+'_logstd',
                                                                     dtype = self.dtype,
                                                                     shape = [self.arch_list[layer]],
                                                                     initializer = tf.zeros_initializer())
                                                     
                     W_random_noise_temp = tf.random_normal([self.arch_list[layer-1],self.arch_list[layer]], mean = 0., stddev = self.stddev_prior)
                     b_random_noise_temp = tf.random_normal([self.arch_list[layer]], mean = 0., stddev = self.stddev_prior)  

                     W_temp = W_mean_temp + tf.multiply(tf.log(1. + tf.exp(W_logstd_temp)), W_random_noise_temp)                     
                     b_temp = b_mean_temp + tf.multiply(tf.log(1. + tf.exp(b_logstd_temp)), b_random_noise_temp)
                     
                     self.net_W_means.append(W_mean_temp)
                     self.net_W_logstds.append(W_logstd_temp)
                     self.net_Ws.append(W_temp)
                     self.net_b_means.append(b_mean_temp)
                     self.net_b_logstds.append(b_logstd_temp)
                     self.net_bs.append(b_temp)
					 
                 for layer in range(self.layer_num-1):        

                     self.sample_log_pw += tf.reduce_sum(self.log_gaussian(self.net_Ws[layer],0.,self.stddev_prior))    
                     self.sample_log_pw += tf.reduce_sum(self.log_gaussian(self.net_bs[layer],0.,self.stddev_prior))

                     self.sample_log_qw += tf.reduce_sum(self.log_gaussian_logstd(self.net_Ws[layer],self.net_W_means[layer],self.net_W_logstds[layer]*2))
                     self.sample_log_qw += tf.reduce_sum(self.log_gaussian_logstd(self.net_bs[layer],self.net_b_means[layer],self.net_b_logstds[layer]*2))
			

                 for layer in range(1,self.layer_num,1):
			     
                     if layer == 1:   
                   
                        h_temp_noisy = tf.add(tf.matmul(self.input_tensor,self.net_Ws[layer-1]),self.net_bs[layer-1])
                        self.net_hs_noisy.append(self.act_func(h_temp_noisy))
					  
                        h_temp_mean = tf.add(tf.matmul(self.input_tensor,self.net_W_means[layer-1]),self.net_b_means[layer-1])
                        self.net_hs_mean.append(self.act_func(h_temp_mean))
                 
                     elif layer > 1 and layer < self.layer_num-1 :   
                   
                        h_temp_noisy = tf.add(tf.matmul(h_temp_noisy,self.net_Ws[layer-1]),self.net_bs[layer-1])
                        self.net_hs_noisy.append(self.act_func(h_temp_noisy))
					  
                        h_temp_mean = tf.add(tf.matmul(h_temp_mean,self.net_W_means[layer-1]),self.net_b_means[layer-1])
                        self.net_hs_mean.append(self.act_func(h_temp_mean))
                 
                     elif layer == self.layer_num-1 :   
                   
                        h_temp_noisy = tf.add(tf.matmul(h_temp_noisy,self.net_Ws[layer-1]),self.net_bs[layer-1])
                        self.net_hs_noisy.append(h_temp_noisy) 
				 
                        h_temp_mean = tf.add(tf.matmul(h_temp_mean,self.net_W_means[layer-1]),self.net_b_means[layer-1])
                        self.net_hs_mean.append(self.act_func(h_temp_mean))
				
                 self.pred_noisy = self.net_hs_noisy[-1] 	 
                 self.softmax_pred_noisy = tf.nn.softmax(self.net_hs_noisy[-1])
			 
                 self.pred_mean = self.net_hs_mean[-1] 	
                 self.softmax_pred_mean = tf.nn.softmax(self.net_hs_mean[-1])
			 
			 
                 if self.regression_or_classification :
                    self.sample_log_likelihood = tf.reduce_sum(self.log_gaussian(self.target_tensor,self.softmax_pred_noisy,self.stddev_prior))
                 else:
                    self.sample_log_likelihood = tf.reduce_sum(self.log_gaussian(self.target_tensor,self.pred_noisy,self.stddev_prior))
				
                 self.log_pw = self.sample_log_pw
                 self.log_qw = self.sample_log_qw
                 self.log_likelihood = self.sample_log_likelihood
                 self.pi = (2**(self.epoch_num-self.batch_idx-1))/(2**self.epoch_num-1)

                 self.variational_free_energy =  tf.reduce_mean(self.pi * (self.log_qw - self.log_pw) - self.log_likelihood)
			     
             
            # Collect network variables
            self.net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net_name)
            
            self.sess = sess    
            
      def get_hyper_parameters(self,sess):
         
          hyper_parameters = sess.run([self.net_W_means, self.net_W_logstds, self.net_b_means, self.net_b_logstds])
          
          return hyper_parameters 
                
            
      def get_info_gain(self,hyper_parameters,pre_hyper_parameters):	 
		  
          W_means = hyper_parameters[0]
          W_logstds = hyper_parameters[1]
          b_means = hyper_parameters[2]  
          b_logstds = hyper_parameters[3]   
          
          
          pre_W_means = pre_hyper_parameters[0]
          pre_W_logstds = pre_hyper_parameters[1]
          pre_b_means = pre_hyper_parameters[2]  
          pre_b_logstds = pre_hyper_parameters[3]   

          length = len(pre_W_means)
          total_div = 0.
          
          for l in range(length):   
              W_div = self.KL_div(W_means[l],W_logstds[l],pre_W_means[l],pre_W_logstds[l])  
              b_div = self.KL_div(b_means[l],b_logstds[l],pre_b_means[l],pre_b_logstds[l])              
      
              total_div = total_div + W_div + b_div
              
          return total_div
      
      def KL_div(self,p_mean,p_std,q_mean,q_std):
      
          return np.mean(np.log(np.divide(np.log(1.+np.exp(q_std)),np.log(1.+np.exp(p_std)))) + np.divide(np.square(np.log(1.+np.exp(p_std)))+np.square(p_mean-q_mean),2*np.square(np.log(1.+np.exp(q_std)))) - 0.5)
      
      def log_gaussian(self,x,mean,std):
          
           return -0.5*np.log(2*np.pi)-tf.log(tf.abs(std))-((x-mean)**2)/(2*(std**2)) 
      
      def log_gaussian_logstd(self,x,mean,logstd):   

           return -0.5*np.log(2*np.pi)-logstd/2.-((x-mean)**2)/(2.*tf.exp(logstd)) 
