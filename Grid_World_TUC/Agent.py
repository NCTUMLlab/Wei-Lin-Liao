from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
import numpy as np
import keras
import pickle
import scipy.io as sio
import numpy as np



class DNN:
      def __init__(self,
                   sess,
                   net_name,
                   arch_list,
                   act_function,
                   learning_rate):

            self.sess = sess
            self.net_name = net_name
            self.arch_list = arch_list
            self.act_function = act_function
            self.learning_rate = learning_rate
            self.layer_num = len(arch_list)

            self.h_temps = []
            self.hs = []

            with tf.variable_scope(self.net_name, reuse = False):
                self.X = tf.placeholder(tf.float32, [None, self.arch_list[0]], name = "X" )
                self.Y = tf.placeholder(tf.float32, [None, self.arch_list[-1]], name = "Y" )

                for layer in range(1,self.layer_num):
                     if layer == 1:
                        h_temp = tf.layers.dense(inputs = self.X,
                                              units = self.arch_list[layer],
                                              activation = None,  
                                              kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              name = self.net_name+'_layer_'+str(layer))
                        h = self.act_function(h_temp)
                        self.h_temps.append(h_temp)
                        self.hs.append(h)

                     elif layer == self.layer_num-1:
                        h_temp = tf.layers.dense(inputs = h,
                                              units = self.arch_list[layer],
                                              activation = None,  
                                              kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              name = self.net_name+'_layer_'+str(layer))
                        h = h_temp
                        self.h_temps.append(h_temp)
                        self.hs.append(h)
                     else:
                        h_temp = tf.layers.dense(inputs = h,
                                              units = self.arch_list[layer],
                                              activation = None,  
                                              kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                              bias_initializer = tf.constant_initializer(0.1),
                                              name = self.net_name+'_layer_'+str(layer))
                        h = self.act_function(h_temp)
                        self.h_temps.append(h_temp)
                        self.hs.append(h)

            self.pred = self.hs[-1]
            self.net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.net_name)

            with tf.variable_scope(net_name + '_loss', reuse = False):
                   self.loss = tf.reduce_mean(tf.square(self.pred-self.Y))
                   self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.net_vars)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

      def train(self,X,Y):
          loss, _ = self.sess.run((self.loss,self.optimizer),feed_dict={self.X : X, self.Y : Y})
          return loss

      def dump_latent_features(self,X,layer):
          feature = self.sess.run(self.hs[layer],feed_dict={self.X : X})
          return feature

      def save_model(self, path_and_name):
          self.saver.save(self.sess, path_and_name)
           
      def load_model(self,path_and_name):
          self.saver.restore(self.sess, path_and_name)  


class PG:
      def __init__(self):
          self.render = False

          self.action_space = [0, 1, 2, 3, 4]
          self.action_size = len(self.action_space)
          self.state_size = 22
          self.discount_factor = 0.99  # decay rate
          self.learning_rate = 0.01

          self.model = self.build_model()
          self.optimizer = self.optimizer()
          self.states, self.actions, self.rewards = [], [], []

      def build_model(self):
          model = Sequential()
          model.add(Dense(15, input_dim=self.state_size, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
          model.add(Dense(10, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
          model.add(Dense(self.action_size, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
          model.summary()

          return model

      def optimizer(self):
          action = K.placeholder(shape=[None, 5])
          discounted_rewards = K.placeholder(shape=[None, ])

          # Policy Gradient 
          good_prob = K.sum(action * self.model.output, axis=1)
          eligibility = K.log(good_prob) * discounted_rewards
          loss = -K.sum(eligibility)

          optimizer = Adam(lr=self.learning_rate)
          updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
          train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

          return train

      def get_action(self, state):
          policy = self.model.predict(state, batch_size=1).flatten()
          return np.random.choice(self.action_size, 1, p=policy)[0]

      def discount_rewards(self, rewards):
          discounted_rewards = np.zeros_like(rewards)
          running_add = 0
          for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
          return discounted_rewards

      def memory(self, state, action, reward):
          self.states.append(state[0])
          self.rewards.append(reward)
          act = np.zeros(self.action_size)
          act[action] = 1
          self.actions.append(act)

      def train_episodes(self):
          discounted_rewards = np.float32(self.discount_rewards(self.rewards))
          discounted_rewards -= np.mean(discounted_rewards)
          discounted_rewards /= np.std(discounted_rewards)

          self.optimizer([self.states, self.actions, discounted_rewards])
          self.states, self.actions, self.rewards = [], [], []

      def load_model(self, name):
          self.model.load_weights(name)

      def save_model(self, name):
          self.model.save_weights(name)
          
class VIME:
      def __init__(self,
                   sess,
                   net_name,
                   state_dim,
                   action_dim,
				   batch_num,
                   learning_rate,
				   act_func = tf.nn.sigmoid,
                   dtype = 'float',
                   stddev = 0.1,
                   stddev_prior = tf.exp(-3.0)):   
                   
                   
          self.net_name = net_name 
          self.batch_num = batch_num
          self.act_func = act_func
          self.learning_rate = learning_rate
          self.stddev = stddev
          self.stddev_prior = stddev_prior

          self.states_actions = tf.placeholder("float",shape = [None, state_dim+action_dim])
          self.next_states = tf.placeholder("float",shape = [None, state_dim])
          self.batch_idx = tf.placeholder("float",shape = None)
          
          with tf.variable_scope(self.net_name, reuse = False): 
          
               # Prior of weights and biases
               self.W1_mean = tf.get_variable(self.net_name+'_W1_mean',
                                      initializer = tf.truncated_normal([state_dim+action_dim,15],stddev = self.stddev))  
               self.W1_logstd = tf.get_variable(self.net_name+'_W1_logstd',
                                      initializer = tf.truncated_normal([state_dim+action_dim,15],stddev = self.stddev)) 
                                      
               self.b1_mean = tf.get_variable(self.net_name+'_b1_mean',
                                      initializer = tf.truncated_normal([15],stddev = self.stddev))  
               self.b1_logstd = tf.get_variable(self.net_name+'_b1_logstd',
                                      initializer = tf.truncated_normal([15],stddev = self.stddev))

               self.W1_noise = tf.random_normal([state_dim+action_dim,15], mean = 0., stddev = self.stddev_prior) 
               self.b1_noise = tf.random_normal([15], mean = 0., stddev = self.stddev_prior)               
          
               self.W2_mean = tf.get_variable(self.net_name+'_W2_mean',
                                      initializer = tf.truncated_normal([15,state_dim],stddev = self.stddev))  
               self.W2_logstd = tf.get_variable(self.net_name+'_W2_logstd',
                                      initializer = tf.truncated_normal([15,state_dim],stddev = self.stddev))              

               self.b2_mean = tf.get_variable(self.net_name+'_b2_mean',
                                      initializer = tf.truncated_normal([state_dim],stddev = self.stddev))  
               self.b2_logstd = tf.get_variable(self.net_name+'_b2_logstd',
                                      initializer = tf.truncated_normal([state_dim],stddev = self.stddev))
          
               self.W2_noise = tf.random_normal([15,state_dim], mean = 0., stddev = self.stddev_prior) 
               self.b2_noise = tf.random_normal([state_dim], mean = 0., stddev = self.stddev_prior) 
          
               self.bbn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.net_name)
          
               # Weights and biases
               self.W1 = self.W1_mean + tf.multiply(tf.log(1. + tf.exp(self.W1_logstd)), self.W1_noise)   
               self.b1 = self.b1_mean + tf.multiply(tf.log(1. + tf.exp(self.b1_logstd)), self.b1_noise)   
          
               self.W2 = self.W2_mean + tf.multiply(tf.log(1. + tf.exp(self.W2_logstd)), self.W2_noise) 
               self.b2 = self.b2_mean + tf.multiply(tf.log(1. + tf.exp(self.b2_logstd)), self.b2_noise)             
          
               # Connection
               self.h1 = self.act_func(tf.add(tf.matmul(self.states_actions, self.W1), self.b1))
               self.pred = tf.add(tf.matmul(self.h1, self.W2), self.b2)
          
               # Loss function
               self.sample_log_pw = tf.reduce_sum(self.log_gaussian(self.W1,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.b1,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.W2,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.b2,0.,self.stddev_prior))
          
               self.sample_log_qw = tf.reduce_sum(self.log_gaussian_logstd(self.W1,self.W1_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.b1,self.b1_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.W2,self.W2_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.b2,self.b2_mean,self.stddev_prior*2))
          
               self.sample_log_likelihood = tf.reduce_sum(self.log_gaussian(self.next_states,self.pred,self.stddev_prior))
               self.pi = (2**(self.batch_num-self.batch_idx-1))/(2**self.batch_num-1)
          
               self.loss = tf.reduce_mean(self.pi * (self.sample_log_qw - self.sample_log_pw) - self.sample_log_likelihood)
               self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.bbn_vars)

               self.sess = sess
               self.sess.run(tf.global_variables_initializer())
               self.saver = tf.train.Saver()    
          
          
      def save_model(self, path_and_name):
          self.saver.save(self.sess, path_and_name)
           
      def load_model(self,path_and_name):
          self.saver.restore(self.sess, path_and_name)
          
      def train(self, states_actions, next_states, batch_idx):
          self.sess.run(self.optimizer, feed_dict = {self.states_actions : states_actions,#[np.newaxis, :],
                                                     self.next_states : next_states,#[np.newaxis, :],
                                                     self.batch_idx : batch_idx})      
          
      def get_hyper_parameters(self):
         
          hyper_parameters = self.sess.run([[self.W1_mean, self.W2_mean], 
                                       [self.W1_logstd,self.W2_logstd], 
                                       [self.b1_mean, self.b2_mean], 
                                       [self.b1_logstd,self.b2_logstd]])
          
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
          
               
      def KL_div(self,mean_1,log_std_1,mean_2,log_std_2):
      
           term_1 = np.mean(np.square(np.divide(log_std_1,log_std_2))) 
           term_2 = np.mean(2*log_std_2-2*log_std_1)
           term_3 = np.mean(np.divide(np.square(mean_1-mean_2),np.square(log_std_2)))
          
           return np.maximum(0,0.5*(term_1+term_2+term_2-1))
         
          
      def log_gaussian(self,x,mean,std):
          
           return -0.5*np.log(2*np.pi)-tf.log(tf.abs(std))-((x-mean)**2)/(2*(std**2)) 
      
      def log_gaussian_logstd(self,x,mean,logstd):   

           return -0.5*np.log(2*np.pi)-logstd/2.-((x-mean)**2)/(2.*tf.exp(logstd)) 



class VAE:
      def __init__(self,
                   sess,
                   net_name,
                   state_dim,
                   hidden_dim,
                   action_dim,
                   learning_rate):
          
          # Network architecture
          self.state_dim = state_dim
          self.action_dim = action_dim
         
          # Input of graph
          self.states_actions = tf.placeholder(tf.float32, [None, state_dim+action_dim], name = "states_actions" )
          self.next_states = tf.placeholder(tf.float32, [None, state_dim], name = "next_states" )

          self.net_scope = net_name + "_VAE"
          with tf.variable_scope(self.net_scope, reuse = False):

                 self.enc_hidden_1 = tf.layers.dense(inputs = self.states_actions,
                                                                    units = hidden_dim,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.net_scope+'_enc_h1')  


                 self.total_z_mean = tf.layers.dense(inputs = self.enc_hidden_1,
                                                                units = hidden_dim,
                                                                activation = None,  
                                                                kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                bias_initializer = tf.constant_initializer(0.1),
                                                                name = self.net_scope+'_mean')  

                 self.total_z_std = tf.layers.dense(inputs = self.enc_hidden_1,
                                                             units = hidden_dim,
                                                             activation = tf.nn.sigmoid,
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.net_scope+'_std') 
                    

                     
                 self.KL_div_loss = -0.5*tf.reduce_mean(1 + self.total_z_std - tf.square(self.total_z_mean) - tf.exp(self.total_z_std))
                 self.noise = tf.random_normal(tf.shape(self.total_z_std), mean = 0., stddev = 1.) 

                 self.vae_total_z = self.total_z_mean + tf.multiply(self.total_z_std, self.noise)   

                 self.dec_hidden_1 = tf.layers.dense(inputs = self.vae_total_z,
                                                                    units = hidden_dim,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.net_scope+'_dec_h1')      

                 self.dec_pred = tf.layers.dense(inputs = self.dec_hidden_1,
                                                             units = state_dim,
                                                             activation = None,  
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.net_scope+'_pred')      

                 self.vae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.net_scope)
                 self.vae_loss = tf.reduce_mean(tf.square(self.dec_pred-self.next_states)) + tf.reduce_mean(self.KL_div_loss)
                 self.vae_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.vae_loss, var_list = self.vae_vars)
               
         
          self.sess = sess
          self.sess.run(tf.global_variables_initializer())
          self.saver = tf.train.Saver()
          
      def save_model(self, path_and_name):

          self.saver.save(self.sess, path_and_name)
           
      def load_model(self,path_and_name):

          self.saver.restore(self.sess, path_and_name)
          
      def train(self, X, Y):       
  
          vae_loss, _= self.sess.run((self.vae_loss, self.vae_optimizer), feed_dict = {self.states_actions : X, self.next_states : Y})  

          return  vae_loss 

      def dump_z(self, X):

          total_z = self.sess.run((self.vae_total_z), feed_dict = {self.states_actions : X})  

          return total_z


class VAE:
      def __init__(self,
                   sess,
                   net_name,
                   state_dim,
                   action_dim,
                   learning_rate):
          
          # Network architecture
          self.state_dim = state_dim
          self.action_dim = action_dim
         
          # Input of graph
          self.states_actions = tf.placeholder(tf.float32, [None, state_dim+action_dim], name = "states_actions" )
          self.next_states = tf.placeholder(tf.float32, [None, state_dim], name = "next_states" )

          self.net_scope = net_name + "_VAE"
          with tf.variable_scope(self.net_scope, reuse = False):

                 self.enc_hidden_1 = tf.layers.dense(inputs = self.states_actions,
                                                                    units = 15,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.net_scope+'_enc_h1')  

                 '''
                 self.enc_hidden_2 = tf.layers.dense(inputs = self.enc_hidden_1,
                                                                    units =15,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.net_scope+'_enc_h2')  
                 '''

                 self.total_z_mean = tf.layers.dense(inputs = self.enc_hidden_1,
                                                                units = 5,
                                                                activation = None,  
                                                                kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                bias_initializer = tf.constant_initializer(0.1),
                                                                name = self.net_scope+'_mean')  

                 self.total_z_std = tf.layers.dense(inputs = self.enc_hidden_1,
                                                             units = 5,
                                                             activation = tf.nn.sigmoid,
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.net_scope+'_std') 
                    

                     
                 self.KL_div_loss = -0.5*tf.reduce_mean(1 + self.total_z_std - tf.square(self.total_z_mean) - tf.exp(self.total_z_std))
                 self.noise = tf.random_normal(tf.shape(self.total_z_std), mean = 0., stddev = 1.) 

                 self.vae_total_z = self.total_z_mean + tf.multiply(self.total_z_std, self.noise)   

                 self.dec_hidden_1 = tf.layers.dense(inputs = self.vae_total_z,
                                                                    units = 15,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.net_scope+'_dec_h1')      

                 self.dec_pred = tf.layers.dense(inputs = self.dec_hidden_1,
                                                             units = state_dim,
                                                             activation = None,  
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.net_scope+'_pred')      

                 self.vae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.net_scope)
                 self.vae_loss = tf.reduce_mean(tf.square(self.dec_pred-self.next_states)) + tf.reduce_mean(self.KL_div_loss)
                 self.vae_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.vae_loss, var_list = self.vae_vars)
               
         
          self.sess = sess
          self.sess.run(tf.global_variables_initializer())
          self.saver = tf.train.Saver()
          
      def save_model(self, path_and_name):

          self.saver.save(self.sess, path_and_name)
           
      def load_model(self,path_and_name):

          self.saver.restore(self.sess, path_and_name)
          
      def train(self, X, Y):       
  
          vae_loss, _= self.sess.run((self.vae_loss, self.vae_optimizer), feed_dict = {self.states_actions : X, self.next_states : Y})  

          return  vae_loss 

      def dump_z(self, X):

          total_z = self.sess.run((self.vae_total_z), feed_dict = {self.states_actions : X})  

          return total_z


            
class TUC:
      def __init__(self,
                   sess,
                   net_name,
                   state_dim,
                   action_dim,
                   learning_rate):
          
          # Network architecture
          self.state_dim = state_dim
          self.action_dim = action_dim
          
          # Input of graph
          self.states = tf.placeholder(tf.float32, [None, state_dim], name = "states" )
          self.actions = tf.placeholder(tf.float32, [None , action_dim], name = "actions" )
          self.next_states = tf.placeholder(tf.float32, [None, state_dim], name = "next_states" )
          self.critic_target_values = tf.placeholder(tf.float32, [None, 1], name = "critic_target_value")

          self.tuc_scope = net_name + "_TUC"
          self.z_dim = 5
          with tf.variable_scope(self.tuc_scope, reuse = False):

                 self.split_actions = tf.split(self.actions, num_or_size_splits = action_dim, axis=1)
                 self.actions_weight = [] 
 
                 for m in range(action_dim):
                      self.actions_weight.append(tf.tile(self.split_actions[m],[1,self.z_dim]))

                 self.vae_sub_z = []
                 self.vae_sub_z_means = []
                 self.vae_sub_z_stds = []

                 self.enc_hidden_1 = tf.layers.dense(inputs = self.states,
                                                                    units = 15,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.tuc_scope+'_enc_h1')  

                 for m in range(action_dim):
                      temp_mean = tf.layers.dense(inputs = self.enc_hidden_1,
                                                                units = self.z_dim,
                                                                activation = None,  
                                                                kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                bias_initializer = tf.constant_initializer(0.1),
                                                                name = self.tuc_scope+'_mean_'+str(m+1))  

                      temp_std = tf.layers.dense(inputs = self.enc_hidden_1,
                                                             units = self.z_dim,
                                                             activation = tf.nn.sigmoid,
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.tuc_scope+'_std_'+str(m+1)) 
                    

                      if m == 0 :
                         self.total_z_mean = tf.multiply(self.actions_weight[m],temp_mean)
                         self.total_z_std = tf.square(tf.multiply(self.actions_weight[m],temp_std))
                      else : 
                         self.total_z_mean += tf.multiply(self.actions_weight[m],temp_mean)
                         self.total_z_std += tf.square(tf.multiply(self.actions_weight[m],temp_std))

                      self.vae_sub_z_means.append(tf.multiply(self.actions_weight[m],temp_mean))
                      self.vae_sub_z_stds.append(tf.multiply(self.actions_weight[m],temp_std))
          
                 self.total_z_std = tf.sqrt(self.total_z_std)
                 self.KL_div_loss = -0.5*tf.reduce_mean(1 + self.total_z_std - tf.square(self.total_z_mean) - tf.exp(self.total_z_std))
                 self.noise = tf.random_normal(tf.shape(self.total_z_std), mean = 0., stddev = 1.) 

                 self.vae_total_z = self.total_z_mean + tf.multiply(self.total_z_std, self.noise)   

                 self.dec_hidden_1 = tf.layers.dense(inputs = self.vae_total_z,
                                                                    units = 15,
                                                                    activation = tf.nn.tanh,  
                                                                    kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                    bias_initializer = tf.constant_initializer(0.1),
                                                                    name = self.tuc_scope+'_dec_h1')      

                 self.dec_pred = tf.layers.dense(inputs = self.dec_hidden_1,
                                                             units = state_dim,
                                                             activation = None,  
                                                             kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                             bias_initializer = tf.constant_initializer(0.1),
                                                             name = self.tuc_scope+'_pred')      

                 self.tuc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.tuc_scope)
                 self.tuc_loss = tf.reduce_mean(tf.square(self.dec_pred-self.next_states)) + tf.reduce_mean(self.KL_div_loss)
                 self.tuc_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.tuc_loss, var_list = self.tuc_vars)
               
          self.critic_scope = net_name + "_CRITIC"
          with tf.variable_scope(self.critic_scope, reuse = False):
                 self.critic_hidden_1 = tf.layers.dense(inputs = self.vae_total_z, 
                                                                  units = 3,
                                                                  activation = tf.nn.tanh,  
                                                                  kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                  bias_initializer = tf.constant_initializer(0.1),
                                                                  name = self.critic_scope+'_hidden_1')      
                                                                  
                 self.critic_pred = tf.layers.dense(inputs = self.critic_hidden_1, 
                                                                  units = 1,
                                                                  activation = None,  
                                                                  kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.03),
                                                                  bias_initializer = tf.constant_initializer(0.1),
                                                                  name = self.critic_scope+'_pred')
                                                                  
                 self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.critic_scope)
                 self.critic_loss = tf.reduce_mean(tf.square(self.critic_pred-self.critic_target_values))
                 self.critic_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss, var_list = self.critic_vars)
          
          self.sess = sess
          self.sess.run(tf.global_variables_initializer())
          self.saver = tf.train.Saver()
          
      def save_model(self, path_and_name):

          self.saver.save(self.sess, path_and_name)
           
      def load_model(self,path_and_name):

          self.saver.restore(self.sess, path_and_name)
          
      def train_tuc(self, states, next_states, actions):       
  
          tuc_loss, _= self.sess.run((self.tuc_loss, self.tuc_optimizer), feed_dict = {self.states : states, self.next_states : next_states, self.actions : actions})  

          return  tuc_loss
      
      def train_critic(self, states, critic_target_values, actions): 

          critic_loss, _= self.sess.run((self.critic_loss, self.critic_optimizer), feed_dict = {self.states : states, self.critic_target_values : critic_target_values, self.actions : actions})  
          
          return  critic_loss
          
      def dump_regret(self, states, act_chosen):
      
          temp_actions = np.eye(self.action_dim)
          temp_states = np.tile(states, [self.action_dim,1])
          
          critic_values = self.sess.run(self.critic_pred, feed_dict = {self.states : temp_states, self.actions : temp_actions})
          
          regret = np.max(critic_values) - critic_values[act_chosen,0]
          
          return regret
      
      def dump_z_mean_std(self, states, actions):

          total_z_mean, total_z_std = self.sess.run((self.total_z_mean, self.total_z_std), feed_dict = {self.states : states, self.actions : actions})  

          return total_z_mean, total_z_std   

      def dump_z(self, states, actions):

          total_z = self.sess.run((self.vae_total_z), feed_dict = {self.states : states, self.actions : actions})  

          return total_z
