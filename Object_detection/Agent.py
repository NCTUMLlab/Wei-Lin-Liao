import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import numpy as np

#from sklearn.datasets import fetch_mldata
#from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing

class PG:

      def __init__(self, learning_rate, gamma):

            self.model = nn.Sequential(
                nn.Linear(25112, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 6),
                nn.Softmax(),
            )
          
            self.gamma = gamma
            self.learning_rate = learning_rate

            self.model = self.model.cuda()
            self.model.apply(self.weights_init)

            self.states = []
            self.actions = []
            self.rewards = []
            
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
      
      def weights_init(self,model):
            classname = model.__class__.__name__
            if classname.find('Linear') != -1:
               model.weight.data.normal_(0.0, 0.02)
               model.bias.data.fill_(0.1)

      def select_action(self, state):

            policy_prob = self.model(Variable(state))
            prob = policy_prob.data.cpu().numpy()
            action = np.random.choice(range(prob.shape[1]), p=prob.ravel()) 

            return action

      def get_policy_prob(self,state):
 
            policy_prob = self.model(Variable(state))
            prob = policy_prob.data.cpu().numpy()

            return prob

      def store_transition(self, state, action, reward):

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(float(reward))

      def get_values(self):
       
            discounted_values = np.zeros_like(self.rewards)
            running_add = 0.
            for t in reversed(range(0, len(self.rewards))):
                 running_add = running_add * self.gamma + self.rewards[t]
                 discounted_values[t] = running_add

            return discounted_values.tolist()
 
      
 
 
      def save_model(self,name):
            torch.save(self.model, name+'.pkl')
            print("model of agent saved !!")
   
      def load_model(self,name):
            self.model = torch.load(name+'.pkl')
            print("model of agent loaded !!")


      def REINFORCE(self):

            T = len(self.states)
            values = self.get_values()

            self.optimizer.zero_grad()

            loss = 0
            for t in range(T) :
                 pg_state = self.states[t]
                 pg_action = Variable(torch.from_numpy(np.array(self.actions[t])).float().cuda())
                 pg_value= Variable(torch.from_numpy(np.array(values[t])).float().cuda())
                 #policy_prob = self.select_action(pg_state)
                 #print(type(policy_prob))
                 policy_prob = self.model(Variable(pg_state))
                 cate = Categorical(policy_prob)
                 loss = - cate.log_prob(pg_action)*pg_value
                 loss.backward() 

            self.optimizer.step() 
            self.states = []
            self.actions = []
            self.rewards = []

class BBN_layer(nn.Module): 

      def __init__(self, n_input, n_output, sigma_prior):
            super(BBN_layer, self).__init__()
            self.n_input = n_input
            self.n_output = n_output
            self.sigma_prior = sigma_prior
            self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.1))
            self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.1))
            self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, 0.1))
            self.b_logsigma = nn.Parameter(torch.Tensor(n_output).normal_(0, 0.1))
            self.lpw = 0
            self.lqw = 0
            self.layer_parameters = [self.W_mu, self.W_logsigma, self.b_mu, self.b_logsigma] 



      def forward(self, X, infer=False):
            if infer:
               output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
               return output

            epsilon_W, epsilon_b = self.get_random()
            W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
            b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
            output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
            self.lpw = self.log_gaussian(W, 0, self.sigma_prior).sum() + self.log_gaussian(b, 0, self.sigma_prior).sum()
            self.lqw = self.log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + self.log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
            return output

      def get_random(self):
            return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.sigma_prior).cuda()), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior).cuda())


      def log_gaussian(self, x, mu, sigma):
            return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


      def log_gaussian_logsigma(self, x, mu, logsigma):
            return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class BBN_module(nn.Module):
      def __init__(self):
            super(BBN_module, self).__init__()
            self.sigma_prior = float(np.exp(-3))
            self.layer_1 = BBN_layer(25112+6, 256, self.sigma_prior)
            self.layer_1_relu = nn.ReLU()
            self.layer_2 = BBN_layer(256, 128, self.sigma_prior)
            self.layer_2_relu = nn.ReLU()
            self.layer_3 = BBN_layer(128, 256, self.sigma_prior)
            self.layer_3_relu = nn.ReLU()
            self.layer_4 = BBN_layer(256, 25112, self.sigma_prior)

            self.module_parameters = [self.layer_1.layer_parameters, self.layer_2.layer_parameters, self.layer_3.layer_parameters, self.layer_4.layer_parameters]

      def forward(self, X, infer=False):
            output = self.layer_1_relu(self.layer_1(X, infer))
            output = self.layer_2_relu(self.layer_2(output, infer))
            output = self.layer_3_relu(self.layer_3(output, infer))
            output = self.layer_4(output, infer)
            return output
 
      def get_lpw_lqw(self):
            lpw = self.layer_1.lpw + self.layer_2.lpw + self.layer_3.lpw + self.layer_4.lpw
            lqw = self.layer_1.lqw + self.layer_2.lqw + self.layer_3.lqw + self.layer_4.lqw
            return lpw, lqw

class BBN(BBN_module):

      def __init__(self, BBN_module, learning_rate):
           super(BBN, self).__init__()
           self.net = BBN_module
           self.net = self.net.cuda()
           self.sigma_prior = float(np.exp(-3))
           self.learning_rate = learning_rate 
           self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

      def log_gaussian(self, x, mu, sigma):
            return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


      def log_gaussian_logsigma(self, x, mu, logsigma):
            return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)
           
           
      def forward_pass_samples(self, X, y):
            s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
            output = self.net(X)
            sample_log_pw, sample_log_qw = self.net.get_lpw_lqw()
            sample_log_likelihood = self.log_gaussian(y, output, self.sigma_prior).sum()
            
            return sample_log_pw, sample_log_qw, sample_log_likelihood

      def criterion(self, l_pw, l_qw, l_likelihood):

            return ((l_qw - l_pw) - l_likelihood).sum()

      def train(self, X, y):
            log_pw, log_qw, log_likelihood = self.forward_pass_samples(X, y)
            loss = self.criterion(log_pw, log_qw, log_likelihood)
            loss.backward()
            self.optimizer.step()
            
      def save_model(self,name):
            torch.save(self.net, name+'.pkl')
            print("model of agent saved !!")
   
      def load_model(self,name):
            self.net = torch.load(name+'.pkl')
            print("model of agent loaded !!")
            

      def dump_hyparameters(self):
            all_W_means = []
            all_W_logsigmas = []
            all_b_means = []
            all_b_logsigmas = []
            for layer in range(4):
                 all_W_means.append(self.net.module_parameters[layer][0].data.cpu().numpy())
                 all_W_logsigmas.append(self.net.module_parameters[layer][1].data.cpu().numpy())
                 all_b_means.append(self.net.module_parameters[layer][2].data.cpu().numpy())
                 all_b_logsigmas.append(self.net.module_parameters[layer][3].data.cpu().numpy())
            
            return [all_W_means, all_W_logsigmas, all_b_means, all_b_logsigmas]

      def get_info_gain(self, hyparameters, pre_hyparameters):
            length = len(hyparameters[0])
            total_div = 0.
            for layer in range(length):          
                 W_div = self.KL_div(hyparameters[0][layer],hyparameters[1][layer],pre_hyparameters[0][layer],pre_hyparameters[1][layer])
                 b_div = self.KL_div(hyparameters[2][layer],hyparameters[3][layer],pre_hyparameters[2][layer],pre_hyparameters[3][layer])        
                 total_div += (W_div+b_div)

            return total_div

      def KL_div(self,mean_1,log_std_1,mean_2,log_std_2):
      
            term_1 = np.mean(np.square(np.divide(log_std_1,log_std_2+1e-5))) 
            term_2 = np.mean(2*log_std_2-2*log_std_1)
            term_3 = np.mean(np.divide(np.square(mean_1-mean_2),np.square(log_std_2)+1e-5))
            
            return np.maximum(0,0.5*(term_1+term_2+term_2-1))


class TUC:

      def __init__(self, learning_rate):

            self.build_network()
            self.learning_rate = learning_rate
            self.enc_dec_optimizer = torch.optim.Adam(self.enc_dec_parameters, lr=self.learning_rate)  
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters , lr=self.learning_rate)     

      def save_model(self,name):
            torch.save(self.all_parameters, name+'.pkl')
            print("model of TUC saved !!")
   
      def load_model(self,name):
            self.all_parameters = torch.load(name+'.pkl')
            print("model of TUC loaded !!")

      def dump_z_mean_std(self, state, action_vec):       
      
          self.action_vec_split = torch.split(Variable(action_vec), 1, dim=1)
          
          # forward
          self.h1 = self.enc_share(Variable(state))
          self.total_mean = self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64))\
                          + self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64))\
                          + self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64))\
                          + self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64))\
                          + self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64))\
                          + self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64))
          
          self.total_std = ((self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64)))**2\
                          + (self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64)))**2\
                          + (self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64)))**2\
                          + (self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64)))**2\
                          + (self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64)))**2\
                          + (self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64)))**2)**0.5
                          
          total_mean = self.total_mean.data.cpu().numpy()
          total_std = self.total_std.data.cpu().numpy()
          
          return total_mean, total_std

      def dump_exploration_reward(self,  pre_mean, pre_std, mean, std):
           
           return self.KL_divergence(pre_mean,pre_std,mean,std)      
 

      def dump_regret(self, state, action_chosen):
      
          n = state.size()[0]
          self.action_vec_split = torch.split(Variable(torch.eye(6)), 1, dim=1)       

          # forward
          self.h1 = self.enc_share(Variable(state.repeat(6,1)))
          self.total_mean = self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64).cuda())\
                          + self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64).cuda())\
                          + self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64).cuda())\
                          + self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64).cuda())\
                          + self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64).cuda())\
                          + self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64).cuda())
          
          self.total_std = ((self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64).cuda()))**2\
                          + (self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64).cuda()))**2\
                          + (self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64).cuda()))**2\
                          + (self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64).cuda()))**2\
                          + (self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64).cuda()))**2\
                          + (self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64).cuda()))**2)**0.5
          
          self.noise = Variable(torch.Tensor(n, 64).normal_(0, 1).cuda())
          self.z = self.total_mean + self.total_std*self.noise          
          self.critic_pred = self.critic(self.z).data.cpu().numpy()
                   

          regret =  self.critic_pred.max() - self.critic_pred[action_chosen,0]

          return regret         

      def KL_divergence(self, mean_1, log_std_1, mean_2, log_std_2):    

          term_1 = np.sum(np.square(np.divide(np.exp(log_std_1),np.exp(log_std_2)))) 
          term_2 = np.sum(2*log_std_2-2*log_std_1)
          term_3 = np.sum(np.divide(np.square(mean_1-mean_2),np.square(np.exp(log_std_2))))
          
          return np.maximum(0,0.5*(term_1+term_2+term_2-1))    
          
            
      def train_enc_dec(self, state, next_state, action_vec):
 
          self.action_vec_split = torch.split(Variable(action_vec), 1, dim=1)
          
          # forward
          self.h1 = self.enc_share(Variable(state))
          self.total_mean = self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64))\
                          + self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64))\
                          + self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64))\
                          + self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64))\
                          + self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64))\
                          + self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64))
          
          self.total_std = ((self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64)))**2\
                          + (self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64)))**2\
                          + (self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64)))**2\
                          + (self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64)))**2\
                          + (self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64)))**2\
                          + (self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64)))**2)**0.5
          
          self.noise = Variable(torch.Tensor(1, 64).normal_(0, 1).cuda())
          self.z = self.total_mean + self.total_std*self.noise
          self.dec_pred = self.dec(self.z)             

          self.enc_dec_loss = torch.mean(self.dec_pred-Variable(next_state)) + -0.5*torch.mean(1 + self.total_std - self.total_mean**2 - torch.exp(self.total_std))
          self.enc_dec_loss.backward()    
          self.enc_dec_optimizer.step()

      def train_critic(self, state, value, action_vec):  
      
          self.action_vec_split = torch.split(Variable(action_vec), 1, dim=1)
          
          # forward
          self.h1 = self.enc_share(Variable(state))
          self.total_mean = self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64))\
                          + self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64))\
                          + self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64))\
                          + self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64))\
                          + self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64))\
                          + self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64))
          
          self.total_std = ((self.enc_mean_0(self.h1)*(self.action_vec_split[0].repeat(1,64)))**2\
                          + (self.enc_mean_1(self.h1)*(self.action_vec_split[1].repeat(1,64)))**2\
                          + (self.enc_mean_2(self.h1)*(self.action_vec_split[2].repeat(1,64)))**2\
                          + (self.enc_mean_3(self.h1)*(self.action_vec_split[3].repeat(1,64)))**2\
                          + (self.enc_mean_4(self.h1)*(self.action_vec_split[4].repeat(1,64)))**2\
                          + (self.enc_mean_5(self.h1)*(self.action_vec_split[5].repeat(1,64)))**2)**0.5
          
          self.noise = Variable(torch.Tensor(1, 64).normal_(0, 1).cuda())
          self.z = self.total_mean + self.total_std*self.noise
          self.critic_pred = self.critic(self.z)    
          
          self.critic_loss = torch.mean(self.critic_pred-Variable(value)) 
          self.critic_loss.backward()    
          self.critic_optimizer.step()          
           
      def build_network(self):

            self.enc_share = nn.Sequential(
                nn.Linear(25112, 256),
                nn.Sigmoid(),
            )
 
            self.enc_mean_0 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )
            
            self.enc_std_0 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )
    
            self.enc_mean_1 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )
            
            self.enc_std_1 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )

            self.enc_mean_2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

            self.enc_std_2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )

            self.enc_mean_3 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

            self.enc_std_3 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )

            self.enc_mean_4 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

            self.enc_std_4 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )

            self.enc_mean_5 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

            self.enc_std_5 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
            )

            self.dec = nn.Sequential(
                nn.Linear(64, 128),
                nn.Sigmoid(),
                nn.Linear(128, 256),
                nn.Sigmoid(),
                nn.Linear(256, 25112),
            )

            self.critic = nn.Sequential(
                nn.Linear(64, 32),
                nn.Sigmoid(),
                nn.Linear(32, 1),
            )
            
            self.enc_share = self.enc_share.cuda()
            self.enc_mean_0 = self.enc_mean_0.cuda()
            self.enc_mean_1 = self.enc_mean_1.cuda()
            self.enc_mean_2 = self.enc_mean_2.cuda()
            self.enc_mean_3 = self.enc_mean_3.cuda()
            self.enc_mean_4 = self.enc_mean_4.cuda()
            self.enc_mean_5 = self.enc_mean_5.cuda()
            self.enc_std_0 = self.enc_std_0.cuda()
            self.enc_std_1 = self.enc_std_1.cuda()
            self.enc_std_2 = self.enc_std_2.cuda()
            self.enc_std_3 = self.enc_std_3.cuda()
            self.enc_std_4 = self.enc_std_4.cuda()
            self.enc_std_5 = self.enc_std_5.cuda()
            self.dec = self.dec.cuda()
            self.critic = self.critic.cuda()
            
            self.enc_dec_parameters = list(self.enc_share.parameters())\
                                + list(self.enc_mean_0.parameters())\
                                + list(self.enc_mean_1.parameters())\
                                + list(self.enc_mean_2.parameters())\
                                + list(self.enc_mean_3.parameters())\
                                + list(self.enc_mean_4.parameters())\
                                + list(self.enc_mean_5.parameters())\
                                + list(self.enc_std_0.parameters())\
                                + list(self.enc_std_1.parameters())\
                                + list(self.enc_std_2.parameters())\
                                + list(self.enc_std_3.parameters())\
                                + list(self.enc_std_4.parameters())\
                                + list(self.enc_std_5.parameters())\
                                + list(self.dec.parameters())
           
            self.critic_parameters = self.critic.parameters()

            self.all_parameters = list(self.enc_share.parameters())\
                                + list(self.enc_mean_0.parameters())\
                                + list(self.enc_mean_1.parameters())\
                                + list(self.enc_mean_2.parameters())\
                                + list(self.enc_mean_3.parameters())\
                                + list(self.enc_mean_4.parameters())\
                                + list(self.enc_mean_5.parameters())\
                                + list(self.enc_std_0.parameters())\
                                + list(self.enc_std_1.parameters())\
                                + list(self.enc_std_2.parameters())\
                                + list(self.enc_std_3.parameters())\
                                + list(self.enc_std_4.parameters())\
                                + list(self.enc_std_5.parameters())\
                                + list(self.dec.parameters())\
                                + list(self.critic.parameters())