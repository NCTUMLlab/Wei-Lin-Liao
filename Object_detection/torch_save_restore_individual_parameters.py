import torch
import torch.nn as nn
import torch.nn.functional as F

NN = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

BB = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
            )

all_parameters = list(NN.parameters())+list(BB.parameters())

torch.save(all_parameters, 'test.pkl')

all_parameters =  torch.load('test.pkl')