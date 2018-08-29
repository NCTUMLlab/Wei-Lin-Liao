import torch
import numpy as np

a = np.array([[0,0,0,0,2,0]])
A = torch.from_numpy(a).float()#.cuda()


split_A = torch.split(A, 1, dim=1)
print(split_A[0])
print(split_A[0].repeat(1, 3))
print(split_A[0]+split_A[4])
print(split_A[0]*split_A[4])
print(split_A[4]**2)
print(split_A[4]**0.5)
print(torch.mean(A))
print(A**2)
print(torch.exp(A))
print(A.size()[0])
#print(split_A)
#print(A,A.size())

#c = torch.randn(56, 80)
#d = torch.split(c, 40, dim=1)
#print(c, d)