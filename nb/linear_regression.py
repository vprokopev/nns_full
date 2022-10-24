import numpy as np
import torch
import torch.nn as nn

X = [[1, 1, 4],
     [2,  1, 1],
     [1, 1, 7],
     [4, 2, 2],
     [5, 2, 6],
     [3, 1, 1],
     [6, 3, 8],
     [1, 1, 9],
     [3, 1, 2]]
y = [3.4, 6.1, 1.9, 8.3, 7.4, 7.1, 9.2, 1.4,6.8]
X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
y = torch.tensor(y, requires_grad=False, dtype=torch.float32)
w = torch.zeros(size=(3, ), requires_grad=True, dtype=torch.float32)
b = torch.zeros((1, ), requires_grad=True, dtype=torch.float32)

# get gradient vector
errors = torch.matmul(X, w) + b - y
mse = torch.dot(errors, errors)/X.size()[1]
gradients = 2*(torch.matmul(X.T, errors)/X.size()[1])

mse.backward()

w.grad, b.grad

gradients

