import torch
import numpy as np 
from shared_functions import *

'''
This original DP loss codes suffers from both time and memory complexity issues 
to the point that it is infeasible for the MIDI reconstruction problem
'''

def construct_theta(x, x_hat, zero = 0):
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.zeros((m*n, m*n))
    grad_theta = torch.zeros((m*n, m*n, x_hat.shape[0], x_hat.shape[1]))
    theta[:,:] = np.Inf
    try:
        for i in range(1,m):
            for j in range(1,n):
                if (x[:, i-1] == x_hat[:, j-1]).all():
                    theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = zero
                else:
                    theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                    grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)][:,j-1] = distance_derivative(x[:,i-1]-x_hat[:,j-1]) # FIX ZEROS
                theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
                grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
                theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
        return -theta, -grad_theta
    except RuntimeError as err:
        print(err, x.shape, x_hat.shape)
        return torch.zeros((m*n, m*n)), torch.zeros((m*n, m*n, x_hat.shape[0], x_hat.shape[1]))

def diffable_recursion(theta, gamma=0.3):
    ''' Initializations '''
    N = theta.shape[0] 
    e_bar = torch.zeros(N)
    e_bar[N-1]=1
    v = torch.zeros(N)
    q = torch.zeros((N,N))
    E = torch.zeros((N,N))
    for j in range(2, N): # looping through and looking at PARENTS of j
        parent_indices = torch.where(theta[:,j]>np.NINF)[0]
        u = torch.tensor(np.asarray([theta[i,j] + v[i] for i in parent_indices]))
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma)))
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma))
        for k, i in enumerate(parent_indices):
            q[i,j] = q_vals[k]
    for i in range(N-1,0, -1): # looping through and looking at CHILDREN of i
        children_indices = torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            E[i,j] = q[i,j]*e_bar[j]
            e_bar[i] += E[i,j]

    return -v[N-1], -E # loss, gradient w.r.t. theta

'''
Works with batched MIDIs
'''
class DynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X):
    grad_L_x = torch.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    for i in range(X_hat.shape[0]):
        theta, grad_theta_xhat = construct_theta(X[i][0], X_hat[i][0])
        loss, grad_L_theta = diffable_recursion(theta)
        n_2 = grad_L_theta.shape[0]
        for j in range(n_2):
            for k in range(n_2):
                if torch.abs(grad_L_theta[j][k]) != 0 and torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                    cur_grad = grad_L_theta[j][k] * grad_theta_xhat[j][k]
                    grad_L_x = torch.add(grad_L_x[i][0], cur_grad)
    
    ctx.save_for_backward(grad_L_x)
    return loss
  
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None

'''
For testing with 2d tensors
'''
class DynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X):
    theta, grad_theta_xhat = construct_theta(X, X_hat)
    loss, grad_L_theta = diffable_recursion(theta)
    n_2 = grad_L_theta.shape[0]
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    for j in range(n_2):
        for k in range(n_2):
            if torch.abs(grad_L_theta[j][k]) != 0 and torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                cur_grad = grad_L_theta[j][k] * grad_theta_xhat[j][k]
                grad_L_x = torch.add(grad_L_x, cur_grad)

    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    return loss, grad_L_x, theta, grad_L_theta

  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None