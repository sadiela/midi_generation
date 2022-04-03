import torch
import numpy as np 
from shared_functions import *

def construct_theta(x, x_hat, zero = 0):
    #print("CONSTRUCTING THETA:", x.shape, x_hat.shape)
    # can I construct gradient of theta alongside this? 
    # for each theta (k,l), will have gradient w.r.t. x_hat
    #   gradient will be 0 for all except x_hat[:,j]
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.zeros((m*n, m*n))
    grad_theta = torch.zeros((m*n, m*n, x_hat.shape[0], x_hat.shape[1]))
    #print(x.shape, x_hat.shape, grad_theta.shape)
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


def diffable_recursion(theta, gamma=0.1):
    N = theta.shape[0] 
    e_bar = torch.zeros(N)
    e_bar[N-1]=1
    v = torch.zeros(N)
    q = torch.zeros((N,N))
    E = torch.zeros((N,N))
    for j in range(2, N): # looping through and looking at PARENTS of j
        parent_indices = torch.where(theta[:,j]>np.NINF)[0]
        #print("Parents:", parent_indices)
        u = torch.tensor(np.asarray([theta[i,j] + v[i] for i in parent_indices]))
        #print(i, u)
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma)))
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma))
        #print(u, q_vals)
        for k, i in enumerate(parent_indices):
            q[i,j] = q_vals[k]
    for i in range(N-1,0, -1): # looping through and looking at CHILDREN of i
        children_indices = torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            E[i,j] = q[i,j]*e_bar[j]
            e_bar[i] += E[i,j]

    return -v[N-1], -E

class DynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X):
    # build theta from original data and reconstruction
    theta, grad_theta_xhat = construct_theta(X, X_hat)
    #print(grad_theta_xhat[0][1])
    loss, grad_L_theta = diffable_recursion(theta)
    #loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
    #print(grad_L_theta)
    n_2 = grad_L_theta.shape[0]
    #print(n_2)
    #print("DL_DTheta:", torch.count_nonzero(grad_L_theta), grad_L_theta)
    #print("DTheta_Dx:", torch.count_nonzero(grad_theta_xhat)) #, grad_L_theta)
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    for i in range(n_2):
      for j in range(n_2):
        if torch.abs(grad_L_theta[i][j]) != 0 and torch.count_nonzero(grad_theta_xhat[i][j]) != 0:
          #print(grad_L_theta[i][j] * grad_theta_xhat[i][j])
          #  print(grad_theta_xhat[i][j])
          #if torch.count_nonzero(grad_theta_xhat[i][j]) != 0:
          #  print('DX IJ NON ZERO', i,j, grad_theta_xhat[i][j])
          cur_grad = grad_L_theta[i][j] * grad_theta_xhat[i][j]
          grad_L_x = torch.add(grad_L_x, cur_grad)
    #grad =torch.einsum('ij,ijkl->kl', grad.double(), grad_theta.double())
    print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss
  
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None

class DynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    theta, grad_theta_xhat = construct_theta(X, X_hat)
    #print("THETA:", theta)
    loss, grad_L_theta = diffable_recursion(theta)
    #loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
    #print("LOSSES:", loss, -loss_exact)
    #print(grad_L_theta)
    n_2 = grad_L_theta.shape[0]
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    for j in range(n_2):
        for k in range(n_2):
            if torch.abs(grad_L_theta[j][k]) != 0 and torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                cur_grad = grad_L_theta[j][k] * grad_theta_xhat[j][k]
                grad_L_x = torch.add(grad_L_x, cur_grad)

    print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss