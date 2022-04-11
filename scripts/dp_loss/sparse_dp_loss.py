import torch
import numpy as np 
#from dp_loss import *
import torch.nn.functional as F
from shared_functions import *


def construct_theta_sparse(x, x_hat, device):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n), device=device)
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]), device=device)
    for i in range(1,m):
            for j in range(1,n):
                #print(i,j)
                if (x[:, i-1] == x_hat[:, j-1]).all():
                    theta = torch.add(theta, torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], 0.001, (m*n, m*n), device=device)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0
                else:
                    theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], note_diff(x[:, i-1] ,x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                    grad_theta = grad_theta + sparse_add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j, m,n), j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]), m,n, device=device)
                    #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)][:,j-1] = distance_derivative(x[:,i-1]-x_hat[:,j-1]) # FIX ZEROS
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j-1, m,n)]], single_note_val(x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
                grad_theta = grad_theta + sparse_add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]), m,n, device=device)
                #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i-1,j, m,n)]],  single_note_val(x[:, i-1]), (m*n, m*n), device=device)#theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
    return -theta, -grad_theta

def sparse_diffable_recursion(theta, device, gamma=0.3): # passed in sparse
    N = theta.size()[0] # 
    e_bar = torch.zeros(N, device=device)
    e_bar[N-1] = 1
    v = torch.zeros(N, device=device)
    q = torch.sparse_coo_tensor((N,N), device=device) #torch.zeros((N,N)) # SPARSIFY
    E = torch.sparse_coo_tensor((N,N), device=device) #torch.zeros((N,N)) # SPARSIFY
    for j in range(2, N): # looping through and looking at PARENTS of j
        parent_indices = get_parent_indices(theta, j) # torch.where(theta[:,j]>np.NINF)[0] # CHANGE
        u = torch.tensor(np.asarray([(i[1] + v[i[0]]) for i in parent_indices], dtype=np.float32)) # CHANGE
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma))) # this is fine
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma)) # this is fine
        q = q + q_additions(parent_indices, q_vals, j, N, device)
    for i in range(N-1,0, -1): # looping through and looking at CHILDREN of i
        children_indices = get_child_indices(theta, i) #torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            q_ij = get_ijth_val(q, i,j[0]).to(device) # value at ij
            E += E_val(i, j[0], q_ij*e_bar[j[0]], N, device)
            e_bar[i] += get_ijth_val(E, i, j[0]).to(device) # do i have to do this for an int? 
    return -v[N-1], -E

class SparseDynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    grad_L_x = torch.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) # THIS SIZE FINE
    grad_L_x.to(device)
    print(X_hat.shape[0])
    for i in range(X_hat.shape[0]):
        print(i)
        theta, grad_theta_xhat = construct_theta_sparse(X[i][0], X_hat[i][0], device)
        theta = theta.coalesce()
        grad_theta_xhat = grad_theta_xhat.coalesce()
        loss, grad_L_theta = sparse_diffable_recursion(theta, device)
        grad_L_theta.coalesce()
        n_2 = grad_L_theta.size()[0]
        for j in range(n_2):
            for k in range(n_2): ##### NOT DONE W THIS PART!!!! ####
                grad_L_theta_val = get_ijth_val(grad_L_theta, j,k)
                if grad_L_theta_val != 0 and has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                    cur_grad = grad_L_theta_val * get_slice(grad_theta_xhat, j,k) # scalar times pxn
                    grad_L_x[i][0] += cur_grad
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss
    
class SparseDynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    theta, grad_theta_xhat = construct_theta_sparse(X, X_hat, device)
    theta = theta.coalesce()
    grad_theta_xhat = grad_theta_xhat.coalesce()
    loss, grad_L_theta = sparse_diffable_recursion(theta, device)
    grad_L_theta.coalesce()
    n_2 = grad_L_theta.size()[0]
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    for j in range(n_2):
        for k in range(n_2): ##### NOT DONE W THIS PART!!!! ####
            grad_L_theta_val = get_ijth_val(grad_L_theta, j,k)
            if grad_L_theta_val != 0 and has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k) # scalar times pxn
                grad_L_x += cur_grad
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss, grad_L_x, theta, grad_L_theta
  
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None
