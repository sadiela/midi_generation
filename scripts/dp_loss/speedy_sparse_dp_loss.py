import torch
import numpy as np 
from dp_loss.shared_functions import *
from torch.profiler import profile, record_function, ProfilerActivity

def construct_theta_sparse_k_loop(x, x_hat, device):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n), device=device)
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]), device=device)
    for k in range(0,m*n): # should hit all ij combos here!
        i, j = ij_from_k(k,m,n)
        print(k,i,j)
        if i > 0 and j > 0:
            if (x[:, i-1] == x_hat[:, j-1]).all():
                theta = torch.add(theta, torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k]], 0.001, (m*n, m*n), device=device))
            else:
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k]], note_diff(x[:, i-1] ,x_hat[:, j-1]), (m*n, m*n), device=device)  # replacing; cost depends on ...?
                grad_theta = grad_theta + sparse_add_gradients(k_from_ij(i-1,j-1, m,n), k, j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]), m,n, device=device)
            theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j-1, m,n)]], single_note_val(x_hat[:, j-1]), (m*n, m*n), device=device) #deletion
            grad_theta = grad_theta + sparse_add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]), m,n, device=device)
            #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
            theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i-1,j, m,n)]],  single_note_val(x[:, i-1]), (m*n, m*n), device=device) # insertion I think i want these both dependent on x_hat... is that possible? 
    return -theta, -grad_theta

def speedy_sparse_diffable_recursion(theta, device, gamma=0.3): # passed in sparse
    N = theta.size()[0] # 
    e_bar = torch.zeros(N, device=device)
    e_bar[N-1]=1
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

def forward_loop(i, X, X_hat, device, grad_L_x):
    theta, grad_theta_xhat = construct_theta_sparse_k_loop(X[i][0], X_hat[i][0], device)
    print("theta constructed")
    theta = theta.coalesce()
    grad_theta_xhat = grad_theta_xhat.coalesce()
    loss, grad_L_theta = speedy_sparse_diffable_recursion(theta, device)
    grad_L_theta.coalesce()
    # Just loop through sparse index pairs!
    grad_L_theta = grad_L_theta.coalesce()
    nonzero_indices = grad_L_theta.indices()
    nonzero_vals = grad_L_theta.values()
    for idx in range(nonzero_indices.size()[1]):
        j = nonzero_indices[0,idx] 
        k = nonzero_indices[1,idx] 
        grad_L_theta_val = nonzero_vals[idx]
        if has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
            cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k, device)# scalar times pxn
            grad_L_x[i][0] += cur_grad
    return loss, grad_L_x

class SpeedySparseDynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    full_loss = 0
    grad_L_x = torch.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) # THIS SIZE FINE
    grad_L_x.to(device)
    print(X_hat.shape[0])
    for i in range(X_hat.shape[0]):
        print(i)
        if i == 0: 
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
            ) as prof:
                loss, grad_L_x = forward_loop(i, X, X_hat, device, grad_L_x)
                full_loss += loss
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            loss, grad_L_x = forward_loop(i, X, X_hat, device, grad_L_x)
            full_loss += loss

    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss

  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None
    
class SpeedySparseDynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    theta, grad_theta_xhat = construct_theta_sparse_k_loop(X, X_hat, device)
    theta = theta.coalesce()
    grad_theta_xhat = grad_theta_xhat.coalesce()
    #print("GRADTHETA:", torch.count_nonzero(grad_theta_xhat.to_dense()))
    loss, grad_L_theta = speedy_sparse_diffable_recursion(theta, device)
    grad_L_theta.coalesce()
    #print("GRADLTHETA:", torch.count_nonzero(grad_L_theta.to_dense()))
    n_2 = grad_L_theta.size()[0]
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    grad_L_theta = grad_L_theta.coalesce()
    nonzero_indices = grad_L_theta.indices()
    nonzero_vals = grad_L_theta.values()
    for idx in range(nonzero_indices.size()[1]):
        j = nonzero_indices[0,idx] 
        k = nonzero_indices[1,idx] 
        grad_L_theta_val = nonzero_vals[idx]
        if has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
            cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k, device)# scalar times pxn
            grad_L_x += cur_grad
    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss, grad_L_x, theta, grad_L_theta
    
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None