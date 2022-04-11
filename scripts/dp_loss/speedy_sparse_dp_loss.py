from pickle import EXT1
import torch
import numpy as np 
from dp_loss.shared_functions import *
from torch.profiler import profile, record_function, ProfilerActivity

'''
i = [[0, 1, 1],
         [2, 0, 2]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
Upper bound, make too long, then cut
Loose upper bound = 3n^2
''' 
# Try with/without pre-allocating the lists (stack overflow suggests it may not matter)
# getindices(k_from_ij(i-1,j-1, m,n), k, j-1, distance_derivative(-x_hat[:,j-1]), m,n)
def getindices(idx1, idx2, idx4, deriv):
    index0 = []
    index1 = []
    index2 = []
    index3 = []
    v = []

    for i, val in enumerate(deriv):
        if val !=0:
            v.append(val)
            index0.append(idx1)
            index1.append(idx2)
            index2.append(i) # y direction
            index3.append(idx4)
    return index0, index1, index2, index3, v

def construct_theta_sparse_k_loop(x, x_hat, device):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta_indices= [[],
        []]
    theta_v = []
    grad_theta_indices = [[],
        [],
        [],
        []]
    grad_theta_v = []
    for k in range(0,m*n): # should hit all ij combos here!
        i, j = ij_from_k(k,m,n)
        if i > 0 and j > 0:
            if (x[:, i-1] == x_hat[:, j-1]).all():
                theta_v.append(0.001)
                theta_indices[0].append(k_from_ij(i-1,j-1, m,n))
                theta_indices[1].append(k)
            else:
                theta_v.append(note_diff(x[:, i-1] ,x_hat[:, j-1]))
                theta_indices[0].append(k_from_ij(i-1,j-1, m,n))
                theta_indices[1].append(k)
                i0, i1, i2, i3, v = getindices(k_from_ij(i-1,j-1, m,n), k, j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]))
                grad_theta_v += v #.append(note_diff(x[:, i-1] ,x_hat[:, j-1]))
                grad_theta_indices[0] += i0 #.append(k_from_ij(i-1,j-1, m,n))
                grad_theta_indices[1] += i1 #.append(k)
                grad_theta_indices[2] += i2 #.append(k_from_ij(i-1,j-1, m,n))
                grad_theta_indices[3] += i3 #.append(k)
            theta_v.append(single_note_val(x_hat[:, j-1]))
            theta_indices[0].append(k_from_ij(i-1,j-1, m,n))
            theta_indices[1].append(k_from_ij(i,j-1, m,n))
            i1, i2, i3, i4, v = getindices(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]))
            grad_theta_v += v #.append(note_diff(x[:, i-1] ,x_hat[:, j-1]))
            grad_theta_indices[0] += i1 #.append(k_from_ij(i-1,j-1, m,n))
            grad_theta_indices[1] += i2 #.append(k)
            grad_theta_indices[2] += i3 #.append(k_from_ij(i-1,j-1, m,n))
            grad_theta_indices[3] += i4 #.append(k)
            #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
            theta_v.append(single_note_val(x[:, i-1]))
            theta_indices[0].append(k_from_ij(i-1,j-1, m,n))
            theta_indices[1].append(k_from_ij(i-1,j, m,n))
    # s = torch.sparse_coo_tensor(i, v, (2, 3))
    # allocate memory for tensors only once
    prealloc_theta = torch.sparse_coo_tensor(theta_indices, theta_v, (m*n, m*n), device=device)
    prealloc_grad_theta = torch.sparse_coo_tensor(grad_theta_indices, grad_theta_v, (m*n, m*n, x_hat.shape[0], x_hat.shape[1]), device=device)
    return -prealloc_theta, -prealloc_grad_theta

def q_additions_speedy(parent_indices, q_vals, j):
    indices= [[],
        []]
    v = []
    for k, i in enumerate(parent_indices):
        #q[i[0],j] = q_vals[k]
        indices[0].append(i[0])
        indices[1].append(j)
        v.append(q_vals[k])
    return indices[0], indices[1], v

def speedy_sparse_diffable_recursion(theta, device, gamma=0.3): # passed in sparse
    N = theta.size()[0] # 
    e_bar = torch.zeros(N, device=device)
    e_bar[N-1]=1
    v = torch.zeros(N, device=device)
    q_indices= [[],
        []]
    q_v = []
    E_indices= [[],
        []]
    E_v = []
    for j in range(2, N): # looping through and looking at PARENTS of j
        parent_indices = get_parent_indices(theta, j) # torch.where(theta[:,j]>np.NINF)[0] # CHANGE
        u = torch.tensor(np.asarray([(i[1] + v[i[0]]) for i in parent_indices], dtype=np.float32)) # CHANGE
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma))) # this is fine
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma)) # this is fine
        #q = q + q_additions(parent_indices, q_vals, j, N, device) # creates a new tensor each time !!!
        q0, q1, qs = q_additions_speedy(parent_indices, q_vals, j)
        q_indices[0] += q0
        q_indices[1] += q1
        q_v += qs
    prealloc_q = torch.sparse_coo_tensor(q_indices, q_v, (N,N), device=device)
    for i in range(N-1,0, -1): # looping through and looking at CHILDREN of i
        children_indices = get_child_indices(theta, i) #torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            q_ij = get_ijth_val(prealloc_q, i,j[0]).to(device) # value at ij
            #E += E_val(i, j[0], q_ij*e_bar[j[0]], N, device) # creates a new tensor each time !!!!!
            #E0, E1, Es = E_val_speedy(i, j[0], q_ij*e_bar[j[0]])
            E_indices[0] += [i]
            E_indices[1] += [j[0]]
            E_v += [q_ij*e_bar[j[0]]]
            e_bar[i] += q_ij*e_bar[j[0]] # do i have to do this for an int? 
    prealloc_E = torch.sparse_coo_tensor(E_indices, E_v, (N,N), device=device)
    #print("E's equal:", torch.equal(E.to_dense(), prealloc_E.to_dense()))
    return -v[N-1], -prealloc_E

class SpeedySparseDynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    full_loss=0
    grad_L_x = torch.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) # THIS SIZE FINE
    grad_L_x.to(device)
    print(X_hat.shape[0])
    for i in range(X_hat.shape[0]):
        theta, grad_theta_xhat = construct_theta_sparse_k_loop(X[i][0], X_hat[i][0], device)
        print("theta constructed")
        theta = theta.coalesce()
        grad_theta_xhat = grad_theta_xhat.coalesce()
        loss, grad_L_theta = speedy_sparse_diffable_recursion(theta, device)
        full_loss += loss
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
    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return full_loss/X.shape[0]

  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None
    
class SpeedySparseDynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
            ) as prof:
        theta, grad_theta_xhat = construct_theta_sparse_k_loop(X, X_hat, device)
        theta = theta.coalesce()
        grad_theta_xhat = grad_theta_xhat.coalesce()
        #print("GRADTHETA:", torch.count_nonzero(grad_theta_xhat.to_dense()))
        loss, grad_L_theta = speedy_sparse_diffable_recursion(theta, device)
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
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss, grad_L_x, theta, grad_L_theta
    
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None