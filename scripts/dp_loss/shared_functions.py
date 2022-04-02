import torch
import numpy as np 
#from dp_loss import *
#import torch.nn.functional as F

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def note_diff(a,b):
    not_equal = torch.where(torch.not_equal(a,b))
    return max(torch.sum(torch.abs(a[not_equal] - b[not_equal])), 0.001) # scalar

def single_note_val(a):
    return max(torch.sum(a), 0.001)
    #return torch.sum(a) # scalar 

def distance_derivative(x):
    x[x>0] = 1
    x[x<0] = -1
    return x

def pitch_shift_distance(a,b):
    return 0

########################
### SPARSE FUNCTIONS ###
########################

def sparse_add_gradients(idx1, idx2, idx4, deriv, m, n, device):
    indices= [[],
        [],
        [],
        []]
    v = []

    for i, val in enumerate(deriv):
        if val !=0:
            v.append(val)
            indices[0].append(idx1)
            indices[1].append(idx2)
            indices[2].append(i)
            indices[3].append(idx4)
    return torch.sparse_coo_tensor(indices, v, (m*n,m*n,deriv.shape[0], n-1), device=device)

def get_parent_indices(theta, j):
    # we want all the nonzero values in the jth column (i.e. 2nd index is j)
    parent_indices = []
    indices = theta.indices() 
    values = theta.values()
    for i, idx2 in enumerate(indices[1,:]): # loop through column indices
        if idx2 == j:
            parent_indices.append([indices[0,i], values[i]])
    return parent_indices

def get_child_indices(theta, i):
    children_indices = []
    indices = theta.indices() 
    values = theta.values()
    for j, idx1 in enumerate(indices[0]): # loop through 
        if idx1 == i:
            children_indices.append([indices[1,j], values[j]])
    return children_indices

def q_additions(parent_indices, q_vals, j, N, device):
    indices= [[],
        []]
    v = []
    for k, i in enumerate(parent_indices):
        #q[i[0],j] = q_vals[k]
        indices[0].append(i[0])
        indices[1].append(j)
        v.append(q_vals[k])
    return torch.sparse_coo_tensor(indices, v, (N,N), device=device)

def E_val(idx1, idx2, val, N, device):
    indices = [[idx1],[idx2]]
    values = [val]
    return torch.sparse_coo_tensor(indices, values, (N,N), device=device)

def get_ijth_val(sparsemat, i,j):
    sparsemat = sparsemat.coalesce()
    indices = sparsemat.indices()
    values = sparsemat.values()
    for idx in range(indices.size()[1]):
        if indices[0,idx] == i and indices[1,idx] == j:
            return values[idx]
    return 0

def has_values(sparse_mat, i, j):
    sparse_mat = sparse_mat.coalesce()
    indices = sparse_mat.indices()
    for idx1, idx2 in zip(indices[0], indices[1]):
        if i==idx1 and j==idx2:
            return True
    return False

def get_slice(sparse_mat, idx1,idx2, device):
    m, n = sparse_mat.size()[2], sparse_mat.size()[3]
    sparse_mat = sparse_mat.coalesce()
    indices = sparse_mat.indices()
    vals = sparse_mat.values()
    result = torch.zeros((m,n), device=device)
    for i, j, k, l, v in zip(indices[0],indices[1],indices[2],indices[3], vals):
        if i==idx1 and j==idx2:
            result[k][l] = v
    return result