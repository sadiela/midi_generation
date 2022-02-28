
import torch
import numpy as np 

def construct_theta(x, x_hat):
    print("CONSTRUCTING THETA:", x.shape, x_hat.shape)
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
                    theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0
                else:
                    theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                    grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)][:,j-1] = distance_derivative(x[:,i-1]-x_hat[:,j-1]) # FIX ZEROS
                theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
                grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
                theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
                # NOTHING (gradient w.r.t. x_hat)
                # shifting?
                # gradient is telling you how much to change each x value... we will have
        #print("GRAD THETA NONZEROS:", torch.count_nonzero(grad_theta)) # there are 12...
        return -theta, -grad_theta
    except RuntimeError as err:
        print(err, x.shape, x_hat.shape)
        return torch.zeros((m*n, m*n)), torch.zeros((m*n, m*n, x_hat.shape[0], x_hat.shape[1]))

def construct_theta(x, x_hat):
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n))
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]))




i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
a = torch.sparse_coo_tensor((2,3))
b = torch.sparse_coo_tensor(i, v, (2,3))
c = a + b
print(a.to_dense())
print(b.to_dense())
print(c.to_dense())