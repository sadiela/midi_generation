
import scipy as sp
import torch
import numpy as np 

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def note_diff(a,b):
    #a[a>0] = 1
    #b[b>0] = 1
    not_equal = np.where(np.not_equal(a,b))
    return max(torch.sum(torch.abs(a[not_equal] - b[not_equal])), 0.1) # scalar

def single_note_val(a):
    return max(torch.sum(a), 0.1)
    #return torch.sum(a) # scalar 

def distance_derivative(x):
    x[x>0] = 1
    x[x<0] = -1
    return x

def pitch_shift_distance(a,b):
    return 0

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
                    theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0.01
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

def add_gradients(idx1, idx2, idx4, deriv, m,n,):
    indices= [[],
        [],
        [],
        []]
    v = []
    #print("DERIV SHAPE:", deriv.shape)
    #input("Continue...")
    for i, val in enumerate(deriv):
        if val !=0:
            #print("nonzero derivative")
            v.append(val)
            indices[0].append(idx1)
            indices[1].append(idx2)
            indices[2].append(i)
            indices[3].append(idx4)
    return torch.sparse_coo_tensor(indices, v, (m*n,m*n,deriv.shape[0], n-1))

def construct_theta_sparse(x, x_hat):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n))
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]))
    for i in range(1,m):
            for j in range(1,n):
                if (x[:, i-1] == x_hat[:, j-1]).all():
                    theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], 0.01, (m*n, m*n)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0
                else:
                    theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], note_diff(x[:, i-1] ,x_hat[:, j-1]), (m*n, m*n)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                    grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j, m,n), j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]), m,n)
                    #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)][:,j-1] = distance_derivative(x[:,i-1]-x_hat[:,j-1]) # FIX ZEROS
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j-1, m,n)]], single_note_val(x_hat[:, j-1]), (m*n, m*n)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
                grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]), m,n)
                #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i-1,j, m,n)]],  single_note_val(x[:, i-1]), (m*n, m*n)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
                # NOTHING (gradient w.r.t. x_hat)
                # shifting?
                # gradient is telling you how much to change each x value... we will have
    #print("GRAD THETA NONZEROS:", torch.count_nonzero(grad_theta)) # there are 12...
    return -theta, -grad_theta

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

def q_additions(parent_indices, q_vals, j, N):
    indices= [[],
        []]
    v = []
    for k, i in enumerate(parent_indices):
        #q[i[0],j] = q_vals[k]
        indices[0].append(i[0])
        indices[1].append(j)
        v.append(q_vals[k])
    return torch.sparse_coo_tensor(indices, v, (N,N))

def add_E_val(idx1, idx2, val, N):
    indices = [[idx1],[idx2]]
    values = [val]
    return torch.sparse_coo_tensor(indices, values, (N,N))

def get_ijth_val(sparsemat, i,j):
    sparsemat = sparsemat.coalesce()
    indices = sparsemat.indices()
    values = sparsemat.values()
    for idx in range(indices.size()[1]):
        if indices[0,idx] == i and indices[1,idx] == j:
            return values[idx]

def sparse_diffable_recursion(theta, gamma=0.3): # passed in sparse
    N = theta.size()[0] # 
    e_bar = torch.zeros(N)
    e_bar[N-1]=1
    v = torch.zeros(N)
    q = torch.sparse_coo_tensor((N,N)) #torch.zeros((N,N)) # SPARSIFY
    E = torch.sparse_coo_tensor((N,N)) #torch.zeros((N,N)) # SPARSIFY
    for j in range(2, N): # looping through and looking at PARENTS of j
        print()
        parent_indices = get_parent_indices(theta, j) # torch.where(theta[:,j]>np.NINF)[0] # CHANGE
        #print("Parents:", parent_indices)
        u = torch.tensor(np.asarray([(i[1] + v[i[0]]) for i in parent_indices])) # CHANGE
        # u, v should be able to stay the same 
        #print(i, u)
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma))) # this is fine
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma)) # this is fine
        q = q + q_additions(parent_indices, q_vals, j, N)
    for i in range(N-1,0, -1): # looping through and looking at CHILDREN of i
        children_indices = get_child_indices(theta, i) #torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            q_ij = j[1] # value at ij
            E += add_E_val(i, j[0], q_ij*e_bar[j[0]], N)
            # E[i,j] = q[i,j]*e_bar[j]
            e_bar[i] += get_ijth_val(E, i, j[0])
    return -v[N-1], -E

def diffable_recursion(theta, gamma=0.3):
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
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    grad_L_x = torch.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) # THIS SIZE FINE
    for i in range(X_hat.shape[0]):
        theta, grad_theta_xhat = construct_theta(X[i][0], X_hat[i][0])
        theta = theta.coalesce()
        grad_theta_xhat = grad_theta_xhat.coalesce()
        #print("THETA:", theta)
        loss, grad_L_theta = diffable_recursion(theta)
        grad_L_theta.coalesce()
        #loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
        #print("LOSSES:", loss, -loss_exact)
        #print(grad_L_theta)
        n_2 = grad_L_theta.size()[0]
        #print(n_2)
        #print("DL_DTheta:", torch.count_nonzero(grad_L_theta), grad_L_theta)
        #print("DTheta_Dx:", torch.count_nonzero(grad_theta_xhat)) #, grad_L_theta)
        for i in range(n_2):
            for j in range(n_2): ##### NOT DONE W THIS PART!!!! ####
                if torch.abs(get_ijth_val(grad_L_theta, i,j)) != 0 and torch.count_nonzero(grad_theta_xhat[i][j]) != 0:
                    #print('NON ZERO PAIR', i,j, grad_L_theta[i][j], grad_theta_xhat[i][j])
                    #print(grad_L_theta[i][j] * grad_theta_xhat[i][j])
                    #  print(grad_theta_xhat[i][j])
                    #if torch.count_nonzero(grad_theta_xhat[i][j]) != 0:
                    #  print('DX IJ NON ZERO', i,j, grad_theta_xhat[i][j])
                    cur_grad = grad_L_theta[i][j] * grad_theta_xhat[i][j]
                    grad_L_x[i][0] = torch.add(grad_L_x, cur_grad)

    #grad =torch.einsum('ij,ijkl->kl', grad.double(), grad_theta.double())
    print('FINAL GRADIENT:', torch.round(grad_L_x))
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss
  
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None


mid1 = torch.tensor([
        [1,1],#,0,3],
        [0,0]#,1,0],
        ], dtype=torch.float32)  

mid2 = torch.tensor([
        [1,1],#,1,8],
        [0,1],#0,0],
        ], dtype=torch.float32) 

orig_theta, grad_theta = construct_theta(mid1, mid2)
sparsetheta, _ = construct_theta_sparse(mid1, mid2)
sparsetheta = sparsetheta.coalesce()

loss_orig, lossgrad = diffable_recursion(orig_theta, gamma=0.3)

loss_sparse, sparselossgrad = sparse_diffable_recursion(sparsetheta, gamma=0.3)

print("LOSSES:", loss_orig, loss_sparse)
print(torch.equal(loss_orig, loss_sparse))
print(torch.equal(lossgrad, sparselossgrad.to_dense()))
print(torch.eq(lossgrad, sparselossgrad.to_dense()).sum())
print(torch.norm(lossgrad - sparselossgrad.to_dense()))
#sparse_theta, sparse_grad_theta = construct_theta_sparse(mid1, mid2)

#print(sparse_grad_theta.size()[0])

#print("EQUAL GRADS:", torch.equal(grad_theta, sparse_grad_theta.to_dense()))

'''
sparse_theta = sparse_theta.to_dense()
print(sparse_theta)
sparse_theta[sparse_theta ==0] = np.NINF
#sparse_theta[sparse_theta==-0.1] = 0


print("ORIGINAL THETA:", orig_theta, orig_theta.dtype)

print("NEW THETA:", sparse_theta, sparse_theta.dtype)
print(torch.equal(orig_theta, sparse_theta))
'''
'''
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
a = torch.sparse_coo_tensor((2,3))
b = torch.sparse_coo_tensor(i, v, (2,3))
c = a + b
#print(a.to_dense(), a.indices(), a.values())
print(b.to_dense())
c =c.coalesce()
print(c.to_dense(), c.indices()[:,0], c.values())

c.indices()[:,0] # the first set of indices corresponding to the first value in the mat
c.values()[0] # CORRESPONDING VALUE!
'''
