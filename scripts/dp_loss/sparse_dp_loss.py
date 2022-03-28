import torch
import numpy as np 
#from dp_loss import *
import torch.nn.functional as F

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def note_diff(a,b):
    #a[a>0] = 1
    #b[b>0] = 1
    not_equal = torch.where(torch.not_equal(a,b))
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

def add_gradients(idx1, idx2, idx4, deriv, m, n, device):
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
    return torch.sparse_coo_tensor(indices, v, (m*n,m*n,deriv.shape[0], n-1), device=device)

def construct_theta_sparse(x, x_hat, device):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n), device=device)
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]), device=device)
    for i in range(1,m):
            for j in range(1,n):
                print(i,j)
                if (x[:, i-1] == x_hat[:, j-1]).all():
                    theta = torch.add(theta, torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], 0.01, (m*n, m*n), device=device)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0
                else:
                    theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j, m,n)]], note_diff(x[:, i-1] ,x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                    grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j, m,n), j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]), m,n, device=device)
                    #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)][:,j-1] = distance_derivative(x[:,i-1]-x_hat[:,j-1]) # FIX ZEROS
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j-1, m,n)]], single_note_val(x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
                grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]), m,n, device=device)
                #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i-1,j, m,n)]],  single_note_val(x[:, i-1]), (m*n, m*n), device=device)#theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
                # NOTHING (gradient w.r.t. x_hat)
                # shifting?
                # gradient is telling you how much to change each x value... we will have
    return -theta, -grad_theta

def construct_theta_sparse_k_loop(x, x_hat, device):
    # theta: only adding one entry at a time
    # grad_theta: add rows of entries
    m = x.shape[1] + 1
    n = x_hat.shape[1] + 1
    theta = torch.sparse_coo_tensor((m*n, m*n), device=device)
    grad_theta = torch.sparse_coo_tensor((m*n, m*n,  x_hat.shape[0], x_hat.shape[1]), device=device)
    for k in range(0,m*n):
        i, j = ij_from_k(k,m*n)
        #print("k,i,j:", k, i, j)
        if i > 0 and j > 0:
            if (x[:, i-1] == x_hat[:, j-1]).all():
                theta = torch.add(theta, torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k]], 0.01, (m*n, m*n), device=device)) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = 0
            else:
                theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k]], note_diff(x[:, i-1] ,x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j, m,n)] = note_diff(x[:, i-1] ,x_hat[:, j-1]) # replacing; cost depends on ...?
                grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k, j-1, distance_derivative(x[:,i-1]-x_hat[:,j-1]), m,n, device=device)
            theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i,j-1, m,n)]], single_note_val(x_hat[:, j-1]), (m*n, m*n), device=device) #theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)]= single_note_val(x_hat[:, j-1])# deletion
            grad_theta = grad_theta + add_gradients(k_from_ij(i-1,j-1, m,n), k_from_ij(i,j-1, m,n), j-1, distance_derivative(-x_hat[:,j-1]), m,n, device=device)
            #grad_theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i,j-1, m,n)][:,j-1] = distance_derivative(-x_hat[:,j-1]) #, np.abs(-x_hat[:,j-1])) # FIX
            theta = theta + torch.sparse_coo_tensor([[k_from_ij(i-1,j-1, m,n)],[k_from_ij(i-1,j, m,n)]],  single_note_val(x[:, i-1]), (m*n, m*n), device=device)#theta[k_from_ij(i-1,j-1, m,n)][k_from_ij(i-1,j, m,n)] = single_note_val(x[:, i-1]) # insertion I think i want these both dependent on x_hat... is that possible? 
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

def sparse_diffable_recursion(theta, device, gamma=0.3): # passed in sparse
    N = theta.size()[0] # 
    e_bar = torch.zeros(N, device=device)
    e_bar[N-1]=1
    v = torch.zeros(N, device=device)
    q = torch.sparse_coo_tensor((N,N), device=device) #torch.zeros((N,N)) # SPARSIFY
    E = torch.sparse_coo_tensor((N,N), device=device) #torch.zeros((N,N)) # SPARSIFY
    for j in range(2, N): # looping through and looking at PARENTS of j
        parent_indices = get_parent_indices(theta, j) # torch.where(theta[:,j]>np.NINF)[0] # CHANGE
        #print("Parents:", parent_indices)
        u = torch.tensor(np.asarray([(i[1] + v[i[0]]) for i in parent_indices], dtype=np.float32)) # CHANGE
        # u, v should be able to stay the same 
        #print(i, u)
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

def has_values(sparse_mat, i, j):
    sparse_mat = sparse_mat.coalesce()
    indices = sparse_mat.indices()
    for idx1, idx2 in zip(indices[0], indices[1]):
        if i==idx1 and j==idx2:
            return True
    return False

def get_slice(sparse_mat, idx1,idx2):
    m, n = sparse_mat.size()[2], sparse_mat.size()[3]
    sparse_mat = sparse_mat.coalesce()
    indices = sparse_mat.indices()
    vals = sparse_mat.values()
    result = torch.zeros((m,n), device=device)
    for i, j, k, l, v in zip(indices[0],indices[1],indices[2],indices[3], vals):
        if i==idx1 and j==idx2:
            result[k][l] = v
    return result

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
        theta, grad_theta_xhat = construct_theta_sparse_k_loop(X[i][0], X_hat[i][0], device)
        theta = theta.coalesce()
        grad_theta_xhat = grad_theta_xhat.coalesce()
        #print("THETA:", theta)
        loss, grad_L_theta = sparse_diffable_recursion(theta, device)
        grad_L_theta.coalesce()
        #loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
        #print("LOSSES:", loss, -loss_exact)
        n_2 = grad_L_theta.size()[0]
        #print(n_2)
        # Just loop through sparse index pairs!
        grad_L_theta = grad_L_theta.coalesce()
        nonzero_indices = grad_L_theta.indices()
        nonzero_vals = grad_L_theta.values()
        for idx in range(nonzero_indices.size()[1]):
            j = nonzero_indices[0,idx] 
            k = nonzero_indices[1,idx] 
            grad_L_theta_val = nonzero_vals[idx]
            if has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k)# scalar times pxn
                grad_L_x[i][0] += cur_grad

    #print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss
    
class SparseDynamicLossSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X_hat, X, device):
    # X_hat, X are bigger than we thought...
    # build theta from original data and reconstruction
    theta, grad_theta_xhat = construct_theta_sparse_k_loop(X, X_hat, device)
    theta = theta.coalesce()
    grad_theta_xhat = grad_theta_xhat.coalesce()
    print("GRADTHETA:", torch.count_nonzero(grad_theta_xhat.to_dense()))
    loss, grad_L_theta = sparse_diffable_recursion(theta, device)
    grad_L_theta.coalesce()
    print("GRADLTHETA:", torch.count_nonzero(grad_L_theta.to_dense()))
    #loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
    #print("LOSSES:", loss, -loss_exact)
    #print(grad_L_theta)
    n_2 = grad_L_theta.size()[0]
    #print(n_2)
    '''grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    for j in range(n_2):
        for k in range(n_2): ##### NOT DONE W THIS PART!!!! ####
            grad_L_theta_val = get_ijth_val(grad_L_theta, j,k)
            if grad_L_theta_val != 0 and has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
                cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k) # scalar times pxn
                grad_L_x += cur_grad'''
    grad_L_x = torch.zeros((X_hat.shape[0], X_hat.shape[1]))
    grad_L_theta = grad_L_theta.coalesce()
    nonzero_indices = grad_L_theta.indices()
    nonzero_vals = grad_L_theta.values()
    for idx in range(nonzero_indices.size()[1]):
        j = nonzero_indices[0,idx] 
        k = nonzero_indices[1,idx] 
        grad_L_theta_val = nonzero_vals[idx]
        if has_values(grad_theta_xhat, j,k): #torch.count_nonzero(grad_theta_xhat[j][k]) != 0:
            cur_grad = grad_L_theta_val *  get_slice(grad_theta_xhat, j,k)# scalar times pxn
            grad_L_x += cur_grad
    

    #grad =torch.einsum('ij,ijkl->kl', grad.double(), grad_theta.double())
    print('FINAL GRADIENT:', grad_L_x)
    ctx.save_for_backward(grad_L_x)
    # determine answer
    return loss
  
  @staticmethod
  def backward(ctx, grad_output):
    grad_L_x, = ctx.saved_tensors
    return grad_L_x, None

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mid1 = torch.tensor([
            [1,1, 0],#,0,3],
            [1,0, 0]#,1,0],
            ], dtype=torch.float32)  

    mid2 = torch.tensor([
            [1,0, 0],#,1,8],
            [0,1, 1],#0,0],
            ], dtype=torch.float32) 

    #origtheta, gradtheta = construct_theta(mid1, mid2)
    #sparsetheta, sparsegradtheta = construct_theta_sparse(mid1, mid2, device=device)
    #sparsetheta = sparsetheta.coalesce()
    #print(sparsegradtheta)
    #loss_orig, lossgrad = diffable_recursion(origtheta, gamma=0.3)

    #loss_sparse, sparselossgrad = sparse_diffable_recursion(sparsetheta, device, gamma=0.3)

    sparsedyn = SparseDynamicLossSingle.apply 
    sparse_dyn_loss = sparsedyn(mid2,mid1, device)
    print(sparse_dyn_loss)
    #sparsetheta = sparsetheta.to_dense()
    #sparsegradtheta = sparsegradtheta.to_dense()
    #sparsetheta[sparsetheta==0] = np.NINF
    #print(sparsegradtheta)

    '''print("LOSSES:", loss_orig, loss_sparse)
    print("LOSSES EQUAL:", torch.equal(loss_orig, loss_sparse))
    print("THETAS:", torch.equal(origtheta, sparsetheta))
    print("GRAD THETAS:", torch.equal(gradtheta, sparsegradtheta.to_dense()))
    print("GRAD LOSSES:", torch.equal(lossgrad, sparselossgrad.to_dense()))

    print("MATCHING INDICES:", torch.eq(lossgrad, sparselossgrad.to_dense()).sum(), "out of", lossgrad.numel())

    midis1 = [mid1, mid2]
    midis2 = [mid2, mid1]

    def test_sparse_loss():
        for m1, m2 in zip(midis1, midis2):
            origtheta, gradtheta = construct_theta(m1, m2)
            sparsetheta, sparsegradtheta = construct_theta_sparse(m1, m2)
            sparsetheta = sparsetheta.coalesce()
            loss_orig, lossgrad = diffable_recursion(origtheta, gamma=0.3)
            loss_sparse, sparselossgrad = sparse_diffable_recursion(sparsetheta, gamma=0.3)
            sparsetheta = sparsetheta.to_dense()
            sparsetheta[sparsetheta==0] = np.NINF
            assert (torch.equal(origtheta, sparsetheta))
            assert (torch.equal(gradtheta, sparsegradtheta.to_dense()))
            assert (torch.equal(loss_orig, loss_sparse))
            assert (torch.equal(lossgrad, sparselossgrad.to_dense()))

    test_sparse_loss()
    '''
    '''
    dynamic_loss = DynamicLossSingle.apply
    sparse_dynamic_loss = SparseDynamicLossSingle.apply

    #print("LOSSES:", loss_orig, loss_sparse)
    #print(torch.equal(loss_orig, loss_sparse))
    #print(torch.equal(lossgrad, sparselossgrad.to_dense()))
    #print(torch.eq(lossgrad, sparselossgrad.to_dense()).sum())
    #print(torch.norm(lossgrad - sparselossgrad.to_dense()))
    #sparse_theta, sparse_grad_theta = construct_theta_sparse(mid1, mid2)

    l2_loss = F.mse_loss(mid1, mid2)
    dyn_loss = dynamic_loss(mid1, mid2)
    sparse_dyn_loss = sparse_dynamic_loss(mid1,mid2)
    print("L2:", l2_loss.data)
    print("Dynamic:", dyn_loss.data)
    print("Sparse dynamic:", sparse_dyn_loss.data)

    #print(sparse_grad_theta.size()[0])

    #print("EQUAL GRADS:", torch.equal(grad_theta, sparse_grad_theta.to_dense()))


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
