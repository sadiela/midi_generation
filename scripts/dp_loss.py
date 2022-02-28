# new loss function that uses dynamic programming
# NOT differentiable

# Input: 2 tensor MIDI representations
# Output: loss characterizing how far away the two representations are from each other
import numpy as np 
import torch

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def note_diff(a,b):
    #a[a>0] = 1
    #b[b>0] = 1
    not_equal = np.where(np.not_equal(a,b))
    return torch.sum(torch.abs(a[not_equal] - b[not_equal])) # scalar

def single_note_val(a):
    return torch.sum(a) # scalar 

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

def exact_recursive_formula(j, theta): 
    # we assume we have the edge representation of the graph theta (i,j) (parent,child)
    # base case
    if j==0:# or np.where(theta[:,j]>np.NINF)[0].size==0: # j is the first node or has no parents
        return 0

    else: 
        # get list of parents of j --> parents of j are all >-infinity
        parent_indices = np.where(theta[:,j]>np.NINF)[0]
        #print(j, " PARENTS:", parent_indices)

        # just get the max length path
        answer = max([theta[idx,j] + exact_recursive_formula(idx,theta) for idx in parent_indices]) 
        return answer

'''      
def diffable_recursion(theta, gamma=0.5):
    N = theta.shape[0] 
    e_bar = torch.zeros(N)
    e_bar[N-1]=1
    v = torch.zeros(N)
    q = torch.zeros((N,N))
    E = torch.zeros((N,N))
    for j in range(2, N): # go through children
        parent_indices = torch.where(theta[:,j]>np.NINF)[0]
        print("Parents:", parent_indices)
        u = torch.tensor(np.asarray([theta[idx,j] + v[idx] for idx in parent_indices]))
        #print(i, u)
        v[j] = gamma * torch.log(torch.sum(torch.exp(u/gamma)))
        print(j, v[j])
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma)) # q gradients
        #print(u, q_vals)
        for k, idx in enumerate(parent_indices):
            q[idx,j] = q_vals[k]
    for i in range(N-1,0, -1): # i is the parent index
        children_indices = torch.where(theta[i,:]>np.NINF)[0]
        for j in children_indices:
            E[i,j] = e_bar[i]*q[i,j]
            e_bar[j] += E[i,j]

    return -v[N-1], -E'''

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
        #print("THETA:", theta)
        loss, grad_L_theta = diffable_recursion(theta)
        loss_exact = exact_recursive_formula(theta.shape[0]-1, theta)
        #print("LOSSES:", loss, -loss_exact)
        #print(grad_L_theta)
        n_2 = grad_L_theta.shape[0]
        #print(n_2)
        #print("DL_DTheta:", torch.count_nonzero(grad_L_theta), grad_L_theta)
        #print("DTheta_Dx:", torch.count_nonzero(grad_theta_xhat)) #, grad_L_theta)
        for i in range(n_2):
            for j in range(n_2):
                if torch.abs(grad_L_theta[i][j]) != 0 and torch.count_nonzero(grad_theta_xhat[i][j]) != 0:
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

if __name__ == "__main__":
    # try with two example midis:
    mid1 = np.array([
        [1,1,2,0],
        [0,0,0,1],
        [0,20,0,0]
        ])  

    mid2 = np.array([
        [0,20,0,10],
        [0,10,0,10],
        [0,0,0,10]
        ])  

    mid1 = torch.from_numpy(mid1, requires_grad=True)
    mid2 = torch.from_numpy(mid2, requires_grad=True)

    theta, grad_theta = construct_theta(mid1, mid2)
    ans1 = exact_recursive_formula(theta.shape[0]-1,theta)
    ans2, E = diffable_recursion(theta, gamma=0.1)
    print(-ans1, -ans2)
