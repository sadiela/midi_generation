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

def num_note_diff(a,b):
    #a[a>0] = 1
    #b[b>0] = 1
    not_equal = np.where(np.not_equal(a,b))
    return torch.sum(torch.abs(a[not_equal] - b[not_equal])) # scalar

def single_note_val(a):
    return torch.sum(a) # scalar 

def construct_theta(midi1, midi2):
    # can I construct gradient of theta alongside this? 
    # for each theta (k,l), will have gradient w.r.t. midi2
    #   gradient will be 0 for all except midi2[:,j]
    m = midi1.shape[1] + 1
    n = midi2.shape[1] + 1
    theta = torch.zeros((m*n, m*n))
    grad_theta = torch.zeros((m*n, m*n, midi2.shape[0], midi2.shape[1]))
    print(midi1.shape, midi2.shape, grad_theta.shape)
    theta[:,:] = np.Inf

    for i in range(m-1):
        for j in range(n-1):
            #print(midi1[:,i].shape, grad_theta[0][0][:,0].shape)
            #print(str1[i], str2[j])
            if (midi1[:, i] == midi2[:, j]).all():
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = 0
                #grad_theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)][:][:] = np.zeros((midi2.shape[0], midi2.shape[1]))
            else:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = num_note_diff(midi1[:, i] ,midi2[:, j]) # replacing; cost depends on ...?
                grad_theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)][:,j] = np.abs(midi1[:,i]-midi2[:,j])
            theta[k_from_ij(i,j, m,n)][k_from_ij(i,j+1, m,n)]= single_note_val(midi2[:, j])# deletion
            grad_theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)][:,j] = midi2[:,j]
            theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j, m,n)]= single_note_val(midi1[:, i]) # insertion I think i want these both dependent on midi2... is that possible? 
            # NOTHING (gradient w.r.t. midi2)
            # shifting?
            # gradient is telling you how much to change each x value... we will have
    print("GRAD THETA NONZEROS:", torch.count_nonzero(grad_theta)) # there are 12...
    return -theta, -grad_theta

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
            
def diffable_recursion(theta, gamma=0.5):
    N = theta.shape[0] 
    e_bar = torch.zeros(N)
    e_bar[N-1]=1
    v = torch.zeros(N)
    q = torch.zeros((N,N))
    E = torch.zeros((N,N))
    for i in range(2, N):
        parent_indices = torch.where(theta[:,i]>np.NINF)[0]
        print("Parents:", parent_indices)
        u = torch.tensor(np.asarray([theta[idx,i] + v[idx] for idx in parent_indices]))
        #print(i, u)
        v[i] = gamma * torch.log(torch.sum(torch.exp(u/gamma)))
        print(i, v[i])
        q_vals = torch.exp(u/gamma)/torch.sum(torch.exp(u/gamma))
        #print(u, q_vals)
        for k, idx in enumerate(parent_indices):
            q[i,idx] = q_vals[k]
    for j in range(N-1,0, -1):
        children_indices = torch.where(theta[j,:]>np.NINF)[0]
        for i in children_indices:
            E[i,j] = q[i,j]*e_bar[i]
            e_bar[j] += E[i,j]

    return -v[N-1], -E
            
def dp_loss(y, n, y_hat, m): 
    # assume y, y_hat the same dimension (pxn)
    
    #_,n = y.shape
    #_,m = y_hat.shape
    
    # base case
    if m == 0:
        return n 
    if n == 0:
        return m
    
    cost = sum(y[:, m-1] - y_hat[:, n-1])

    return min(dp_loss(y, n-1, y_hat, m) + 1,
               dp_loss(y, n, y_hat, m-1) + 1, 
               dp_loss(y, n-1, y_hat, m-1) + cost
            )

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
