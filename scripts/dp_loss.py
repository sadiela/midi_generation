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
    a[a>0] = 1
    b[b>0] = 1
    not_equal = np.where(np.not_equal(a,b))
    return np.abs(np.count_nonzero(a[not_equal])+np.count_nonzero(b[not_equal]))

def single_note_val(a):
    return np.count_nonzero(a)

def construct_theta(midi1, midi2):
    m = midi1.shape[1] + 1
    n = midi2.shape[1] + 1
    theta = np.zeros((m*n, m*n))
    print(theta.shape)
    theta[:,:] = np.Inf

    for idx in range(m):
        theta[0, m] = 

    for i in range(m-1):
        for j in range(n-1):
            #print(str1[i], str2[j])
            if (midi1[:, i] == midi2[:, j]).all():
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = 0
            else:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = num_note_diff(midi1[:, i] ,midi2[:, j] ) # replacing; cost depends on ...?
            theta[k_from_ij(i,j, m,n)][k_from_ij(i,j+1, m,n)]= single_note_val(midi2[:, j])# deletion
            theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j, m,n)]= single_note_val(midi1[:, i]) # insertion
            # shifting?
    return -theta

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
            
def diffable_recursion(theta, gamma=1):
    N = theta.shape[0] 
    e_bar = np.zeros(N)
    e_bar[N-1]=1
    v = np.zeros(N)
    q = np.zeros((N,N))
    E = np.zeros((N,N))
    for i in range(2, N):
        parent_indices = np.where(theta[:,i]>np.NINF)[0]
        #print("Parents:", parent_indices)
        u = np.asarray([theta[idx,i] + v[idx] for idx in parent_indices])
        #print(i, u)
        v[i] = gamma * np.log(np.sum(np.exp(u/gamma)))
        q_vals = np.exp(u/gamma)/np.sum(np.exp(u/gamma))
        #print(u, q_vals)
        for k, idx in enumerate(parent_indices):
            q[i,idx] = q_vals[k]
    for j in range(N-1,0, -1):
        children_indices = np.where(theta[j,:]>np.NINF)[0]
        for i in children_indices:
            E[i,j] = q[i,j]*e_bar[i]
            e_bar[j] += E[i,j]

    return v[N-1], E
            
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


# try with two example midis:
mid1 = np.array([
    [0,0,0,0,0,2],
    [2,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,5,0,1,0,2],
    [0,0,0,0,0,0]
    ])  

mid2 = np.array([
    [1,0,0,0,0,0],
    [0,0,2,0,0,0],
    [0,2,0,0,0,0],
    [0,0,0,1,0,2],
    [0,0,0,0,0,0]
    ])  

theta= construct_theta(mid1, mid2)
ans1 = exact_recursive_formula(theta.shape[0]-1,theta)
ans2, E = diffable_recursion(theta, gamma=0.1)
print(-ans1, -ans2)
