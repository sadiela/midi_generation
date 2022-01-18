
import numpy as np
#import torch

def indices_from_j(j, W):
    k = (j-1)%(W+1)
    l = j - k*(W+1) - 1
    return k,l

def j_from_indices(k,l, W):
    return k*(W+1) + l + 1

def top_sort():
    return 0

def construct_graph(W, wt, val, n):
    # there will be (n+1)x(W+1)+1 nodes 
    # We want a matrix representation. This will be the theta matrix
    # node 1 = S node, has a 0-weighted edge to each (0,l) node
    # don't have to be topologically sorted at first
    # we need to know which nodes correspond to which (i,j) pairs... right? 
    num_nodes= 1 + (n+1)*(W+1)
    theta = np.zeros((num_nodes, num_nodes))

    # node k,l is at k*(W+1) + l + 1

    theta[0][0] = 1
    # so, first we need an edge from the source to each (0,l) node

    # then, we determine the existence of other edges based on the weight and value properties
    for k in range(num_nodes):
        for l in range(num_nodes):
            print("")
    return 0

def diffable_knapsack():
    return 0


def grad_dp(theta, v, gamma):
    q = np.zeros(theta.shape)
    print(q.shape)
    for i in range(theta.shape[0]):
        l_i = theta[:,i] + v
        q[:,i] = np.exp(l_i/gamma)/np.sum(np.exp(l_i/gamma))

    E = np.zeros(theta.shape)

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
        v[i] = gamma * np.log10(np.sum(np.exp(u/gamma)))
        q_vals = np.exp(u/gamma)/np.sum(np.exp(u/gamma))
        #print(u, q_vals)
        for k, idx in enumerate(parent_indices):
            q[i,idx] = q_vals[k]
    for j in range(N-1,1):
        children_indices = np.where(theta[j,:]>np.NINF)[0]
        for i in children_indices:
            E[i,j] = q[i,j]*e_bar[i]
            e_bar[j] += E[i,j]

    return v[N-1], E

def diffable_recursion_1(j,theta, gamma=1): 
    if j==0: # j is the first node 
        return 0

    else: 
        # get list of parents of j --> parents of j are all non-infinite
        parent_indices = np.where(theta[:,j]>np.NINF)[0]
        #print("PARENTS:", parent_indices)
        # all edges that exist of the form (i,j)
        u =np.asarray([theta[idx,j] + recursive_formula(idx,theta) for idx in parent_indices])
        #print(j, u)
        #print("U vector:", u)
        answer = gamma * np.log10(np.sum(np.exp(u/gamma)))
        #print(j, answer)
        return answer

def recursive_formula(j, theta): 
    # we assume we have the edge representation of the graph theta (i,j) (parent,child)
    # base case
    if j==0 or np.where(theta[:,j]>np.NINF)[0].size==0: # j is the first node or has no parents
        return 0

    # recursive step
    #    This will go all the way up to (W+1)x(n+1) + 1, the total # of
    #    nodes in our graph (31 for us I think?)
    else: 
        # get list of parents of j --> parents of j are all non-infinite
        parent_indices = np.where(theta[:,j]>np.NINF)[0]
        #print("PARENTS:", parent_indices)
        # all edges that exist of the form (i,j)
        answer = max([theta[idx,j] + recursive_formula(idx,theta) for idx in parent_indices]) 
        #print(j, answer)
        return answer


def knapSack(W, wt, val, n):
    K = np.zeros((n+1, W+1)) #[[0 for x in range(W + 1)] for x in range(n + 1)]

    print(K.shape) # K is (# of items + 1) x (total possible weight) in dimension

    # K[i][j] = the max profit possible considering items from 0 to i and
    #           the total weight limit as j 
  
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
  
    print(K)
    return K[n][W]


# check algorithm is working by 
theta = np.array([
    [1      ,0      ,0      ,0      ,0,      np.NINF,6,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,6,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      9,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      9,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      9,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF,10,     np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF,np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,      np.NINF],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0      ],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0      ],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0      ],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0      ],
    [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,np.NINF],

])
  
print(diffable_recursion_1(16,theta))
v, E = diffable_recursion(theta)
print(v)
'''gammas = [ 0.1, 0.5, 1, 10, 15, 25]
for g in gammas:
    print("GAMMA =", g)
    print(diffable_recursion(16, theta, gamma=g))'''


# Driver program to test above function
'''val = [6, 10, 12, 8]
wt = [4, 2, 2, 1]
W = 5 # maximum weight
n = len(val)
print(knapSack(W, wt, val, n))'''