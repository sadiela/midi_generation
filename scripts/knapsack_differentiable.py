
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

def recursive_formula(j, theta):
    # base case
    if j==1: # j is the first node 
        return 0

    # recursive step
    # we will go all the way up to (W+1)x(n+1) + 1, the total # of
    #    nodes in our graph (31 for us I think?)
    else: 
        # get list of parents of j
        # all edges that exist of the form (i,j)
        i_s = []
        indices = [i*(W+1) + j + 1 for  i in i_s]
        #choices = [theta[idx] + recursion_approx(idx,theta) for  idx in indices]
        return max([theta[idx] + recursive_formula(idx,theta) for idx in indices]) 


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
  
# Driver program to test above function
val = [6, 10, 12, 8]
wt = [4, 2, 2, 1]
W = 5 # maximum weight
n = len(val)
print(knapSack(W, wt, val, n))