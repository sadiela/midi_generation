import torch
import numpy as np 
from shared_functions import *

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