import numpy as np

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def construct_theta(str1, str2):
    m = len(str1) + 1
    n = len(str2) + 1
    theta = np.zeros((m*n, m*n))
    theta[:,:] = np.NINF

    for i in range(m-1):
        for j in range(n-1):
            print(str1[i], str2[j])
            if str1[i] == str2[j]:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = 0
            else:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = -1
            theta[k_from_ij(i,j, m,n)][k_from_ij(i,j+1, m,n)]=-1
            theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j, m,n)]=-1
    return theta
            
            
def exact_recursive_formula(j, theta): 
    # we assume we have the edge representation of the graph theta (i,j) (parent,child)
    # base case
    if j==0 or np.where(theta[:,j]>np.NINF)[0].size==0: # j is the first node or has no parents
        return 0

    else: 
        # get list of parents of j --> parents of j are all >-infinity
        parent_indices = np.where(theta[:,j]>np.NINF)[0]
        print(j, " PARENTS:", parent_indices)

        # just get the max length path
        answer = max([theta[idx,j] + exact_recursive_formula(idx,theta) for idx in parent_indices]) 
        return answer

# ORIGINAL EDIT DISTANCE PROBLEM 
def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
 
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
 
    return dp[m][n]
 
 
# Driver code
str1 = "ma"
str2 = "mom"
 
theta = construct_theta(str1, str2)
print(theta)
print(exact_recursive_formula(11, theta))
#print(editDistDP(str1, str2, len(str1), len(str2)))