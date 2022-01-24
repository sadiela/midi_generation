import numpy as np

def ij_from_k(k, N):
    return k//N, k%N - 1

def k_from_ij(i,j,m,n):
    return n*i + j 

def construct_theta(str1, str2):
    m = len(str1) + 1
    n = len(str2) + 1
    theta = np.zeros((m*n, m*n))
    theta[:,:] = np.Inf

    for i in range(m-1):
        for j in range(n-1):
            #print(str1[i], str2[j])
            if str1[i] == str2[j]:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = 0
            else:
                theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j+1, m,n)] = 1 # replacing; cost depends on ...?
            theta[k_from_ij(i,j, m,n)][k_from_ij(i,j+1, m,n)]= 1 # insertion
            theta[k_from_ij(i,j, m,n)][k_from_ij(i+1,j, m,n)]= 1 # deletion
            # shifting?
    return -theta
            
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

# ORIGINAL EDIT DISTANCE PROBLEM 
def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
 
            # If first string is empty, only option is to
            # insert all characters of second string

            try: 
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
            except IndexError:
                print("ERROR", str1, str2, i, j)
 
    return dp[m][n]
 
 
# NEED TO FORMULATE AS A MAXIMIZATION PROBLEM!
#   Instead of minimum edit distance, we want to maximize the negative of the edit distance
#   Can do this by inverting edge weights 

# Driver code
str1 = "Saturday"
str2 = "Sunday"
 
theta = construct_theta(str1, str2)
print(theta)
print(editDistDP(str1, str2, 8, 6), -exact_recursive_formula(theta.shape[0]-1, theta))
#print(editDistDP(str1, str2, len(str1), len(str2)))
list1 = ['mom', 'saturday', 'today', 'hello']
list2 = ['maim', 'sunday', 'tomorrow', 'help']
g = 0.05

for w1, w2 in zip(list1, list2):
    print(w1, w2)
    theta = construct_theta(w1, w2)
    print(theta.shape)
    ans1 = editDistDP(w1, w2, len(w1), len(w2))
    ans2 = -exact_recursive_formula(theta.shape[0]-1,theta)
    ans3 = -diffable_recursion(theta, g)[0]
    print(ans1, ans2, ans3)

# what alignment does the "answer" refer to?