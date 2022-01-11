# new loss function that uses dynamic programming
# NOT differentiable

# Input: 2 tensor MIDI representations
# Output: loss characterizing how far away the two representations are from each other

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
