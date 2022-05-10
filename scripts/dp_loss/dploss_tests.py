import itertools 
import torch
import numpy as np
import torch.nn.functional as F

#rom scripts.dp_loss.diffable_dp_loss 
from exact_dp_loss import exact_recursive_formula
from diffable_dp_loss import construct_theta, DynamicLossSingle, diffable_recursion
from sparse_dp_loss import construct_theta_sparse, SparseDynamicLossSingle
from speedy_sparse_dp_loss import construct_theta_sparse_k_loop, SpeedySparseDynamicLossSingle

'''
The DP loss code has undergone several iterative improvements to deal with various problems it has had.
1. Original DP loss function: NOT a smooth approximation
2. Differentiable DP loss: using implementation detailed in DP paper
3. Sparse differentiable DP loss: to deal with large memory requirements
4. Speedy sparse differentiable DP loss: several improvements made to decrease runtime

We will compare these four methods to see how their losses, gradients (when applicable), and runtimes compare
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mid1 = torch.tensor([
        [1,1,0,0],
        [1,0,0,0],
        ], dtype=torch.float32)  

mid2 = torch.tensor([
        [1,0,0,0],
        [0,1,3,7],
        ], dtype=torch.float32) 

mid3 = torch.tensor([
        [1,0,1,8],
        [0,1,0,3],
        ], dtype=torch.float32) 

mid4 = torch.tensor([
        [1,0,1,5],
        [0,1, 0,0],
        ], dtype=torch.float32) 

midis= [mid1, mid2, mid3, mid4]

# Three versions of the loss function
dynamic_loss = DynamicLossSingle.apply
sparse_dynamic_loss = SparseDynamicLossSingle.apply
speedy_sparse_dynamic_loss = SpeedySparseDynamicLossSingle.apply

def test_exact():
    theta1, _ = construct_theta(mid2, mid1)
    exact_loss1 = exact_recursive_formula(theta1.shape[0]-1, theta1)
    theta2, _ = construct_theta(mid1, mid2)
    exact_loss2 = exact_recursive_formula(theta2.shape[0]-1, theta2)
    print(f"Losses:{exact_loss1:.6f} {exact_loss2:6f}")

def test_diffable_dp():
    loss1, grad_L_x1, theta, grad_L_theta = dynamic_loss(mid2, mid1)
    loss2, grad_L_x2, theta, grad_L_theta = dynamic_loss(mid1, mid2)
    print(f"Losses:{loss1:.6f} {loss2:6f}")
    print("Gradients:")
    print(grad_L_x1)
    print(grad_L_x2)

def test_sparse_dp():
    loss1, grad_L_x1, theta, grad_L_theta = sparse_dynamic_loss(mid2, mid1, device)
    loss2, grad_L_x2, theta, grad_L_theta = sparse_dynamic_loss(mid1, mid2, device)
    print(f"Losses:{loss1:.6f} {loss2:6f}")
    print("Gradients:")
    print(grad_L_x1)
    print(grad_L_x2)

def test_speedy_sparse_dp():
    loss1, grad_L_x1, theta, grad_L_theta = speedy_sparse_dynamic_loss(mid2, mid1, device)
    loss2, grad_L_x2, theta, grad_L_theta = speedy_sparse_dynamic_loss(mid1, mid2, device)
    print(f"Losses:{loss1:.6f} {loss2:6f}")
    print("Gradients:")
    print(grad_L_x1)
    print(grad_L_x2)

# See how varying gammas affects the loss approximation
def vary_gamma():
    gammas = [round(.10*i, 3) for i in range(12)]
    print(gammas)
    theta, _ = construct_theta(mid4, mid3)
    exact_loss = exact_recursive_formula(theta.shape[0]-1, theta)
    losses = []
    for g in gammas:
        cur_loss, _ = diffable_recursion(theta, g)
        losses.append(cur_loss)
    print("{:<10} {:<10}".format("gamma val", "loss"))
    print("{:<10} {:<10}".format("EXACT", -exact_loss))
    for g, loss in zip(gammas, losses):
        print ("{:<10} {:<10}".format(g, loss))

def compare_thetas():
    # compare the thetas generated by the following algorithms:
    #   construct_theta_sparse_k_loop
    #   construct_theta_sparse
    #   construct_theta
    for pair in itertools.permutations(midis, 2):
        orig_theta, _ = construct_theta(pair[0], pair[1], zero= 0.001)
        sparse_theta, _ = construct_theta_sparse(pair[0],pair[1], device)
        speedy_sparse_theta, _ = construct_theta_sparse_k_loop(pair[0],pair[1], device) # NOT WORKING!
        sparse_theta = sparse_theta.to_dense()
        speedy_sparse_theta = speedy_sparse_theta.to_dense()
        sparse_theta[sparse_theta==0] = np.NINF
        speedy_sparse_theta[speedy_sparse_theta==0] = np.NINF
        #print(sparse_theta, speedy_sparse_theta)
        assert torch.equal(orig_theta, sparse_theta)
        print(orig_theta.numel() - torch.eq(orig_theta, sparse_theta).sum(), orig_theta.numel() - torch.eq(sparse_theta, speedy_sparse_theta).sum())
        #assert torch.equal(sparse_theta,speedy_sparse_theta)'''

def compare_losses():
    # compare the losses generated by the following algorithms:
    #   construct_theta_sparse_k_loop
    #   construct_theta_sparse
    #   construct_theta
    print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format("L2", "exact_zero:", "exact", "original", "sparse", "speedy"))
    for pair in itertools.permutations(midis, 2):
        theta_zero, _ = construct_theta(pair[0], pair[1], zero= 0)
        theta, _ = construct_theta(pair[0], pair[1], zero= 0.001)
        l2_loss = F.mse_loss(pair[0], pair[1])
        exact_loss_zero = exact_recursive_formula(theta.shape[0]-1, theta)
        exact_loss = exact_recursive_formula(theta_zero.shape[0]-1, theta_zero)
        loss,_,_,_ = dynamic_loss(pair[0],pair[1])
        sparse_loss,_,_,_ = sparse_dynamic_loss(pair[0],pair[1], device) # NOT WORKING!
        speedy_sparse_loss,_,_,_ = speedy_sparse_dynamic_loss(pair[0],pair[1], device) # NOT WORKING!
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(l2_loss.data, -exact_loss_zero, -exact_loss, loss, sparse_loss, speedy_sparse_loss))

def compare_gradients(): 
    for pair in itertools.permutations(midis, 2):
        _,grad_L_x,_,_ = dynamic_loss(pair[0],pair[1])
        _,sparse_grad_L_x,_,_ = sparse_dynamic_loss(pair[0],pair[1], device) # NOT WORKING!
        _,speedy_sparse_grad_L_x,_,_ = speedy_sparse_dynamic_loss(pair[0],pair[1], device) # NOT W
        #print(grad_L_x)
        #print(sparse_grad_L_x)
        #print(grad_L_x.numel() - torch.eq(grad_L_x, sparse_grad_L_x).sum(), grad_L_x.numel() - torch.eq(sparse_grad_L_x, speedy_sparse_grad_L_x).sum())
        print(torch.norm(grad_L_x - sparse_grad_L_x), torch.norm(grad_L_x - speedy_sparse_grad_L_x))

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    #test_exact()
    #test_diffable_dp()
    #test_sparse_dp()
    test_speedy_sparse_dp()
    #compare_thetas()
    #vary_gamma()
    #compare_losses()
    #compare_gradients()
