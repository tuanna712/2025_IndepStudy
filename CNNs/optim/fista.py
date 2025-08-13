import torch
from torch.optim.optimizer import Optimizer

def soft_thresholding(x, threshold):
    """
    Applies the soft-thresholding operator. When g(x) = lambda ||x||_1,
    the proximal operator is defined as:
        threshold = t * lambda
        prox_{lambda * t}(x) = sign(x) * max(0, abs(x) - threshold)
    where lambda is the regularization parameter and t is the step size.
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

class FISTA(Optimizer):
    """
    Implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    
    FISTA solves problems of the form: min_x f(x) + g(x), where f(x) is a
    smooth convex function and g(x) is a nonsmooth convex function.
    
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): The learning rate, which corresponds to 1/L, where L is the
                    L-Lipschitz constant of the gradient of f.
        prox_op (callable): A function that acts as the proximal operator for g(x).
                            It should accept a tensor and the learning rate as input.
    """
    def __init__(self, params, lr=1e-3, prox_op=None):
        defaults = dict(lr=lr)
        super(FISTA, self).__init__(params, defaults)
        self.prox_op = prox_op

        # Initialize the state for each parameter group
        for group in self.param_groups:
            # initialize momentum term 'y' to the starting parameters (x_0)
            group['y'] = [p.clone().detach() for p in group['params']]
            # initialize iteration k = 1
            group['k'] = 1

    def step(self):
        """
        Performs a single optimization step of the FISTA algorithm.
        """
        # Iterate over all parameter groups
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            
            # Iterate over each parameter tensor in the group
            for i, p in enumerate(group['params']):
                # Skip if a parameter has no gradient
                if p.grad is None:
                    continue
                
                # Get the current parameter 'x_k' and the momentum term 'y_k'
                x_k = p.data
                y_k = group['y'][i].data

                # Gradient at the momentum point 'y_k'.
                grad_y = p.grad.data
                
                # Proximal gradient step: x_{k+1} = prox_tg(y_k - t*grad(f(y_k)))
                x_next = self.prox_op(y_k - lr * grad_y, lr)

                # Momentum-based acceleration step: y_{k+1} = x_{k+1} + (k / (k + 3)) * (x_{k+1} - x_k)
                momentum_coeff = k / (k + 3)
                y_next = x_next + momentum_coeff * (x_next - x_k)

                # Update the parameters and momentum term for the next iteration
                p.data = x_next
                group['y'][i].data = y_next
            
            # Increment the iteration counter for the next step
            group['k'] += 1
#====================================================================================
