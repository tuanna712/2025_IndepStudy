import torch
from torch.optim.optimizer import Optimizer, required

class OneStepSGD(Optimizer):
    def __init__(self, params, 
                 lr=required, 
                 momentum=0, 
                 dampening=0,
                 weight_decay=0, 
                 nesterov=False, 
                 ):
        # Validate hyperparameters
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # Define default hyperparameters for all parameter groups
        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        dampening=dampening,
                        weight_decay=weight_decay, 
                        nesterov=nesterov,
                        )

        # Call the parent Optimizer's constructor
        super(OneStepSGD, self).__init__(params, defaults)

    @torch.no_grad() # Crucial: we don't want to compute gradients for the optimizer's internal operations
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. This is optional and typically used by
                optimizers like LBFGS. For SGD, you usually call loss.backward()
                before optimizer.step().
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over all parameter groups (useful for different LRs for different layers)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad # Current gradient for the parameter

                # Standard SGD with weight decay
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay) # d_p = d_p + weight_decay * p

                # Handle momentum
                if momentum != 0:
                    param_state = self.state[p] # Optimizer state for this parameter
                    if 'momentum_buffer' not in param_state:
                        # Initialize momentum buffer if it doesn't exist
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # Update momentum buffer: buf = momentum * buf + (1 - dampening) * d_p
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        # Nesterov momentum: p_next = p - lr * (d_p + momentum * buf_prev)
                        # Instead of just d_p, use d_p + momentum * buf
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        # Standard momentum: p_next = p - lr * buf
                        d_p = buf

                # Update the parameter: p = p - lr * d_p
                p.add_(d_p, alpha=-lr) # In-place update

        return loss