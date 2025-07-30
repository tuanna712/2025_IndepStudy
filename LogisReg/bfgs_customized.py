
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
criterion = nn.MSELoss()
data = torch.randn(1, 10)
target = torch.randn(1, 1)

class LBFGS:
    def __init__(self, 
                 params, 
                 history_size=10, 
                 tolerance=1e-5,
                 ):
        self.params = list(params) # Parameters to optimize
        self.history_size = history_size
        self.s_history = [] # History of s_k = x_{k+1} - x_k
        self.y_history = [] # History of y_k = grad_{k+1} - grad_k
        self.prev_params_flat = None # To store params from previous step
        self.prev_grad_flat = None   # To store gradient from previous step
        self.tolerance = tolerance # For line search and convergence checks

        # Initialize current parameters and gradients for the first step
        # These will be updated by the first closure call
        self.current_loss = None
        self.current_params_flat = self._get_flat_params()
        self.current_grad_flat = None # Will be set after first closure call

    def _get_flat_params(self):
        # Helper to get all parameters as a single flattened tensor
        return torch.cat([p.data.view(-1) for p in self.params])

    def _set_flat_params(self, flat_params):
        # Helper to set parameters from a flattened tensor
        offset = 0
        for p in self.params:
            num_elements = p.numel()
            p.data.copy_(flat_params[offset:offset + num_elements].view(p.size()))
            offset += num_elements

    def _get_flat_grad(self):
        # Helper to get all gradients as a single flattened tensor
        # Ensure all gradients are computed
        return torch.cat([p.grad.data.view(-1) for p in self.params])

    # --- THE CORE L-BFGS STEP ---
    def step(self, closure):
        # 0. Initial evaluation and gradient computation (if first step)
        #    or to update prev_grad_flat for current step's history
        loss = closure() # Call the closure to get current loss and populate .grad
        current_grad_flat = self._get_flat_grad()

        # For the first step, we don't have previous (s, y) to compute direction
        if self.prev_params_flat is None:
            self.prev_params_flat = self._get_flat_params().clone()
            self.prev_grad_flat = current_grad_flat.clone()
            # For the very first step, we might just do a steepest descent step
            # or rely on a heuristic to get the first (s, y) pair.
            # A common approach is to just take a small step in the negative gradient direction
            # and then start the L-BFGS updates from the next iteration.
            # For simplicity here, we'll assume we need one iteration to establish history.
            # A full L-BFGS implementation would have a fallback for the first step.
            # For this conceptual example, let's just use the current loss as the return.
            self.current_loss = loss.item()
            return loss.item()

        # 1. Update History (s_k, y_k)
        # s_k = x_k - x_{k-1}
        # y_k = grad_k - grad_{k-1}
        s_k = self._get_flat_params() - self.prev_params_flat
        y_k = current_grad_flat - self.prev_grad_flat

        # Add to history and prune if over history_size
        self.s_history.append(s_k)
        self.y_history.append(y_k)
        if len(self.s_history) > self.history_size:
            self.s_history.pop(0) # Remove oldest
            self.y_history.pop(0)

        # 2. Compute Search Direction (Two-Loop Recursion)
        # This is the magic of L-BFGS!
        # It approximates H_k_inv * grad_k implicitly.
        q = current_grad_flat.clone() # Initial vector for the first loop

        alpha = [0.0] * len(self.s_history)
        # First loop (forward pass)
        for i in reversed(range(len(self.s_history))):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (y_i.dot(s_i) + 1e-10) # Add small epsilon for stability
            alpha[i] = rho_i * s_i.dot(q)
            q.add_(-alpha[i] * y_i)

        # Initial approximation of inverse Hessian (H_k_0)
        # Often chosen as gamma_k * I, where gamma_k is (s_m^T y_m) / (y_m^T y_m)
        # from the most recent pair.
        if len(self.s_history) > 0:
            s_m_last = self.s_history[-1]
            y_m_last = self.y_history[-1]
            gamma_k = (s_m_last.dot(y_m_last)) / (y_m_last.dot(y_m_last) + 1e-10)
        else:
            gamma_k = 1.0 # Default if no history yet

        r = gamma_k * q # Apply initial Hessian approximation

        # Second loop (backward pass)
        for i in range(len(self.s_history)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (y_i.dot(s_i) + 1e-10)
            beta_i = rho_i * y_i.dot(r)
            r.add_(s_i * (alpha[i] - beta_i))

        # The search direction is -r
        search_direction = -r

        # 3. Line Search
        # This is where the closure is called multiple times!
        # PyTorch uses a sophisticated line search (e.g., Wolfe conditions).
        # A simplified Armijo line search for demonstration:
        current_params = self._get_flat_params().clone()
        current_loss_val = loss.item() # Get the scalar loss value
        current_grad_dot_direction = current_grad_flat.dot(search_direction)

        alpha_step = 1.0 # Initial step size
        c1 = 1e-4 # Armijo condition constant

        while True:
            # Propose new parameters
            next_params = current_params + alpha_step * search_direction
            self._set_flat_params(next_params)

            # Evaluate loss at new parameters using the closure
            # This is why closure is passed: it re-evaluates the function
            loss_at_next_params = closure().item()

            # Check Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad_f(x)^T * d
            if loss_at_next_params <= current_loss_val + c1 * alpha_step * current_grad_dot_direction:
                # Condition met, accept this step size
                break
            else:
                # Condition not met, reduce step size
                alpha_step *= 0.5 # Or use a more sophisticated backtracking method
                if alpha_step < self.tolerance: # Prevent infinite loop
                    # If step size becomes too small, just take the last good step
                    # or revert to previous parameters and break.
                    # A robust line search handles this better.
                    self._set_flat_params(current_params) # Revert to previous
                    loss_at_next_params = current_loss_val # Maintain original loss
                    break

        # 4. Update for next iteration
        # The parameters have already been updated by _set_flat_params in the line search.
        # Now, store current parameters and gradients for the next iteration's history
        self.prev_params_flat = self._get_flat_params().clone()
        # The gradient at the *new* parameters would have been computed inside the last closure call
        # in the line search, but let's make sure it's fresh for the next iteration's y_k calculation.
        # Call closure one last time after final step to ensure gradients are up-to-date for prev_grad_flat.
        final_loss = closure() # Re-evaluate at final params to get final loss and grad
        self.prev_grad_flat = self._get_flat_grad().clone()

        self.current_loss = final_loss.item()
        return final_loss.item()


# --- 2. How you would use MyLBFGS ---
print("--- Starting Conceptual L-BFGS Training ---")
my_optimizer = LBFGS(model.parameters(), history_size=10)

# Simulate a training loop
for i in range(5): # Run for a few conceptual steps
    # For a real dataset, you'd iterate through train_loader batches
    # Here, we just use the same dummy data for simplicity
    def closure():
        # Ensure gradients are zeroed for this evaluation
        # This is essential because the line search re-evaluates
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        return loss

    # Perform the optimization step
    step_loss = my_optimizer.step(closure)
    print(f'Conceptual L-BFGS Step {i} \tLoss: {step_loss:.6f}')

# Note: The output will likely not converge well with dummy data and a simplified line search.
# This is purely illustrative of the *internal mechanics*.
print("--- Conceptual L-BFGS Training Finished ---")

# --- Original PyTorch L-BFGS usage (for comparison) ---
model = SimpleModel()
pytorch_optimizer = optim.LBFGS(model.parameters(), lr=1) # PyTorch LBFGS takes lr, but it's for initial step size if no history

print("\n--- Starting PyTorch Built-in L-BFGS Training ---")
for i in range(5):
    def closure_pytorch():
        pytorch_optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        return loss

    loss_pytorch = pytorch_optimizer.step(closure_pytorch)
    print(f'PyTorch Built-in L-BFGS Step {i} \tLoss: {loss_pytorch.item():.6f}')
print("--- PyTorch Built-in L-BFGS Training Finished ---")