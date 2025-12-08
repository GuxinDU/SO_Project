import numpy as np
from scipy.stats import beta

class RandomGenerator:
    def __init__(self, x_range, y_range, mean):
        if not (x_range[0] < mean[0] < x_range[1]):
            raise ValueError("x_mean must be within x_range")
        if not (y_range[0] < mean[1] < y_range[1]):
            raise ValueError("y_mean must be within y_range")
        self.x_lb = x_range[0]
        self.x_ub = x_range[1]
        self.y_lb = y_range[0]
        self.y_ub = y_range[1]
        self.x_mean = mean[0]
        self.y_mean = mean[1]

        self.x_a, self.x_b = self.__set_1d_parameters(self.x_lb, self.x_ub, self.x_mean)
        self.y_a, self.y_b = self.__set_1d_parameters(self.y_lb, self.y_ub, self.y_mean)

    def __set_1d_parameters(self, lb, ub, mean):
        if lb == ub:
            return 0.0, 1.0  # Degenerate case

        p = (mean - lb) / (ub - lb)

        if p <= 0: 
            return 1.0, 100.0  # Concentrate at lower bound
        if p >= 1: 
            return 100.0, 1.0  # Concentrate at upper bound

        min_param = 2.0
        
        nu_min = max(min_param / p, min_param / (1 - p))
        nu = max(10.0, nu_min)
        
        a = p * nu
        b = (1 - p) * nu
        
        return a, b
    
    def sample(self):
        sample_norm_x = beta.rvs(self.x_a, self.x_b)
        sample_norm_y = beta.rvs(self.y_a, self.y_b)
        
        # Scale back
        # return self.x_lb + sample_norm_x * (self.x_ub - self.x_lb), self.y_lb + sample_norm_y * (self.y_ub - self.y_lb)
        sample_x = self.x_lb + sample_norm_x * (self.x_ub - self.x_lb)
        sample_y = self.y_lb + sample_norm_y * (self.y_ub - self.y_lb)
        return np.array([sample_x, sample_y])



    # def sample(self):
def random_sample(x_lb, x_ub, y_lb, y_ub, x_mean, y_mean):
    """
    Samples a point (x, y) in the box [(x_lb, y_lb), (x_ub, y_ub)]
    such that the expectation of the distribution is (x_mean, y_mean).
    
    Uses a Beta distribution scaled to the interval.
    """
    def sample_1d(lb, ub, mean):
        if not (lb <= mean <= ub):
            raise ValueError(f"Mean {mean} must be between {lb} and {ub}")
        
        if lb == ub:
            return lb
            
        # Normalize mean to [0, 1]
        p = (mean - lb) / (ub - lb)
        
        # Handle edge cases where mean is exactly on boundary
        if p <= 0: return lb
        if p >= 1: return ub
        
        # Choose alpha and beta such that mean is p
        # Mean = alpha / (alpha + beta)
        # We choose a concentration parameter nu = alpha + beta
        # A higher nu means less variance.
        # We choose nu such that alpha > 1 and beta > 1 to ensure a unimodal distribution
        # if possible, or at least a reasonable shape.
        
        # Ensure alpha, beta > 1 for unimodal distribution if possible
        # p * nu > 1 => nu > 1/p
        # (1-p) * nu > 1 => nu > 1/(1-p)
        
        min_param = 2.0 # Minimum value for alpha and beta to ensure unimodality (zero slope at ends)
        
        nu_min = max(min_param / p, min_param / (1 - p))
        nu = max(10.0, nu_min) # Use at least 10, or more if needed
        
        a = p * nu
        b = (1 - p) * nu
        
        # Sample from beta
        sample_norm = beta.rvs(a, b)
        
        # Scale back
        return lb + sample_norm * (ub - lb)

    x = sample_1d(x_lb, x_ub, x_mean)
    y = sample_1d(y_lb, y_ub, y_mean)
    
    return x, y
