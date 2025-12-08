import numpy as np
from rvo2.random_generator import random_sample, RandomGenerator

# def test_distribution():
#     x_lb, x_ub = -0.5, 0.5
#     y_lb, y_ub = -0.5, 0.5
#     x_mean, y_mean = 0.1, -0.1
    
#     samples_x = []
#     samples_y = []
    
#     for _ in range(10000):
#         x, y = random_sample(x_lb, x_ub, y_lb, y_ub, x_mean, y_mean)
#         samples_x.append(x)
#         samples_y.append(y)
        
#     avg_x = np.mean(samples_x)
#     avg_y = np.mean(samples_y)
    
#     print(f"Target Mean: ({x_mean}, {y_mean})")
#     print(f"Sample Mean: ({avg_x:.2f}, {avg_y:.2f})")
    
#     assert abs(avg_x - x_mean) < 0.5
#     assert abs(avg_y - y_mean) < 0.5
#     print("Test Passed!")

if __name__ == "__main__":
    # test_distribution()
    np.random.seed(42)
    rg = RandomGenerator((-0.5, 0.5), (-0.5, 0.5), (0.1, -0.1))
    for i in range(10):
        sample = rg.sample()
        print(f"Sample {i+1}: {sample}")