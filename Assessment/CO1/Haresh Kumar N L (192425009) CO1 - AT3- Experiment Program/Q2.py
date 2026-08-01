import numpy as np
np.random.seed(10)
mu = 5
sigma = 2
data = np.random.normal(mu,sigma,1000)
mu_mle = np.mean(data)
var_mle = np.mean((data-mu_mle)**2)
print("Actual Mean :",mu)
print("Estimated Mean :",mu_mle)
print("Actual Variance :",sigma**2)
print("Estimated Variance :",var_mle)
