import numpy as np
from sklearn.metrics import pairwise_distances
for d in [2,10,50,100,500]:
    X = np.random.rand(100,d)
    dist = pairwise_distances(X)
    print("Dimension:",d)
    print("Average Distance:",dist.mean())
