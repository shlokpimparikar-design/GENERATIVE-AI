import numpy as np

a = np.array([2,1,2,3,2,9])
b = np.array([3,4,2,4,5,5])

cosine = ((np.dot(a,b)) / (np.norm(a)* np.norm(b)))
print(cosine)