import numpy as np
a = np.arange(540).reshape(2, 1, 90, 3)
print(a.shape)
b = a.transpose((0, 3, 2, 1))
print(b.shape)
