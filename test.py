import numpy as np


a = [1,2,3]

# if all([value > 0 for value in a]):
#     print('done')

b = np.asarray(a)
c = b.tolist()
d = tuple(c)

print(c, d)

s = np.random.binomial(1, 0.5, 20)
print(s)
