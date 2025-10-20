import numpy as np

l=['a', 'abc', 'wdff']
a=['FFFF', 'EFMKMKF']
l.extend(a)
l.sort()

b=[5,6,1,4,7,5,2]

c=(4,9,7,10)
c=tuple(sorted(c, reverse=True))

#print(c)
epsilon = np.zeros((5))
epsilon[::2] = 1
print(epsilon)
