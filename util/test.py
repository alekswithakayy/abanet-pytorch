from indexing import PathIndex
import matplotlib.pyplot as plt
import numpy as np
size=(32,32)
ind = PathIndex(radius=4, size=size)

for path in ind.paths_by_len_indices:
    print(path.shape)

xs, ys = [], []
a = np.zeros((2,size[0],size[1]))
for x in range(size[0]):
    for y in range(size[1]):
        xs.append(x)
        ys.append(y)
        a[:,x,y] = [x,y]
a = a.reshape([2,-1])

center = a[:,ind.src_indices[0]]
sur_x = []
sur_y = []
for i in ind.dst_indices[:,0]:
    sur_x.append(a[:,i][0])
    sur_y.append(a[:,i][1])
plt.scatter(xs,ys, color='r')
plt.scatter(sur_x, sur_y, color='g')
plt.scatter(center[0],center[1], color='b')
plt.show()
