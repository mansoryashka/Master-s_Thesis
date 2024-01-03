import numpy as np
import matplotlib.pyplot as plt

u_dem = np.load('u_dem.npy')
u_fem = np.load('u_fem.npy')
# print(u_fem.shape); exit()
print(u_dem.shape)

plt.figure()
plt.plot(u_fem[0, 49, 12])
plt.plot(u_dem[0, 24, 99, 0::2])
plt.figure()
plt.plot(u_fem[1, 49, 12])
plt.plot(u_dem[1, 24, 99,0::2])
plt.figure()
plt.plot(u_fem[2, 49, 12])
plt.plot(u_dem[2, 24, 99, 0::2])
plt.show()