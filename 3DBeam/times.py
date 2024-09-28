import numpy as np 
import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set()

import matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['font.size'] = 18


N = np.asarray([5, 10, 15, 20, 25, 30, 40, 50])
DEM_times = np.asarray([421, 587, 439, 622, 493, 701, 839, 1151])/60
FEM_times = np.asarray([0.90, 2.13, 7.94, 27.29, 76.43, 228.62])/60

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(N[:-2], FEM_times)
ax1.plot(N[:-2], np.exp(N[:-2]/5.5)/60, alpha=0.8)
ax1.set_xlabel('$N$')
ax1.set_ylabel('Time [min]')
ax1.legend(['FEM', r'$e^{N/5.5}$'])

ax2.plot(N, DEM_times, 'tab:red')
ax2.plot(N, 400/60+N**1.7/60, alpha=0.8, c='tab:green')
ax2.set_xlabel('$N$')
ax2.set_ylabel('Time [min]')
ax2.legend(['DEM', r'400 + $N^{1.7}$'])
fig.tight_layout()
fig.savefig('figures/times.pdf')
plt.show()