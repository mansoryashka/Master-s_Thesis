from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, '..')
from DEM import L2norm3D

# arrays_path = Path('/stored_arrays')
# u_fem5 = np.load(arrays_path / 'u_fem_N=5.npy')
# u_fem10 = np.load(arrays_path / 'u_fem_N10.npy')
e = []

u_ref = np.load('stored_arrays/u_fem_N30.npy')
# L2_ref = L2norm3D(u_ref, 4*30, 30, 30, 1/120, 1/30, 1/30)

Ns = [5, 10, 15, 20, 25]
for N in Ns:
    u_fem = np.load(f'stored_arrays/u_fem_N{N}.npy')
    e_i = L2norm3D(u_fem - u_ref, 4*30, 30, 30, 1/120, 1/30, 1/30)
    # L2_i = L2norm3D(u_fem, 4*30, 30, 30, 1/120, 1/30, 1/30)
    # print(e_i)
    e.append(e_i)

for i in range(1, len(e)):
    e_new, e_old = e[i], e[i-1]
    h_new, h_old = 1/Ns[i], 1/Ns[i-1]
    q = np.log10(e_new/e_old)/np.log10(h_new/h_old)
    print(q)
    # print(np.log(e[i]/e[i-1]) / np.log((1/Ns[i]) / (1/Ns[i-1])))

