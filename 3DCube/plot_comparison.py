import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from matplotlib.patches import Rectangle


import sys
sys.path.insert(0, "..")
from DEM import MultiLayerNet, L2norm3D, dev, write_vtk_v2, models_path, arrays_path
from EnergyModels import NeoHookeanActiveEnergyModel
from DemCube import DeepEnergyMethodCube, mu, L, H, D

import seaborn as sns
sns.set()

import matplotlib
matplotlib.rcParams['figure.dpi'] = 150

models_path = Path('trained_models') / 'run1'
arrays_path = Path('stored_arrays')

if __name__ == '__main__':
    fem_strain = np.load(arrays_path / 'u_strain_z.npy')
    N_strain = 21
    middle = int(np.floor(N_strain/2))
    z_strain = np.linspace(0, L, N_strain + 2)[1:-1]
    y_strain = np.linspace(0, D, N_strain)
    x_strain = np.linspace(0, H, N_strain)

    X_ref, Y_ref, Z_ref = np.meshgrid(x_strain, y_strain, z_strain)
    # c1 = np.logical_and(Y_ref==0.5, Z_ref==0.5)
    c1 = np.logical_and(Y_ref==0.5, X_ref==0.5)

    # fig = plt.figure(figsize=(14, 8))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_ref, Y_ref, Z_ref, alpha=0.1)
    # ax.scatter(X_ref[c1], Y_ref[c1], Z_ref[c1], 'red')
    
    X_cur = X_ref + fem_strain[0]
    Y_cur = Y_ref + fem_strain[1]
    Z_cur = Z_ref + fem_strain[2]

    x_fem = X_cur[c1]; y_fem = Y_cur[:, -1, 0]
    # fig = plt.figure(figsize=(14, 8))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_cur, Y_cur, Z_cur, alpha=0.1)
    # ax.scatter(X_cur[c1], Y_cur[c1], Z_cur[c1], 'red')

    # plt.show(); exit()
    fig = plt.figure(figsize=(10, 10))
    # plt.style.use('seaborn-v0_8-darkgrid')
    ax11 = plt.subplot2grid((3,2), (0,0), colspan=1, rowspan=3)
    ax12 = plt.subplot2grid((3,2), (0,1))
    ax13 = plt.subplot2grid((3,2), (1,1))
    ax14 = plt.subplot2grid((3,2), (2,1))

    p1 = -3; p2 = 3
    ax11.plot(x_fem, y_fem, label='FEM')
    ax12.plot(x_fem[p1:], y_fem[p1:])
    ax13.plot(x_fem[middle-2:middle+3], y_fem[middle-2:middle+3])
    ax14.plot(x_fem[:p2], y_fem[:p2])

    for nn in [20, 30, 40, 50]:
        model = MultiLayerNet(3, *[nn]*3, 3)
        energy = NeoHookeanActiveEnergyModel(mu)
        Dem_strain = DeepEnergyMethodCube(model, energy)

        model_path = models_path / f'model_lr0.1_nn{nn}_nl3_N40_0'
        Dem_strain.model.load_state_dict(torch.load(model_path))
        U_pred = Dem_strain.evaluate_model(x_strain, y_strain, z_strain)

        x = X_ref + U_pred[0]
        y = Y_ref + U_pred[1]
        z = Z_ref + U_pred[2]
        x = x[c1]; y = y[:, -1, 0]

        ax11.plot(x, y,
                 linestyle='--', linewidth=0.8, 
                 alpha=0.9, label=f'N = {nn}')
        ax12.plot(x[p1:], y[p1:],
                 linestyle='--', linewidth=0.8, 
                 alpha=0.9, label=f'N = {nn}')
        ax13.plot(x[middle-2:middle+3], y[middle-2:middle+3],
                 linestyle='--', linewidth=0.8, 
                 alpha=0.9, label=f'N = {nn}')
        ax14.plot(x[:p2], y[:p2],
                 linestyle='--', linewidth=0.8, 
                 alpha=0.9, label=f'# neurons = {nn}')

    ax11.set_xlabel('$x$-deflection [m]')
    ax11.set_ylabel('$y$ [m]')
    ax11.legend()

    # ax12.set_xlabel('$x$-deflection [m]')
    ax12.set_ylabel('$y$ [m]')
    ax12.set_ylim(bottom=y_fem[p1+1])
    ax12.set_xlim(left=x_fem[p1+1])

    ax13.set_ylim((y_fem[middle-2], y_fem[middle+2]))
    ax13.set_xlim((x_fem[middle]*0.999, x_fem[middle-2]))
    # ax13.set_xlabel('$x$-deflection [m]')
    ax13.set_ylabel('$y$ [m]')

    ax14.set_ylim(top=y_fem[p2-1])
    ax14.set_xlim(left=x_fem[p2-1])
    ax14.set_xlabel('$x$-deflection [m]')
    ax14.set_ylabel('$y$ [m]')

    ticks1 = ax12.get_xticks()
    ax12.set_xticks([ticks1[0], ticks1[-1]])
    ax12.set_xticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])
    # ticks2 = ax12.get_yticks()
    # ax12.set_yticks([ticks2[0], ticks2[-1]])
    # ax12.set_yticklabels([f'{y:.3f}' for y in [ticks2[0], ticks2[-1]]])
    patch1_limx = ax12.get_xlim()
    patch1_limy = ax12.get_ylim()
    # ax12.set_xticks(patch1_limx)
    # ax12.set_yticks(patch1_limy)
    # ax12.set_xticklabels([f'{x:.3f}' for x in patch1_limx])
    # ax12.set_yticklabels([f'{y:.3f}' for y in patch1_limy])

    ticks1 = ax13.get_xticks()
    ax13.set_xticks([ticks1[0], ticks1[-1]])
    ax13.set_xticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])
    # ticks2 = ax13.get_yticks()
    # print(ticks2)
    # ax13.set_yticks([ticks2[0], ticks2[-1]])
    # ax13.set_xticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])
    # ax13.set_yticklabels([f'{y:.3f}' for y in [ticks2[0], ticks2[-1]]])
    patch2_limx = ax13.get_xlim()
    patch2_limy = ax13.get_ylim()
    # ax13.set_xticks(patch2_limx)
    # ax13.set_yticks(patch2_limy)
    # ax13.set_xticklabels([f'{x:.3f}' for x in patch2_limx])
    # ax13.set_yticklabels([f'{y:.3f}' for y in patch2_limy])

    ticks1 = ax14.get_xticks()
    ax14.set_xticks([ticks1[0], ticks1[-1]])
    ax14.set_xticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])
    patch3_limx = ax14.get_xlim()
    patch3_limy = ax14.get_ylim()
    # ax14.set_xticks(patch3_limx)
    # ax14.set_yticks(patch3_limy)
    # ax14.set_xticklabels([f'{x:5.3f}' for x in patch3_limx])
    # ax14.set_yticklabels([f'{y:5.3f}' for y in patch3_limy])

    ax11.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='tab:blue',
                             linestyle='--'
                             ))
    ax12.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='tab:blue',
                             linestyle='--', linewidth=3
                             ))
    
    ax11.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='tab:green',
                             linestyle='--'
                             ))
    ax13.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='tab:green',
                             linestyle='--', linewidth=3
                             ))

    ax11.add_patch(Rectangle([patch3_limx[0], patch3_limy[0]], 
                             patch3_limx[1] - patch3_limx[0], 
                             patch3_limy[1] - patch3_limy[0],
                             facecolor='None', edgecolor='tab:red',
                             linestyle='--'
                             ))
    ax14.add_patch(Rectangle([patch3_limx[0], patch3_limy[0]], 
                             patch3_limx[1] - patch3_limx[0], 
                             patch3_limy[1] - patch3_limy[0],
                             facecolor='None', edgecolor='tab:red',
                             linestyle='--', linewidth=3
                             ))

    fig.tight_layout()
    plt.show()