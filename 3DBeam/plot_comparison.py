import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import seaborn as sns
sns.set()

from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, "..")
from DEM import MultiLayerNet, L2norm3D, dev, write_vtk_v2, models_path, arrays_path
from EnergyModels import NeoHookeanEnergyModel
from DemBeam import DeepEnergyMethodBeam, lmbda, mu, L, H, D

import matplotlib
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == '__main__':
    energy = NeoHookeanEnergyModel(lmbda, mu)

    N_strain = 21
    middle = int(np.floor(2*N_strain))
    x_strain = np.linspace(0, L, 4*N_strain + 2)[1:-1]
    y_strain = np.linspace(0, D, N_strain)
    z_strain = np.linspace(0, H, N_strain)

    X_ref, Y_ref, Z_ref = np.meshgrid(x_strain, y_strain, z_strain)
    c1 = np.logical_and(Y_ref==0.5, Z_ref==0.5)
    c2 = np.logical_and(np.logical_and(Y_ref==1, Z_ref==0.5), X_ref==x_strain[-1])

    fem_strain = np.load(arrays_path / 'u_strain.npy')
    X_fem = X_ref + fem_strain[0]
    Y_fem = Y_ref + fem_strain[1]
    x1_fem = X_fem[c1]
    y1_fem = Y_fem[c1]
    y2_fem = Y_fem[c2]

    fig = plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    ax11 = plt.subplot2grid((2,3), (0,0), colspan=3, rowspan=1)
    ax12 = plt.subplot2grid((2,3), (1,0))
    ax13 = plt.subplot2grid((2,3), (1,1))
    ax14 = plt.subplot2grid((2,3), (1,2))

    p1 = 4; p2 = -4
    # fig2, ax2 = plt.subplots()
    ax11.plot(x1_fem, y1_fem, label='FEM')
    ax12.plot(x1_fem[:p1], y1_fem[:p1], label='FEM')
    ax13.plot(x1_fem[middle-2:middle+3], y1_fem[middle-2:middle+3], label='FEM')
    ax14.plot(x1_fem[p2:], y1_fem[p2:], label='FEM')

    colors = ['C1', 'C2', 'C3', 'yellow']

    for lr, nl, i in zip([0.5, 0.1, 0.05], [2, 3, 4], [1, 2, 3]):
        model = MultiLayerNet(3, *[40]*nl, 3)
        Dem_strain = DeepEnergyMethodBeam(model, energy)
        model_path = Path('trained_models') / f'model_lr{lr}_nl{nl}'
        Dem_strain.model.load_state_dict(torch.load(model_path))
        U_pred = Dem_strain.evaluate_model(x_strain, y_strain, z_strain)
        # exit(np.asarray(U_pred).shape)
        X_cur = X_ref + U_pred[0]
        Y_cur = Y_ref + U_pred[1]
        Z_cur = Z_ref + U_pred[2]

        # x1_ref = X_ref[c1]
        # y1_ref = Y_ref[c1]
        # z1_ref = Z_ref[c1]

        x_cur = X_cur[c1]
        y_cur = Y_cur[c1]
        # z_cur = Z_cur[c1]

        # x2_ref = X_ref[c2]
        # y2_ref = Y_ref[c2]
        # z2_ref = Z_ref[c2]

        # x2_cur = X_cur[c2]
        y2_cur = Y_cur[c2]

        ax11.plot(x_cur, y_cur, #c=colors[i],
                 linestyle='--', linewidth=1.0, 
                 alpha=0.9, label=f'model{i}')
        ax12.plot(x_cur[:p1], y_cur[:p1], #c=colors[i],
                 linestyle='--', linewidth=1.0, 
                 alpha=0.9, label=f'model{2}')
        ax13.plot(x_cur[middle-2:middle+3], y_cur[middle-2:middle+3], #c=colors[i],
                 linestyle='--', linewidth=1.0, 
                 alpha=0.9, label=f'model{3}')
        ax14.plot(x_cur[p2:], y_cur[p2:], #c=colors[i],
                 linestyle='--', linewidth=1.0, 
                 alpha=0.9, label=f'model{4}')
        i += 1

    ax11.set_xlabel('$x$ [m]')
    ax11.set_ylabel('$y$-deflection [m]')
    ax11.legend()

    ax12.set_xlabel('$x$ [m]')
    ax12.set_ylabel('$y$-deflection [m]')
    ax12.set_ylim(bottom=y1_fem[p1-1])
    ax12.set_xlim(right=x1_fem[p1-1])

    ticks1 = ax12.get_yticks()
    ax12.set_yticks([ticks1[0], ticks1[-1]])
    ax12.set_yticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])

    ax13.set_ylim((y1_fem[middle+2], y1_fem[middle-2]))
    ax13.set_xlim((x1_fem[middle-2], x1_fem[middle+2]))
    ax13.set_xlabel('$x$ [m]')

    ticks1 = ax13.get_yticks()
    ax13.set_yticks([ticks1[0], ticks1[-1]])
    ax13.set_yticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])

    ax14.set_ylim(top=y1_fem[p2])
    ax14.set_xlim(left=x1_fem[p2])
    ax14.set_xlabel('$x$ [m]')
    
    ticks1 = ax14.get_yticks()
    ax14.set_yticks([ticks1[0], ticks1[-1]])
    ax14.set_yticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])

    patch1_limx = ax12.get_xlim()
    patch1_limy = ax12.get_ylim()

    patch2_limx = ax13.get_xlim()
    patch2_limy = ax13.get_ylim()

    patch3_limx = ax14.get_xlim()
    patch3_limy = ax14.get_ylim()

    ax11.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='navy',
                             linestyle='--'
                             ))
    ax12.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='navy',
                             linestyle='--', linewidth=5, alpha=0.7
                             ))
    
    ax11.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='forestgreen',
                             linestyle='--'
                             ))
    ax13.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='forestgreen',
                             linestyle='--', linewidth=5, alpha=0.7
                             ))

    ax11.add_patch(Rectangle([patch3_limx[0], patch3_limy[0]], 
                             patch3_limx[1] - patch3_limx[0], 
                             patch3_limy[1] - patch3_limy[0],
                             facecolor='None', edgecolor='firebrick',
                             linestyle='--'
                             ))
    ax14.add_patch(Rectangle([patch3_limx[0], patch3_limy[0]], 
                             patch3_limx[1] - patch3_limx[0], 
                             patch3_limy[1] - patch3_limy[0],
                             facecolor='None', edgecolor='firebrick',
                             linestyle='--', linewidth=5, alpha=0.7
                             ))
    fig.savefig('figures/line_plot.pdf')
    plt.show()
    # plot linjeendring
    # plot enring i toppunkt
    # hent tilsvarende resultater fra FEM