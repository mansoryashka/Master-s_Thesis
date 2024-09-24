import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../..")
from problem2 import C, bf, bt, bfs, n_cond, define_domain, DeepEnergyMethodLV
from DEM import MultiLayerNet, write_vtk_LV
from EnergyModels import *

import seaborn as sns 
sns.set()

from matplotlib.patches import Rectangle
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

if __name__ == '__main__':
    N_test = 21; M_test = 5
    test_domain, _, _ = define_domain(N_test, M_test, n_cond=n_cond)
    test_domain = test_domain.reshape((N_test, M_test, N_test, 3))
    x_test = np.ascontiguousarray(test_domain[..., 0])
    y_test = np.ascontiguousarray(test_domain[..., 1])
    z_test = np.ascontiguousarray(test_domain[..., 2])
    X = np.copy(x_test)
    Y = np.copy(y_test)
    Z = np.copy(z_test)

    # get indexes of line to be plotted
    k = int((N_test - 1) / 2)
    middle_test = int(np.floor(M_test / 2))

    # plt.style.use('seaborn-v0_8-darkgrid')
    fig2, ax = plt.subplots()
    fig = plt.figure(figsize=(14, 10))
    # plt.style.use('seaborn-v0_8-darkgrid')
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,1))
    # for N, M in zip([30, 40, 40, 50, 60, 60, 80], [3, 3, 5, 5, 5, 9, 9]):

    ref_x = X[k, middle_test]
    ref_z = Z[k, middle_test]
    
    U_fem = np.load('stored_arrays/u_fem.npy')
    write_vtk_LV('output/FEM_solution', x_test, y_test, z_test, (U_fem[1], U_fem[2], U_fem[0]))
    # exit()
    X_fem, Y_fem, Z_fem = X + U_fem[1], Y + U_fem[2], Z + U_fem[0]
    fem_x, fem_z = X_fem[k, middle_test], Z_fem[k, middle_test]
    ax1.plot(fem_x, fem_z, label='FEM')
    ax1.plot(ref_x, ref_z, c='gray', linestyle=':')
    ax2.plot(fem_x[-7:-3], fem_z[-7:-3])
    ax3.plot(fem_x[:6], fem_z[:6])

    N = 100; M = 9
#     for lr, nn, nl, i in zip([0.1, 0.1, 0.05], [40, 50, 50], [5, 3, 3], [1, 2, 3]):

    for lr, nn, nl, i in zip([0.1, 0.1, 0.5], [20, 30, 40], [4, 3, 3], [1, 2, 3]):
        model = MultiLayerNet(3, *[nn]*nl, 3)
        # energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E3)
        energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E3)
        DemLV = DeepEnergyMethodLV(model, energy)
        DemLV.model.load_state_dict(torch.load(f'trained_models/model_lr{lr}_nn{nn}_nl{nl}_100'))
        U_pred = DemLV.evaluate_model(x_test, y_test, z_test)

        # np.save(f'stored_arrays/DemLV{N}x{M}', np.asarray(U_pred))
        # U_pred = np.load(f'stored_arrays/DemLV{N}x{M}.npy')

        X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]

        # get line in reference configuration

        # get line in curent configuration
        cur_x = X_cur[k, middle_test]
        cur_z = Z_cur[k, middle_test]

        
        ax1.plot(cur_x, cur_z, 
                label=f"model{i}", alpha=0.8,
                linestyle='--')
        # ax1.set_xticks([-10, -5, 0])

        
        ax2.plot(cur_x[-7:-3], cur_z[-7:-3], 
                label=f"model{i}", alpha=0.8,
                linestyle='--')
        # ax2.set_xticks([-12, -10])

        ax3.plot(cur_x[:5], cur_z[:5], 
                label=f"model{i}", alpha=0.8,
                linestyle='--')
        # ax3.set_ylim((-34, -32))
        # ax3.set_ylim(top=-20)
        # ax3.set_xlim((-5, 0))

        # ax3.set_xticks([-13, -9])
        # ax3.set_yticks([-27, -23])
        # ax3.set_yticks([-27, -23)

        # plt.savefig(f'figures/p2_plot_all2.pdf')

        Z_cur[0, 0, 0], Z_cur[0, -1, 0]
        print(f'\nLocation of apex:\n - Endocardium: {Z_cur[0, 0, 0]:6.2f}mm\n - Epicardium: {Z_cur[0, -1, 0]:6.2f}mm')
        plt.style.use('seaborn-v0_8-darkgrid')
        ax.scatter(N*N*M, Z_cur[0, 0, 0], marker='x', c='tab:blue')
        ax.scatter(N*N*M, Z_cur[0, -1, 0], marker='x', c='tab:orange')
        ax.set_xscale('log')
        ax.legend(['Endocardial apex', 'Epicardial apex'])
        ax.set_xlabel('Nr. of points [N]')
        ax.set_ylabel('$z$-location of deformed apex')

#     fig.tight_layout()
    ax1.legend()
    ax1.set_xlabel('$x$ [mm]')
    ax1.set_ylabel('$y$ [mm]')
    ax2.set_xlabel('$x$ [mm]')
    ax2.set_ylabel('$y$ [mm]')
    ax2.set_ylim((-9, -2))
    ax2.set_yticks([-9, -2])

    ax3.set_xlabel('$x$ [mm]')
    ax3.set_ylabel('$y$ [mm]')
    ax3.set_ylim(top=-25.5)
    ax3.set_yticks([-25.5, -27.3])

    patch1_limx = ax2.get_xlim()
    patch1_limy = ax2.get_ylim()

    patch2_limx = ax3.get_xlim()
    patch2_limy = ax3.get_ylim()

    ax1.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='navy',
                             linestyle='--'
                             ))
    ax2.add_patch(Rectangle([patch1_limx[0], patch1_limy[0]], 
                             patch1_limx[1] - patch1_limx[0], 
                             patch1_limy[1] - patch1_limy[0],
                             facecolor='None', edgecolor='navy',
                             linestyle='--', linewidth=5, alpha=0.5
                             ))
    
    ax1.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='forestgreen',
                             linestyle='--'
                             ))
    ax3.add_patch(Rectangle([patch2_limx[0], patch2_limy[0]], 
                             patch2_limx[1] - patch2_limx[0], 
                             patch2_limy[1] - patch2_limy[0],
                             facecolor='None', edgecolor='forestgreen',
                             linestyle='--', linewidth=5, alpha=0.5
                             ))


    fig.savefig('figures/line_plot_100.pdf')
    plt.show()