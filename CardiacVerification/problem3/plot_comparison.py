import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../..")
from problem3 import C, bf, bt, bfs, n_cond, define_domain, DeepEnergyMethodLV
from DEM import MultiLayerNet, write_vtk_LV
from EnergyModels import *

import seaborn as sns 
sns.set()

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
    fig = plt.figure()
    # plt.style.use('seaborn-v0_8-darkgrid')
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,1))
    # for N, M in zip([30, 40, 40, 50, 60, 60, 80], [3, 3, 5, 5, 5, 9, 9]):

    U_fem = np.load('stored_arrays/u_predFEM3.npy')
    write_vtk_LV('output/FEM_solution', x_test, y_test, z_test, (U_fem[1], U_fem[2], U_fem[0]))
    exit()
    X_fem, Y_fem, Z_fem = X + U_fem[1], Y + U_fem[2], Z + U_fem[0]
    fem_x, fem_z = X_fem[k, middle_test], Z_fem[k, middle_test]
    ax1.plot(fem_x, fem_z)
    for N, M in zip([100], [9]):
        model = MultiLayerNet(3, *[40]*4, 3)
        # energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E3)
        energy = GuccioneIncompressibleEnergyModel(C, bf, bt, bfs, kappa=1E3)
        DemLV = DeepEnergyMethodLV(model, energy)
        DemLV.model.load_state_dict(torch.load(f'trained_models/run1/model_{N}x{M}'))
        U_pred = DemLV.evaluate_model(x_test, y_test, z_test)

        # np.save(f'stored_arrays/DemLV{N}x{M}', np.asarray(U_pred))
        # U_pred = np.load(f'stored_arrays/DemLV{N}x{M}.npy')


        X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]

        # get line in reference configuration
        ref_x = X[k, middle_test]
        ref_z = Z[k, middle_test]
        # get line in curent configuration
        cur_x = X_cur[k, middle_test]
        cur_z = Z_cur[k, middle_test]

        ax1.set_xlabel('$x$ [mm]')
        ax1.set_ylabel('$y$ [mm]')
        ax1.plot(ref_x, ref_z, c='gray', linestyle=':')
        ax1.plot(cur_x, cur_z, label=f"({N}, {M}, {N})")
        # ax1.set_xticks([-10, -5, 0])
        ax1.legend()

        
        ax2.plot(cur_x, cur_z, label=f"({N}, {M}, {N})", alpha=0.5)
        ax2.set_xlabel('$x$ [mm]')
        ax2.set_ylabel('$y$ [mm]')
        ax2.set_ylim((-9, -2))
        ax2.set_yticks([-9, -2])
        ax2.set_xlim(right=-10)
        # ax2.set_xticks([-12, -10])

        ax3.plot(cur_x, cur_z, label=f"({N}, {M}, {N})", alpha=0.5)
        ax3.set_xlabel('$x$ [mm]')
        ax3.set_ylabel('$y$ [mm]')
        # ax3.set_ylim((-34, -32))
        ax3.set_ylim(top=-20)
        # ax3.set_xlim((-5, 0))

        # ax3.set_xticks([-13, -9])
        # ax3.set_yticks([-27, -23])
        # ax3.set_yticks([-27, -23)

        plt.tight_layout()
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
        # fig2.savefig('figures/p2_apex3.pdf')
    plt.show()