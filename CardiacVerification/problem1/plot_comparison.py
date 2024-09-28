import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import torch

import seaborn as sns
sns.set()

from matplotlib.patches import Rectangle

import sys
from problem1 import C, bf, bt, bfs
sys.path.insert(0, "../..")
from DEM import MultiLayerNet, write_vtk_v2, models_path, arrays_path
from EnergyModels import GuccioneTransverseEnergyModel
sys.path.insert(0, "../../3Dbeam")
from DemBeam import DeepEnergyMethodBeam

import matplotlib
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == '__main__':
    L = 10; H = 1; D = 1

    N = 50
    N_test = 10
    middle = int(np.floor(5*N_test))
    middle_z = int(np.floor(N_test))
    x_test = np.linspace(0, L, 10*N_test+1)
    y_test = np.linspace(0, H, N_test+1)
    z_test = np.linspace(0, D, N_test+1)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test)
    
    u_pred_fem = np.load('stored_arrays/u_fem.npy')
    X_fem, Y_fem, Z_fem = X + u_pred_fem[0], Y + u_pred_fem[1], Z + u_pred_fem[2]

    pts_x = np.zeros((10, 3))
    pts_y = np.zeros((10, 3))
    pts_z = np.zeros((10, 3))
    pts_x_fem = np.zeros((10, 3))
    pts_y_fem = np.zeros((10, 3))
    pts_z_fem = np.zeros((10, 3))
    for i in range(10):
        condition1 = np.logical_and(np.logical_and(Z==0.5, Y==0.5), X==i)
        condition2 = np.logical_and(np.logical_and(Z==0.5, Y==0.9), X==i)
        condition3 = np.logical_and(np.logical_and(Z==0.9, Y==0.5), X==i)
        pts_x[i] = X[condition1][0], Y[condition1][0], Z[condition1][0]
        pts_y[i] = X[condition2][0], Y[condition2][0], Z[condition2][0]
        pts_z[i] = X[condition3][0], Y[condition3][0], Z[condition3][0]
        pts_x_fem[i] = X_fem[condition1][0], Y_fem[condition1][0], Z_fem[condition1][0]
        pts_y_fem[i] = X_fem[condition2][0], Y_fem[condition2][0], Z_fem[condition2][0]
        pts_z_fem[i] = X_fem[condition3][0], Y_fem[condition3][0], Z_fem[condition3][0]

    condition4 = np.logical_and(Y==0.5, Z==0.5)
    condition5 = np.logical_and(np.logical_and(X==10, Y==0.5), Z==1)
    line_x, line_y, line_z = X[condition4], Y[condition4], Z[condition4]
    line_x_fem, line_z_fem = X_fem[condition4], Z_fem[condition4]
    end_x, end_y, end_z = X[condition5], Y[condition5], Z[condition5]

    fig1, ax = plt.subplots(1, 3, figsize=(13, 5))
    fig = plt.figure(figsize=(14, 8))
    # plt.style.use('seaborn-v0_8-darkgrid')
    ax11 = plt.subplot2grid((2,3), (0,0), colspan=3, rowspan=1)
    ax12 = plt.subplot2grid((2,3), (1,0))
    ax13 = plt.subplot2grid((2,3), (1,1))
    ax14 = plt.subplot2grid((2,3), (1,2))

    strain_x_fem = (np.linalg.norm(pts_x_fem[:-1] - pts_x_fem[1:], axis=1)
                / np.linalg.norm(pts_x[:-1] - pts_x[1:], axis=1)
                - 1) * 100
    strain_y_fem = (np.linalg.norm(pts_x_fem - pts_y_fem, axis=1)
                / np.linalg.norm(pts_x - pts_y, axis=1)
                - 1) * 100
    strain_z_fem = (np.linalg.norm(pts_x_fem - pts_z_fem, axis=1)
                / np.linalg.norm(pts_x - pts_z, axis=1)
                - 1) * 100

    p1 = 5; p2 = -5
    ax11.plot(line_x_fem, line_z_fem, label='FEM')
    ax12.plot(line_x_fem[:p1], line_z_fem[:p1])
    ax13.plot(line_x_fem[middle-2:middle+3], line_z_fem[middle-2:middle+3])
    ax14.plot(line_x_fem[p2:], line_z_fem[p2:])

    ax[0].plot(strain_x_fem, label='FEM')
    ax[1].plot(strain_y_fem)
    ax[2].plot(strain_z_fem)
    # U_pred = np.load(f'stored_arrays/U_pred{N}.npy')

    for lr, nn, nl, j in zip([0.1, 0.1, 0.05], [40, 50, 50], [5, 3, 3], [1, 2, 3]):
    # for lr, nn, nl, j in zip([0.1, 0.1, 0.5], [20, 30, 40], [4, 3, 3], [1, 2, 3]):
        model = MultiLayerNet(3, *[nn]*nl, 3)
        energy = GuccioneTransverseEnergyModel(C, bf, bt, bfs)
        DemBeam = DeepEnergyMethodBeam(model, energy)
        DemBeam.model.load_state_dict(torch.load(Path('trained_models') / f'model_lr{lr}_nn{nn}_nl{nl}_100'))
        U_pred = DemBeam.evaluate_model(x_test, y_test, z_test)
        X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]

        pts_x_cur = np.zeros((10, 3))
        pts_y_cur = np.zeros((10, 3))
        pts_z_cur = np.zeros((10, 3))

        for i in range(10):
            condition1 = np.logical_and(np.logical_and(Z==0.5, Y==0.5), X==i)
            condition2 = np.logical_and(np.logical_and(Z==0.5, Y==0.9), X==i)
            condition3 = np.logical_and(np.logical_and(Z==0.9, Y==0.5), X==i)

            pts_x_cur[i] = X_cur[condition1][0], Y_cur[condition1][0], Z_cur[condition1][0]
            pts_y_cur[i] = X_cur[condition2][0], Y_cur[condition2][0], Z_cur[condition2][0]
            pts_z_cur[i] = X_cur[condition3][0], Y_cur[condition3][0], Z_cur[condition3][0]

        line_x_cur, line_z_cur = X_cur[condition4], Z_cur[condition4]

        end_x_cur, end_y_cur, end_z_cur = X_cur[condition5], Y_cur[condition5], Z_cur[condition5]

        strain_x = (np.linalg.norm(pts_x_cur[:-1] - pts_x_cur[1:], axis=1)
                    / np.linalg.norm(pts_x[:-1] - pts_x[1:], axis=1)
                    - 1) * 100
        strain_y = (np.linalg.norm(pts_x_cur - pts_y_cur, axis=1)
                    / np.linalg.norm(pts_x - pts_y, axis=1)
                    - 1) * 100
        strain_z = (np.linalg.norm(pts_x_cur - pts_z_cur, axis=1)
                    / np.linalg.norm(pts_x - pts_z, axis=1)
                    - 1) * 100

        ax[0].plot(strain_x, '--x', alpha=0.8, label=rf'model{j}')
        ax[1].plot(strain_y, '--x', alpha=0.8, label=rf'model{j}')
        ax[2].plot(strain_z, '--x', alpha=0.8, label=rf'model{j}')

        ax11.plot(line_x_cur, line_z_cur,
                 linestyle='--', # linewidth=1.0, 
                 alpha=0.8, label=rf'model{j}')
        ax12.plot(line_x_cur[:p1], line_z_cur[:p1],
                 linestyle='--', # linewidth=1.0, 
                 alpha=0.8, label=rf'$\eta$={lr}')
        ax13.plot(line_x_cur[middle-3:middle+3], line_z_cur[middle-3:middle+3],
                 linestyle='--', # linewidth=1.0, 
                 alpha=0.8, label=rf'$\eta$={lr}')
        ax14.plot(line_x_cur[p2-3:], line_z_cur[p2-3:],
                 linestyle='--', # linewidth=1.0, 
                 alpha=0.8, label=rf'$\eta$={lr}')
        
    ax[0].set_ylabel('strain [%]')
    ax[0].set_title('$x$-axis')
    ax[0].set_xticks(np.arange(9))
    ax[0].set_xticklabels(['p1', '', 'p3', '', 'p5', '', 'p7', '', 'p9'])
    ax[1].set_title('$y$-axis')
    ax[1].set_xticks(np.arange(10))
    ax[1].set_xticklabels(['p1', '', 'p3', '', 'p5', '', 'p7', '', 'p9', ''])
    ax[2].set_title('$z$-axis')
    ax[2].set_xticks(np.arange(10)) 
    ax[2].set_xticklabels(['p1', '', 'p3', '', 'p5', '', 'p7', '', 'p9', ''])
    ax[0].legend()

    y1_fem = line_z_fem
    x1_fem = line_x_fem

    ax11.set_xlabel('$x$ [mm]')
    ax11.set_ylabel('$z$-deflection [mm]')
    ax11.legend()

    ax12.set_xlabel('$x$ [mm]')
    ax12.set_ylabel('$z$-deflection [mm]')
    ax12.set_ylim(top=y1_fem[p1-1])
    ax12.set_xlim(right=x1_fem[p1-1])

    ticks1 = ax12.get_yticks()
    ax12.set_yticks([ticks1[0], ticks1[-1]])
    ax12.set_yticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])

    ax13.set_ylim((y1_fem[middle-8], y1_fem[middle+2]))
    ax13.set_xlim((x1_fem[middle-2], x1_fem[middle+2]))
    ax13.set_xlabel('$x$ [mm]')

    ticks1 = ax13.get_yticks()
    ax13.set_yticks([ticks1[0], ticks1[-1]])
    ax13.set_yticklabels([f'{x:.3f}' for x in [ticks1[0], ticks1[-1]]])

    ax14.set_ylim(bottom=y1_fem[p2-14])
    ax14.set_xlim(left=x1_fem[p2])
    ax14.set_xlabel('$x$ [mm]')
    
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
                             linestyle='--', linewidth=5, alpha=0.5
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
                             linestyle='--', linewidth=5, alpha=0.5
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
                             linestyle='--', linewidth=5, alpha=0.5
                             ))

    fig1.savefig(f'figures/strain_plot_100.pdf')
    fig.savefig(f'figures/line_plot_100.pdf')
        # ax11.set_xlabel('$x$ [mm]')
        # ax11.set_ylabel('$z$ [mm]')
    plt.show()