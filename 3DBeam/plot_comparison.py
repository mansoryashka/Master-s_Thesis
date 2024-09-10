import numpy as np
import matplotlib.pyplot as plt
import torch

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from DEM import MultiLayerNet, L2norm3D, dev, write_vtk_v2, models_path, arrays_path
from EnergyModels import NeoHookeanEnergyModel
from DemBeam import DeepEnergyMethodBeam, lmbda, mu, L, H, D

if __name__ == '__main__':
    model = MultiLayerNet(3, *[50]*3, 3)
    energy = NeoHookeanEnergyModel(lmbda, mu)
    Dem_strain = DeepEnergyMethodBeam(model, energy)

    N_strain = 21
    x_strain = np.linspace(0, L, 4*N_strain + 2)[1:-1]
    y_strain = np.linspace(0, D, N_strain)
    z_strain = np.linspace(0, H, N_strain)

    X_ref, Y_ref, Z_ref = np.meshgrid(x_strain, y_strain, z_strain)
    c1 = np.logical_and(Y_ref==0.5, Z_ref==0.5)
    c2 = np.logical_and(np.logical_and(Y_ref==1, Z_ref==0.5), X_ref==x_strain[-1])

    fem_strain = np.load(arrays_path / 'u_strain.npy')
    Y_fem = Y_ref + fem_strain[1]
    y1_fem = Y_fem[c1]
    y2_fem = Y_fem[c2]


    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax2.set_xscale('log')

    # ax.plot(x_strain, y1_fem, label='FEM')
    ax2.scatter(4*30*30*30, y2_fem, label='FEM')
    # plt.show()
    linestyles = [(5, (10, 3)), (0, (5, 10)), 'dashed', 'dotted']
    for i, N in enumerate([30, 40, 50, 60]):
        model_path = models_path / 'run4' / f'model_lr0.5_nn50_nl3_N{N}_0'
        Dem_strain.model.load_state_dict(torch.load(model_path))
        U_pred = Dem_strain.evaluate_model(x_strain, y_strain, z_strain)
        # exit(np.asarray(U_pred).shape)
        X_cur = X_ref + U_pred[0]
        Y_cur = Y_ref + U_pred[1]
        Z_cur = Z_ref + U_pred[2]

        # x1_ref = X_ref[c1]
        # y1_ref = Y_ref[c1]
        # z1_ref = Z_ref[c1]

        # x1_cur = X_cur[c1]
        y1_cur = Y_cur[c1]
        # z1_cur = Z_cur[c1]

        # x2_ref = X_ref[c2]
        # y2_ref = Y_ref[c2]
        # z2_ref = Z_ref[c2]

        # x2_cur = X_cur[c2]
        y2_cur = Y_cur[c2]
        # z2_cur = Z_cur[c2]

        # ax.plot(x_strain, y1_cur, linestyle=linestyles[i], alpha=0.9, label=f'N = {N}', linewidth=0.5)
        ax1.semilogy(x_strain, np.abs(y1_fem - y1_cur), alpha=0.9, label=f'N = {N}', linewidth=1)
        ax2.scatter(4*N*N*N, y2_cur, c='tab:red')
    ax1.legend()
    ax2.legend(['FEM', 'DEM'])
    plt.show()
        # plot linjeendring
        # plot enring i toppunkt
        # hent tilsvarende resultater fra FEM