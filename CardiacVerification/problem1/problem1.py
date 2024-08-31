import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

import sys
sys.path.insert(0, "../../3DBeam")
sys.path.insert(0, "../../")
from DemBeam import DeepEnergyMethodBeam, train_and_evaluate, MultiLayerNet, write_vtk_v2
from EnergyModels import *

current_path = Path.cwd()

L = 10
H = D = 1

x0 = 0; x1 = L
y0 = 0; y1 = H
z0 = 0; z1 = D

N_test = 20
dx = L / (10*N_test)
dy = H / (N_test)
dz = D / (N_test)

C = 2E3
bf = 8
bt = 2
bfs = 4

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = 0.0
n_cond = [0, 0, 4]

def define_domain(L, H, D, N=10):
    x = np.linspace(0, L, 10*N+1)
    y = np.linspace(0, H, N+1)
    z = np.linspace(0, D, N+1)

    # Xm, Ym, Zm = np.meshgrid(x, y, z)
    Xm, Zm, Ym = np.meshgrid(x, z, y)
    Xm = np.expand_dims(Xm.flatten(), 1)
    Ym = np.expand_dims(Ym.flatten(), 1)
    Zm = np.expand_dims(Zm.flatten(), 1)
    domain = np.concatenate((Xm, Ym, Zm), axis=-1)
    db_idx = np.where(Xm == d_boundary)[0]
    db_pts = domain[db_idx, :]
    db_vals = np.ones(np.shape(db_pts)) * d_cond

    nb_idx = np.where(Zm == n_boundary)[0]
    nb_pts = domain[nb_idx, :]
    nb_vals = np.ones(np.shape(nb_pts)) * n_cond

    db_pts_x, db_pts_y, db_pts_z = db_pts.T
    nb_pts_x, nb_pts_y, nb_pts_z = nb_pts.T

    if not Path(current_path / 'domain.png').exists():
        fig = plt.figure(figsize=(5, 3))
        fig.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(Xm, Ym, Zm, facecolor='tab:blue', s=0.5, alpha=0.1)
        ax.scatter(db_pts_x, db_pts_y, db_pts_z, facecolor='tab:green', s=0.5)
        ax.scatter(nb_pts_x, nb_pts_y, nb_pts_z, facecolor='tab:red', s=0.5)
        ax.set_box_aspect((6, 1, 1))
        # ax.set_aspect('equal')
        ax.set_xticks([0.0, 10.0])
        ax.set_yticks([0.0, 1.0])
        ax.set_zticks([0.0, 1.0])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=25, azim=-55)
        
        fig.savefig('domain.png')
        # plt.show(); exit()
        plt.close()

    dirichlet = {
        'coords': db_pts,
        'values': db_vals
    }

    neumann = {
        'coords': nb_pts,
        'values': nb_vals
    }

    return domain, dirichlet, neumann


if __name__ == '__main__':
    N = 20
    LHD = [L, H, D]
    shape = [10*N+1, N+1, N+1]
    
    # x_test = np.linspace(0, L, 10*N)
    # y_test = np.linspace(0, H, N+3)[1:-2]
    # z_test = np.linspace(0, D, N+3)[1:-2]
    x_test = np.linspace(0, L, 10*N+1)
    y_test = np.linspace(0, H, N+1)
    z_test = np.linspace(0, D, N+1)

    domain, dirichlet, neumann = define_domain(L, H, D, N=N)

    dX = np.zeros(domain.shape[0])
    dX[1:] = np.cumsum(domain[1:, 0] - domain[:-1, 0])
    dX = dX.reshape((shape[1], shape[0], shape[2]))[0, :, 0]

    dY = np.zeros(domain.shape[0])
    dY[1:] = np.cumsum(domain[1:, 1] - domain[:-1, 1])
    dY = dY.reshape((shape[1], shape[0], shape[2]))[0, 0, :]

    dZ = np.zeros(domain.shape[0])
    dZ[1:] = np.cumsum(domain[1:, 2] - domain[:-1, 2])
    dZ = dZ.reshape((shape[1], shape[0], shape[2]))[:, 0, 0]
    # exit(dY)

    # dX[:, 1:, 0] /= np.sqrt((x_test[1]-x_test[0])**2 + y_test[-1]**2)
    # dX[:, 1:, 0] *= (x_test[1] - x_test[0])
    # dX[:, 0, 0] /= np.sqrt(y_test[-1]**2 + (z_test[1]-z_test[0])**2)
    # dX[:, 0, 0] *= (z_test[1] - z_test[0])
    # dX = np.cumsum(dX).flatten()

    neumann_domain = neumann['coords']

    dX_neumann = np.zeros(neumann_domain.shape[0])
    dX_neumann[1:] = np.cumsum(neumann_domain[1:, 0] - neumann_domain[:-1, 0])
    dX_neumann = dX_neumann.reshape((shape[0], shape[2]))[:, 0]
    
    # dY_neumann = np.zeros(neumann['coords'].shape[0])
    # dY_neumann[1:] = np.cumsum(np.sqrt((
    #       neumann_domain[1:, 1] - neumann_domain[:-1, 1])**2))
    
    dZ_neumann = np.zeros(neumann['coords'].shape[0])
    dZ_neumann[1:] = np.cumsum(neumann_domain[1:, 1] - neumann_domain[:-1, 1])
    dZ_neumann = dZ_neumann.reshape((shape[0], shape[1]))[0, :]
    
    # exit(dZ_neumann)

    # dX_neumann = dX_neumann.reshape(shape[:-1])
    # dX_neumann[1:, 0] /= np.sqrt((x_test[1]-x_test[0])**2 + y_test[-1]**2)
    # dX_neumann[1:, 0] *= (x_test[1] - x_test[0])
    # dX_neumann = np.cumsum(dX_neumann).flatten()
    # exit(dX.reshape(shape))

    model = MultiLayerNet(3, *[60]*6, 3)
    # energy = GuccioneTransverseEnergyModel(C, bf, bt, bfs)
    # energy = GuccioneTransverseActiveEnergyModel(C, bf, bt, bfs, kappa=1E5, Ta=15E3)
    energy = GuccioneEnergyModel(C, bf, bt, bfs)
    DemBeam = DeepEnergyMethodBeam(model, energy)
    # DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[0, 1], lr=0.1, epochs=15, fb=np.array([[0, 0, 0]]))
    DemBeam.train_model(domain, dirichlet, neumann, shape,dxdydz=[[dX, dY, dZ], [dX_neumann, dZ_neumann]], LHD=LHD, neu_axis=[0, 1], lr=0.5, epochs=15, fb=np.array([[0, 0, 0]]))
    U_pred = DemBeam.evaluate_model(x_test, y_test, z_test)

    # np.save('stored_arrays/U_predXZY', np.asarray(U_pred))
    write_vtk_v2('output/problem1', x_test, y_test, z_test, U_pred)

    # u_pred = np.load('stored_arrays/U_predXZY.npy')
    
    X, Y, Z = np.meshgrid(x_test, y_test, z_test)
    # print(np.array(u_pred).shape)
    # u_pred = np.transpose(np.array(u_pred), [1, 2, 3, 0])
    # print(u_pred.shape)
    X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]
    # print(X_cur[:, -1])

    pts_x1 = np.zeros(10)
    pts_y1 = np.zeros(10)
    pts_z1 = np.zeros(10)

    pts_x2 = np.zeros(10)
    pts_y2 = np.zeros(10)
    pts_z2 = np.zeros(10)
    
    pts_x3 = np.zeros(10)
    pts_y3 = np.zeros(10)
    pts_z3 = np.zeros(10)

    pts_x1_cur = np.zeros(10)
    pts_y1_cur = np.zeros(10)
    pts_z1_cur = np.zeros(10)

    pts_x2_cur = np.zeros(10)
    pts_y2_cur = np.zeros(10)
    pts_z2_cur = np.zeros(10)
    
    pts_x3_cur = np.zeros(10)
    pts_y3_cur = np.zeros(10)
    pts_z3_cur = np.zeros(10)

    # for i in range(10):
    #     condition1 = np.logical_and(np.logical_and(Z==0.5, Y==0.5), X==i)
    #     pts_x1[i] = X[condition1][0]
    #     pts_y1[i] = Y[condition1][0]
    #     pts_z1[i] = Z[condition1][0]
    #     pts_x1_cur[i] = X_cur[condition1][0]
    #     pts_y1_cur[i] = Y_cur[condition1][0]
    #     pts_z1_cur[i] = Z_cur[condition1][0]

    #     condition2 = np.logical_and(np.logical_and(Z==0.5, Y==0.9), X==i)
    #     pts_x2[i] = X[condition2][0]
    #     pts_y2[i] = Y[condition2][0]
    #     pts_z2[i] = Z[condition2][0]
    #     pts_x2_cur[i] = X_cur[condition2][0]
    #     pts_y2_cur[i] = Y_cur[condition2][0]
    #     pts_z2_cur[i] = Z_cur[condition2][0]

    #     condition3 = np.logical_and(np.logical_and(Z==0.9, Y==0.5), X==i)
    #     pts_x3[i] = X[condition3][0]
    #     pts_y3[i] = Y[condition3][0]
    #     pts_z3[i] = Z[condition3][0]
    #     pts_x3_cur[i] = X_cur[condition3][0]
    #     pts_y3_cur[i] = Y_cur[condition3][0]
    #     pts_z3_cur[i] = Z_cur[condition3][0]


    pts_x = np.zeros((10, 3))
    pts_x_cur = np.zeros((10, 3))
    pts_y = np.zeros((10, 3))
    pts_y_cur = np.zeros((10, 3))
    pts_z = np.zeros((10, 3))
    pts_z_cur = np.zeros((10, 3))

    for i in range(10):
        condition1 = np.logical_and(np.logical_and(Z==0.5, Y==0.5), X==i)
        pts_x[i] = X[condition1][0], Y[condition1][0], Z[condition1][0]
        pts_x_cur[i] = X_cur[condition1][0], Y_cur[condition1][0], Z_cur[condition1][0]

        condition2 = np.logical_and(np.logical_and(Z==0.5, Y==0.9), X==i)
        pts_y[i] = X[condition2][0], Y[condition2][0], Z[condition2][0]
        pts_y_cur[i] = X_cur[condition2][0], Y_cur[condition2][0], Z_cur[condition2][0]

        condition3 = np.logical_and(np.logical_and(Z==0.9, Y==0.5), X==i)
        pts_z[i] = X[condition3][0], Y[condition3][0], Z[condition3][0]
        pts_z_cur[i] = X_cur[condition3][0], Y_cur[condition3][0], Z_cur[condition3][0]

    condition4 = np.logical_and(Y==0.5, Z==0.5)
    line_x, line_y, line_z = X[condition4], Y[condition4], Z[condition4]
    line_x_cur, line_y_cur, line_z_cur = X_cur[condition4], Y_cur[condition4], Z_cur[condition4]


    condition5 = np.logical_and(np.logical_and(X==10, Y==0.5), Z==1)
    end_x, end_y, end_z = X[condition5], Y[condition5], Z[condition5]
    end_x_cur, end_y_cur, end_z_cur = X_cur[condition5], Y_cur[condition5], Z_cur[condition5]


    # strain_x = np.zeros(9)
    # for i in range(9):
    #     strain_x[i] = (np.linalg.norm(pts_x_cur[i] - pts_x_cur[i+1]) 
    #                    / np.linalg.norm(pts_x[i] - pts_x[i+1]) - 1) * 100
    strain_x = (np.linalg.norm(pts_x_cur[:-1] - pts_x_cur[1:], axis=1)
                / np.linalg.norm(pts_x[:-1] - pts_x[1:], axis=1)
                - 1) * 100
    strain_y = (np.linalg.norm(pts_x_cur - pts_y_cur, axis=1)
                / np.linalg.norm(pts_x - pts_y, axis=1)
                - 1) * 100
    strain_z = (np.linalg.norm(pts_x_cur - pts_z_cur, axis=1)
                / np.linalg.norm(pts_x - pts_z, axis=1)
                - 1) * 100

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    fig.tight_layout()
    ax[0].plot(strain_x, '-x')
    ax[1].plot(strain_y, '-x')
    ax[2].plot(strain_z, '-x')
    # fig.savefig('figures/strain_plot.pdf')
    fig, ax = plt.subplots(1, 1) #, figsize=(4, 2))
    ax.plot(line_x_cur, line_z_cur)
    # fig.savefig('figures/line_plot.pdf')
    fig, ax = plt.subplots(1, 1) #, figsize=(4, 2))
    ax.plot(line_x_cur, line_z_cur)
    ax.set_xlim([9.25, 9.40])
    ax.set_xticks([9.25, 9.40])
    ax.set_ylim([3.60, 3.75])
    ax.set_yticks([3.60, 3.75])
    # fig.savefig('figures/zoom_plot.pdf')
    plt.show()



    # fig = plt.figure(figsize=(5, 3))
    # fig.tight_layout()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect((8, 1, 6))
    # ax.scatter(X, Y, Z, s=1, alpha=.1)
    # ax.scatter(end_x, end_y, end_z, s=5, c='springgreen')
    # ax.scatter(line_x, line_y, line_z, s=.5, c='firebrick')
    # ax.quiver(X[::2, ::10, ::2], Y[::2, ::10, ::2], Z[::2, ::10, ::2], 
    #           u_pred[0, ::2, ::10, ::2], u_pred[1, ::2, ::10, ::2], u_pred[2, ::2, ::10, ::2],
    #           alpha=.3, length=.5)
    # ax.scatter(X_cur, Y_cur, Z_cur, s=1, alpha=1)
    # ax.scatter(end_x_cur, end_y_cur, end_z_cur, s=5, c='forestgreen')
    # ax.scatter(pts_y_cur[:, 0], pts_y_cur[:, 1], pts_y_cur[:, 2], s=2, c='midnightblue')
    # ax.scatter(line_x_cur, line_y_cur, line_z_cur, s=.5, c='firebrick')
    # ax.scatter(pts_x1, pts_y1, pts_z1, s=2, c='midnightblue')
    # ax.scatter(pts_x2, pts_y2, pts_z2, s=2, c='midnightblue')
    # ax.scatter(pts_x3, pts_y3, pts_z3, s=2, c='midnightblue')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    # plt.show()