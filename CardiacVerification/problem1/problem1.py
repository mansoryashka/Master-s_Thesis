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

C = 2
bf = 8
bt = 2
bfs = 4

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = 0.0
n_cond = [0, 0, 0.004]

def define_domain(L, H, D, N=10):
    x = np.linspace(0, L, 10*N)
    y = np.linspace(0, H, N)
    z = np.linspace(0, D, N)

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
    N = 30

    N_test = 10
    LHD = [L, H, D]
    shape = [10*N, N, N]
    
    x_test = np.linspace(0, L, 10*N_test+1)
    y_test = np.linspace(0, H, N_test+1)
    z_test = np.linspace(0, D, N_test+1)

    domain, dirichlet, neumann = define_domain(L, H, D, N=N)
    energy = GuccioneTransverseEnergyModel(C, bf, bt, bfs)
    
    for lr, nn, nl in zip([0.1, 0.1, 0.5], [20, 30, 40], [4, 3, 3]):
        model = MultiLayerNet(3, *[nn]*nl, 3)
        DemBeam = DeepEnergyMethodBeam(model, energy)
        DemBeam.train_model(domain, dirichlet, neumann,
                            shape, LHD, neu_axis=[0, 1], 
                            lr=lr, epochs=100)
        torch.save(DemBeam.model.state_dict(), Path('trained_models') / f'model_lr{lr}_nn{nn}_nl{nl}_100')
