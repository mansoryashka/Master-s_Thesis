import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import sys
sys.path.insert(0, "../../3DBeam")
sys.path.insert(0, "../../")
from DemBeam3D import DeepEnergyMethodBeam, train_and_evaluate, MultiLayerNet, write_vtk_v2, dev

current_path = Path.cwd()

L = 10
H = D = 1

x0 = 0; x1 = L
y0 = 0; y1 = H
z0 = 0; z1 = D

N_test = 20
dx = L / (10*N_test)
dy = H/N_test
dx = D/N_test

C = 10E3
bf = 1
bt = 1
bfs = 1

f0 = np.array([1, 0, 0])

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = 0.0
n_cond = [0, 4, 0]

def define_domain(L, H, D, N=10):
    x = np.linspace(0, L, 10*N)
    y = np.linspace(0, H, N)
    z = np.linspace(0, D, N)

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
        plt.show()
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

def energy(u, x, J=False):
    kappa = 1E3
    # Guccione energy mode. Get source from verification paper!!!
    duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]


    Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
    Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
    Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
    Fyx = duydxyz[:, 0].unsqueeze(1) + 0
    Fyy = duydxyz[:, 1].unsqueeze(1) + 1
    Fyz = duydxyz[:, 2].unsqueeze(1) + 0
    Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
    Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
    Fzz = duzdxyz[:, 2].unsqueeze(1) + 1


    detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    
    # invF11 = (Fyy * Fzz - Fyz * Fzy) / detF
    # invF12 = -(Fxy * Fzz - Fxz * Fzy) / detF
    # invF13 = (Fxy * Fyz - Fxz * Fyy) / detF
    # invF21 = -(Fyx * Fzz - Fyz * Fzx) / detF
    # invF22 = (Fxx * Fzz - Fxz * Fzy) / detF
    # invF23 = -(Fxx * Fyz - Fxz * Fyx) / detF
    # invF31 = (Fyx * Fzy - Fyy * Fzy) / detF
    # invF32 = -(Fxx * Fzy - Fxy * Fzx) / detF
    # invF33 = (Fxx * Fyy - Fxy * Fyx) / detF

    E11 = 0.5*(Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)
    E12 = 0.5*(Fxx*Fxy + Fyx*Fyy + Fzx*Fzy - 0)
    E13 = 0.5*(Fxx*Fxz + Fyx*Fyz + Fzx*Fzz - 0)
    E21 = 0.5*(Fxy*Fxx + Fyy*Fyx + Fzy*Fzx - 0)
    E22 = 0.5*(Fxy*Fxy + Fyy*Fyy + Fzy*Fzy - 1)
    E23 = 0.5*(Fxy*Fxz + Fyy*Fyz + Fzy*Fzz - 0)
    E31 = 0.5*(Fxz*Fxx + Fyz*Fyx + Fzz*Fzx - 0)
    E32 = 0.5*(Fxz*Fxy + Fyz*Fyy + Fzz*Fzy - 0)
    E33 = 0.5*(Fxz*Fxz + Fyz*Fyz + Fzz*Fzz - 1)

    Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) + bfs*(E12**2 + E21**2 + E13**2 + E31**2)
    W = C / 2 * (torch.exp(Q) - 1)

    compressibility = kappa/2 * (detF - 1)**2

    total_energy = W + compressibility

    # print(W[W == np.inf]); exit()

    if J:
        return total_energy, J
    return total_energy

if __name__ == '__main__':
    N = 10
    LHD = [L, H, D]
    shape = [10*N, N, N]

    x_test = np.linspace(0, L, 10*N)
    y_test = np.linspace(0, H, N+3)[1:-2]
    z_test = np.linspace(0, D, N+3)[1:-2]

    model = MultiLayerNet(3, *[30]*3, 3)
    DemBeam = DeepEnergyMethodBeam(model, energy)
    domain, dirichlet, neumann = define_domain(L, H, D, N=N)
    DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=0.1, epochs=1, fb=np.array([[0, 0, 0]]))
    U_pred = DemBeam.evaluate_model(x_test, y_test, z_test)
    write_vtk_v2('output/problem1', x_test, y_test, z_test, U_pred)
