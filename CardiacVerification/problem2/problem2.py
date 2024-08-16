import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from torch.autograd import grad
from pyevtk.hl import gridToVTK


import sys
sys.path.insert(0, "../../3DBeam")
sys.path.insert(0, "../..")
from DemBeam3D import DeepEnergyMethodBeam, train_and_evaluate, MultiLayerNet, write_vtk_v2, dev
from EnergyModels import GuccioneEnergyModel
plt.style.use('default')
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

C = 10E3
bf = bt = bfs = 1

def define_domain(N=15, M=5):
    K = N
    
    middle = int(K/2)
    assert N % M == 0, 'N must be divisible by M!'

    rs_endo = 7
    rl_endo = 17
    # rs_endo = 7E-3
    # rl_endo = 17E-3
    u_endo = np.linspace(-np.pi, -np.arccos(5/17), N)
    v_endo = np.linspace(-np.pi, np.pi, N)

    rs_epi = 10
    rl_epi = 20
    # rs_epi = 10E-3
    # rl_epi = 20E-3
    u_epi = np.linspace(-np.pi, -np.arccos(5/20), N)
    v_epi = np.linspace(-np.pi, np.pi, N)

    u = np.linspace(u_endo, u_epi, K)
    u = u.T[:, -1]

    v = np.linspace(-np.pi, np.pi, N)
    rs = np.linspace(rs_endo, rs_epi, M)
    rl = np.linspace(rl_endo, rl_epi, M)

    RS = np.ones((N, M, N))
    RL = np.ones((N, M, N))
    for i in range(M):
        # print(f'fra: {int(N*K/M)*i} til: {int(N*K/M)*(i+1)-1}')
        RS[:, i] = rs[i]
        RL[:, i] = rl[i]

    x = RS*np.expand_dims(np.outer(np.cos(v), np.sin(u)), 1)
    y = RS*np.expand_dims(np.outer(np.sin(v), np.sin(u)), 1)
    z = RL*np.expand_dims(np.outer(np.ones(np.size(v)), np.cos(u)), 1)
    # print(x.shape)
    """ Finn ut hvorfor max(z) = 5.10!!! """
    # set z_max to 5
    z = np.where(np.abs(z - 5) < 1, 5, z)
    # z = np.where(np.abs(z - 5E-3) < 1E-3, 5E-3, z)

    # define Dirichlet and Neumann BCs
    dir_BC = lambda z: np.abs(z - 5) < 1
    # dir_BC = lambda z: np.abs(z - 5E-3) < 5E-4
    neu_BC = RS == rs_endo

    # print(RS[RS == rs_endo]); exit()
    # define inner points
    # x0 = x[:, ~neu_BC]
    # y0 = y[:, ~neu_BC]
    # z0 = z[:, ~neu_BC]
    # x0 = x0[~dir_BC(z0)]
    # y0 = y0[~dir_BC(z0)]
    # z0 = z0[~dir_BC(z0)]
    x0 = np.copy(x)
    y0 = np.copy(y)
    z0 = np.copy(z)
    # print(x.shape)
    # define points on Dirichlet boundary
    x1 = x[dir_BC(z)]
    y1 = y[dir_BC(z)]
    z1 = z[dir_BC(z)]

    # define points on Neumann boundary
    x2 = x[neu_BC]
    y2 = y[neu_BC]
    z2 = z[neu_BC]
    # x2 = x2[~dir_BC(z2)]
    # y2 = y2[~dir_BC(z2)]
    # z2 = z2[~dir_BC(z2)]

    # define endocardium surface for illustration
    x_endo = rs_endo*np.outer(np.cos(v_endo), np.sin(u_endo))
    y_endo = rs_endo*np.outer(np.sin(v_endo), np.sin(u_endo))
    z_endo = rl_endo*np.outer(np.ones(np.size(v_endo)), np.cos(u_endo))

    # define epicardium surface for illustration
    x_epi = rs_epi*np.outer(np.cos(v_epi), np.sin(u_epi))
    y_epi = rs_epi*np.outer(np.sin(v_epi), np.sin(u_epi))
    z_epi = rl_epi*np.outer(np.ones(np.size(v_epi)), np.cos(u_epi))

    # define vector penpendicular to the endocardium
    # from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid
    x_perp = 2*np.copy(x_endo) / rs_endo**2
    y_perp = 2*np.copy(y_endo) / rs_endo**2
    z_perp = 2*np.copy(z_endo) / rl_endo**2
    # x_perp[1:, 0] = 0
    # y_perp[1:, 0] = 0
    # z_perp[1:, 0] = 0

    # reshape to have access to different dimentsions
    # dimension 0 is angle
    # dimension 1 is depth layer
    # dimension 2 is vertical level
    x0 = x0.reshape((N, M, N))
    y0 = y0.reshape((N, M, N))
    z0 = z0.reshape((N, M, N))

    x1 = x1.reshape((N, M, 1))
    y1 = y1.reshape((N, M, 1))
    z1 = z1.reshape((N, M, 1))

    x2 = x2.reshape((N, 1, N))
    y2 = y2.reshape((N, 1, N))
    z2 = z2.reshape((N, 1, N))

    # plot domain
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    # ax.scatter(x0, y0, z0, s=1, c='tab:blue')
    # ax.scatter(x0[:3, :, :15], y0[:3, :, :15], z0[:3, :, :15], s=1, c='tab:blue')
    # ax.scatter(x[7], y[7], z[7], s=1, c='tab:blue')
    ax.scatter(x1, y1, z1, s=5, c='tab:green')
    # # ax.scatter(x2, y2, z2, s=5, c='tab:red')
    # plot epicardial and endocardial surfaces
    # ax.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
    # ax.scatter(dx, dy, dz)
    # ax.scatter(x_perp, y_perp, z_perp)
    # ax.quiver(x_endo[:, :-1], y_endo[:, :-1], z_endo[:, :-1], x_perp, y_perp, z_perp, alpha=.1)

    # plt.show(); exit()
    plt.savefig('ventricle.pdf')
    plt.close()

    x0 = np.expand_dims(x0.flatten(), 1)
    y0 = np.expand_dims(y0.flatten(), 1)
    z0 = np.expand_dims(z0.flatten(), 1)
    domain = np.concatenate((x0, y0, z0), -1)

    x1 = np.expand_dims(x1.flatten(), 1)
    y1 = np.expand_dims(y1.flatten(), 1)
    z1 = np.expand_dims(z1.flatten(), 1)
    d_cond = 0
    db_pts = np.concatenate((x1, y1, z1), -1)
    db_vals = np.ones(np.shape(db_pts)) * d_cond


    x_perp = np.expand_dims(x_perp.flatten(), 1)
    y_perp = np.expand_dims(y_perp.flatten(), 1)
    z_perp = np.expand_dims(z_perp.flatten(), 1)

    n_cond = 1E4*np.concatenate((x_perp, y_perp, z_perp), -1)
    
    x2 = np.expand_dims(x2.flatten(), 1)
    y2 = np.expand_dims(y2.flatten(), 1)
    z2 = np.expand_dims(z2.flatten(), 1)
    nb_pts = np.concatenate((x2, y2, z2), -1)
    nb_vals = n_cond

    dirichlet = {
        'coords': db_pts,
        'values': db_vals
    }

    neumann = {
        'coords': nb_pts,
        'values': nb_vals
    }

    return domain, dirichlet, neumann

class DeepEnergyMethodLV(DeepEnergyMethodBeam):
    def getU(self, model, x):
        u = model(x).to(dev)
        Ux, Uy, Uz = (x[:, 2] - 5) * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred
    
    def evaluate_model(self, x, y, z, return_pred_tensor=False):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        Nx = N; Ny = M; Nz = N
        x1D = np.expand_dims(x.flatten(), 1)
        y1D = np.expand_dims(y.flatten(), 1)
        z1D = np.expand_dims(z.flatten(), 1)
        xyz = np.concatenate((x1D, y1D, z1D), axis=-1)

        xyz_tensor = torch.from_numpy(xyz).float().to(dev)
        xyz_tensor.requires_grad_(True)

        u_pred_torch = self.getU(self.model, xyz_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()

        surUx = u_pred[:, 0].reshape(Nx, Ny, Nz)
        surUy = u_pred[:, 1].reshape(Nx, Ny, Nz)
        surUz = u_pred[:, 2].reshape(Nx, Ny, Nz)

        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        return U


def write_vtk_v3(filename, x_space, y_space, z_space, U):
    # xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    xx, yy, zz = x_space, y_space, z_space
    if isinstance(U, dict):
        gridToVTK(filename, xx, yy, zz, pointData=U)
    else:
        gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

if __name__ == '__main__':

    rs_endo =  7
    rl_endo = 17
    rs_epi =  10
    rl_epi =  20
    N = 15; M = 5
    domain, dirichlet, neumann = define_domain(N, M)
    shape = [N, M, N]

    K=N
    u_endo = np.linspace(-np.pi, -np.arccos(5/17), N)
    u_epi = np.linspace(-np.pi, -np.arccos(5/20), N)
    middle = int(N/2)
    u = np.linspace(u_endo, u_epi, K)
    u = u.T[:, middle]
    v = np.linspace(-np.pi, np.pi, N)
    rs = np.linspace(rs_endo, rs_epi, M)
    rl = np.linspace(rl_endo, rl_epi, M)
    
    dx = rs_endo * (v[1] - v[0])
    dy = rs[1] - rs[0]
    dz = ((rl_epi + rs_epi) / 2 + (rl_endo + rs_endo) / 2) / 2 * (u[1] - u[0])

    dxdydz = np.asarray([dx, dy, dz])

    K = N
    u_endo = np.linspace(-np.pi, -np.arccos(5/17), N)
    v_endo = np.linspace(-np.pi, np.pi, N)

    u_epi = np.linspace(-np.pi, -np.arccos(5/20), N)
    v_epi = np.linspace(-np.pi, np.pi, N)


    RS = np.ones((N, M, N))
    RL = np.ones((N, M, N))
    for i in range(M):
        # print(f'fra: {int(N*K/M)*i} til: {int(N*K/M)*(i+1)-1}')
        RS[:, i] = rs[i]
        RL[:, i] = rl[i]

    x = RS*np.expand_dims(np.outer(np.cos(v), np.sin(u)), 1)
    y = RS*np.expand_dims(np.outer(np.sin(v), np.sin(u)), 1)
    z = RL*np.expand_dims(np.outer(np.ones(np.size(v)), np.cos(u)), 1)
    # z = np.where(np.abs(z - 5E-3) < 1E-3, 5E-3, z)
    z = np.where(np.abs(z - 5) < 1, 5, z)
    
    model = MultiLayerNet(3, 60, 60, 60, 60, 3)
    energy = GuccioneEnergyModel(C, bf, bt, bfs)
    DemLV = DeepEnergyMethodLV(model, energy)
    DemLV.train_model(domain, dirichlet, neumann, shape=shape, LHD=None, dxdydz=dxdydz, neu_axis=[0, 2], lr=.5, epochs=30, fb=np.array([[0, 0, 0]]))

    print()
    U_pred = DemLV.evaluate_model(x, y, z)
    write_vtk_v3('output/DemLV', x, y, z, U_pred)

    U = np.asarray(U_pred) + domain.T.reshape((3, N, M, N))
    k = 12
    x = domain[:, 0].reshape((N, M, N))
    z = domain[:, -1].reshape((N, M, N))
    ref_x = x[k, 2]
    ref_z = z[k, 2]

    line = np.asarray(U_pred)
    cur_x = U[0, k, 2]
    # y = U[1, k, 2]
    cur_z = U[2, k, 2]

    plt.figure(figsize=(3, 6))
    
    plt.plot(ref_x, ref_z, c='gray', linestyle=':')
    plt.plot(cur_x, cur_z)
    plt.figure()
    plt.plot(cur_x, cur_z)
    plt.xlim(left=-14,right=-11)
    plt.ylim((-9, -2))
    plt.figure()
    plt.plot(cur_x, cur_z)
    plt.xlim((-5, 0))
    plt.ylim((-28, -25))
    plt.show()
    # exit()

