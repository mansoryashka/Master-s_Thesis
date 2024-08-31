import torch
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK

import sys
sys.path.insert(0, "../..")
from DEM import DeepEnergyMethod, MultiLayerNet, dev
from EnergyModels import *

plt.style.use('default')
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

C = 10E3
bf = bt = bfs = 1

def define_domain(N=15, M=5):
    # N=N+1
    # middle = int(N/2)
    # assert N % M == 0, 'N must be divisible by M!'

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

    u = np.linspace(u_endo, u_epi, M).reshape(1, M, N)

    v = np.linspace(-np.pi, np.pi, N).reshape(N, 1, 1)
    rs = np.linspace(rs_endo, rs_epi, M).reshape((1, M, 1))
    rl = np.linspace(rl_endo, rl_epi, M).reshape((1, M, 1))

    # RS = np.ones((1, M, 1))
    # RL = np.ones((1, M, 1))
    # for i in range(M):
    #     # print(f'fra: {int(N*K/M)*i} til: {int(N*K/M)*(i+1)-1}')
    #     RS[:, i] = rs[i]
    #     RL[:, i] = rl[i]

    x = rs*np.sin(u)*np.cos(v)
    y = rs*np.sin(u)*np.sin(v)
    z = rl*np.cos(u)*np.ones(np.shape(v))

    # x = RS*np.expand_dims(np.outer(np.cos(v), np.sin(u)), 1)
    # y = RS*np.expand_dims(np.outer(np.sin(v), np.sin(u)), 1)
    # z = RL*np.expand_dims(np.outer(np.ones(np.size(v)), np.cos(u)), 1)
    # print(x.shape)
    """ Finn ut hvorfor max(z) = 5.10!!! """
    # set z_max to 5

    # define Dirichlet and Neumann BCs
    dir_BC = 5.0
    neu_BC = rs[0, :, 0] == rs_endo
    # print(neu_BC)
    # print(RS[RS == rs_endo]); exit()
    # define inner points
    # x0 = x[:, ~neu_BC]
    # y0 = y[:, ~neu_BC]
    # z0 = z[:, ~neu_BC]
    # x0 = x0[~dir_BC(z0)]
    # y0 = y0[~dir_BC(z0)]
    # z0 = z0[~dir_BC(z0)]
    z[..., -1] = dir_BC
    x0 = np.copy(x)
    y0 = np.copy(y)
    z0 = np.copy(z)
    # print(x.shape, x0.shape); exit()

    # print(x.shape)
    # define points on Dirichlet boundary
    x1 = x0[:, :, -1]
    y1 = y0[:, :, -1]
    z1 = z0[:, :, -1]

    # define points on Neumann boundary
    x2 = x0[:, neu_BC]
    y2 = y0[:, neu_BC]
    z2 = z0[:, neu_BC]
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
    x_perp = np.copy(x_endo) / rs_endo
    y_perp = np.copy(y_endo) / rs_endo
    z_perp = np.copy(z_endo) / rl_endo

    end = int((N-1)/4)
    x_perp[::2, 1:end] = 0
    y_perp[::2, 1:end] = 0
    z_perp[::2, 1:end] = 0
    x_perp[1:-1:4, 1:end] = 0
    y_perp[1:-1:4, 1:end] = 0
    z_perp[1:-1:4, 1:end] = 0
    x_perp[3:-1:8, 1:end] = 0
    y_perp[3:-1:8, 1:end] = 0
    z_perp[3:-1:8, 1:end] = 0
    x_perp[1:, 0] = 0
    y_perp[1:, 0] = 0
    z_perp[1:, 0] = 0
    x_perp[-1] = 0
    y_perp[-1] = 0
    z_perp[-1] = 0


    # exit(y_perp[0])
    # reshape to have access to different dimentsions
    # dimension 0 is angle
    # dimension 1 is depth layer
    # dimension 2 is vertical level
    # x0 = x0.reshape((N, M, N))
    # y0 = y0.reshape((N, M, N))
    # z0 = z0.reshape((N, M, N))

    # x1 = x1.reshape((N, M, 1))
    # y1 = y1.reshape((N, M, 1))
    # z1 = z1.reshape((N, M, 1))

    # x2 = x2.reshape((N, 1, N))
    # y2 = y2.reshape((N, 1, N))
    # z2 = z2.reshape((N, 1, N))

    # plot domain
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    # ax.scatter(x0[0], y0[0], z0[0], s=.1, alpha=0.5, c='tab:blue')
    # ax.scatter(x0[:, 0, :], y0[:, 0, :], z0[:, 0, :], s=1, c='tab:blue')
    # ax.scatter(x[7], y[7], z[7], s=1, c='tab:blue')
    # ax.scatter(x1, y1, z1, s=5, c='tab:green')
    # ax.scatter(x2, y2, z2, s=5, c='tab:red')
    # plot epicardial and endocardial surfaces
    ax.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
    ax.quiver(x_endo[:, :15], y_endo[:, :15], z_endo[:, :15], x_perp[:, :15], y_perp[:, :15], z_perp[:, :15], alpha=.5)
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

class DeepEnergyMethodLV(DeepEnergyMethod):
    def __call__(self, model, x):
        u = model(x).to(dev)
        Ux, Uy, Uz = (x[:, 2] - 5) * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred
    
    def evaluate_model(self, x, y, z, return_pred_tensor=False):
        Nx = len(x[:, 0, 0])
        Ny = len(y[0, :, 0])
        Nz = len(z[0, 0, :])
        # Nx = N; Ny = M; Nz = N
        x1D = np.expand_dims(x.flatten(), 1)
        y1D = np.expand_dims(y.flatten(), 1)
        z1D = np.expand_dims(z.flatten(), 1)
        xyz = np.concatenate((x1D, y1D, z1D), axis=-1)

        xyz_tensor = torch.from_numpy(xyz).float().to(dev)
        xyz_tensor.requires_grad_(True)

        u_pred_torch = self(self.model, xyz_tensor)
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
    N = 41; M = 5

    middle_layer = int(np.floor(M/2))

    domain, dirichlet, neumann = define_domain(N, M)
    shape = [N, M, N]

    u_endo = np.linspace(-np.pi, -np.arccos(5/17), N)
    u_epi = np.linspace(-np.pi, -np.arccos(5/20), N)
    middle = int(N/2 + 1)
    u = np.linspace(u_endo, u_epi, N)
    u = u.T[:, middle]
    v = np.linspace(-np.pi, np.pi, N)
    rs = np.linspace(rs_endo, rs_epi, M)
    rl = np.linspace(rl_endo, rl_epi, M)
    
    dx = rs_endo / 2 * (v[1] - v[0])
    dy = rs[1] - rs[0]
    dz = ((rl_epi + rs_epi) / 2 + (rl_endo + rs_endo) / 2) / 2 * (u[1] - u[0])
    dxdydz = np.asarray([dx, dy, dz])

    dX = np.zeros(shape[0])
    dY = np.zeros(shape[1])
    dZ = np.zeros(shape[2])

    tmp_domain = domain.reshape((N, M, N, 3))
    
    dZ[1:] = np.cumsum(np.sqrt(
                        (tmp_domain[0, middle_layer, 1:, 0] - tmp_domain[0, middle_layer, :-1, 0])**2
                      + (tmp_domain[0, middle_layer, 1:, 2] - tmp_domain[0, middle_layer, :-1, 2])**2))

    dY[1:] = np.cumsum(tmp_domain[0, 1:, -1, 0] - tmp_domain[0, :-1, -1, 0])

    dX[1:] = np.cumsum(np.sqrt(
                        (tmp_domain[1:, 0, -1, 0] - tmp_domain[:-1, 0, -1, 0])**2
                      + (tmp_domain[1:, 0, -1, 1] - tmp_domain[:-1, 0, -1, 1])**2))

    neumann_domain = neumann['coords'].reshape((N, N, 3))
    # exit(neumann_domain.shape)
    dX_neumann = np.zeros(N)
    dZ_neumann = np.zeros(N)

    dZ_neumann[1:] = np.cumsum(np.sqrt(
                        (neumann_domain[0, 1:, 0] - neumann_domain[0, :-1, 0])**2
                      + (neumann_domain[0, 1:, 2] - neumann_domain[0, :-1, 2])**2))

    dX_neumann[1:] = np.cumsum(np.sqrt(
                        (neumann_domain[1:, -1, 0] - neumann_domain[:-1, -1, 0])**2
                      + (neumann_domain[1:, -1, 1] - neumann_domain[:-1, -1, 1])**2))


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

    z[..., -1] = 5.0

    # z = np.where(np.abs(z - 5E-3) < 1E-3, 5E-3, z)
    
    model = MultiLayerNet(3, *[80]*6, 3)
    energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E5)
    # energy = NeoHookeanEnergyModel(200, 100)
    DemLV = DeepEnergyMethodLV(model, energy)
    # DemLV.train_model(domain, dirichlet, neumann, shape=shape, dxdydz=dxdydz, neu_axis=[0, 2], lr=.1, epochs=20, fb=np.array([[0, 0, 0]]))
    DemLV.train_model(domain, dirichlet, neumann, 
                      shape=shape, dxdydz=[[dX, dY, dZ], [dX_neumann, dZ_neumann]], 
                      LHD=np.zeros(3), neu_axis=[0, 2], lr=.5, epochs=27, fb=np.array([[0, 0, 0]]))

    # # print()
    U_pred = DemLV.evaluate_model(x, y, z)
    write_vtk_v3('output/DemLV', x, y, z, U_pred)
    # # exit()
    # np.save('stored_arrays/DemLV', np.asarray(U_pred))
    U_pred = np.load('stored_arrays/DemLV.npy')
    exit()
    domain = domain.reshape((N, M, N, 3))
    X, Y, Z = domain[..., 0], domain[..., 1], domain[..., 2]
    X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]


    U = np.asarray(U_pred) + domain.T.reshape((3, N, M, N))
    k = int((N-1)/2)

    ref_x = X[k, 2]
    ref_z = Z[k, 2]

    line = np.asarray(U_pred)
    cur_x = X_cur[k, 2]
    cur_z = Z_cur[k, 2]

    fig1, ax1 = plt.subplots(figsize=(3, 6))
    ax1.plot(ref_x, ref_z, c='gray', linestyle=':')
    ax1.plot(cur_x, cur_z, label=f'{k}')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(cur_x, cur_z)
    ax2.set_xlim(left=-14,right=-11)
    ax2.set_ylim((-9, -2))

    fig3, ax3 = plt.subplots()
    ax3.plot(cur_x, cur_z)
    ax3.set_xlim((-5, 0))
    ax3.set_ylim((-28, -25))

    fig4, ax4 = plt.subplots()
    ax4.plot(Z_cur[0, 0, 0], marker='x', c='C0')
    ax4.plot(Z_cur[0, -1, 0], marker='x', c='C0')
    plt.show()

    # exit()