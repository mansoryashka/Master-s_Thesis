import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../..")
from DEM import DeepEnergyMethod, MultiLayerNet, dev, write_vtk_LV
from EnergyModels import *
# import seaborn as sns 
# sns.set()

import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

C = 10
bf = bt = bfs = 1
n_cond = 10

def define_domain(N=13, M=3,
                  n_cond=10,
                  rs_endo=7,
                  rl_endo=17,
                  rs_epi=10,
                  rl_epi=20,
                  plot=False):

    rs = np.linspace(rs_endo, rs_epi, M).reshape((1, M, 1))
    rl = np.linspace(rl_endo, rl_epi, M).reshape((1, M, 1))
    # Dirichlet BC implemented in definition of u
    u = np.linspace(-np.pi, -np.arccos(5/rl), N).reshape((N, M, 1)).T
    v = np.linspace(-np.pi, np.pi, N).reshape(N, 1, 1)

    # shape of resuling domain
    # dimension 0 is angle
    # dimension 1 is depth layer
    # dimension 2 is vertical level
    x = rs*np.sin(u)*np.cos(v)
    y = rs*np.sin(u)*np.sin(v)
    z = rl*np.cos(u)*np.ones(np.shape(v))
    
    # define Neumann BCs
    neu_B = rs[0, :, 0] == rs_endo

    # define points on Dirichlet boundary
    x1 = x[:, :, -1]
    y1 = y[:, :, -1]
    z1 = z[:, :, -1]

    # define points on Neumann boundary
    x2 = x[:, neu_B]
    y2 = y[:, neu_B]
    z2 = z[:, neu_B]

    # define endocardium surface for illustration
    x_endo = x[:, 0]
    y_endo = y[:, 0]
    z_endo = z[:, 0]

    # define epicardium surface for illustration
    x_epi = x[:, -1]
    y_epi = y[:, -1]
    z_epi = z[:, -1]

    # define vector penpendicular to the endocardium
    # and scale vector to unit length
    x_perp = np.copy(x_endo) / rs_endo
    y_perp = np.copy(y_endo) / rs_endo
    z_perp = np.copy(z_endo) / rl_endo

    # calculate distances between points
    dv = np.sqrt(
        (x_perp[1:] - x_perp[:-1])**2
      + (y_perp[1:] - y_perp[:-1])**2
      + (z_perp[1:] - z_perp[:-1])**2
    )
    # scale dv such that maximum value is 1
    dv /= np.max(dv)
    dv = dv[0]*np.ones((N, N))
    # multiply with scaled dv, so the largest 
    # force stays as it is while all other forces
    # get scaled down based on their closeness to other vectors
    x_perp *= dv
    y_perp *= dv
    z_perp *= dv

    # plot domain
    if plot:
        fig = plt.figure()
        # plt.style.use('default')
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        # ax.scatter(x[0], y[0], z[0], s=.1, alpha=0.5, c='tab:blue')
        # ax.scatter(x[:, 0, :], y[:, 0, :], z[:, 0, :], s=1, c='tab:blue')
        # ax.scatter(x[7], y[7], z[7], s=1, c='tab:blue')

        ax.scatter(x[..., -2], y[..., -2], z[..., -2], s=.1, alpha=0.5, c='tab:blue')
        # ax.scatter(x1, y1, z1, s=5, c='tab:green')
        # ax.scatter(x2, y2, z2, s=5, c='tab:red')

        # plot epicardial and endocardial surfaces
        ax.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
        # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
        ax.quiver(x_endo[:, :], y_endo[:, :], z_endo[:, :], 
                  x_perp[:, :], y_perp[:, :], z_perp[:, :], alpha=.5)

        # plt.show(); exit()
        plt.savefig('figures/ventricle.pdf')
        plt.close()

    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    z = z.flatten().reshape(-1, 1)
    domain = np.concatenate((x, y, z), -1)

    x1 = x1.flatten().reshape(-1, 1)
    y1 = y1.flatten().reshape(-1, 1)
    z1 = z1.flatten().reshape(-1, 1)
    d_cond = 0
    db_pts = np.concatenate((x1, y1, z1), -1)
    db_vals = np.ones(np.shape(db_pts)) * d_cond

    x2 = x2.flatten().reshape(-1, 1)
    y2 = y2.flatten().reshape(-1, 1)
    z2 = z2.flatten().reshape(-1, 1)
    nb_pts = np.concatenate((x2, y2, z2), -1)

    x_perp = x_perp.flatten().reshape(-1, 1)
    y_perp = y_perp.flatten().reshape(-1, 1)
    z_perp = z_perp.flatten().reshape(-1, 1)
    nb_vals = n_cond*np.concatenate((x_perp, y_perp, z_perp), -1)

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

def generate_integration_line(domain, neumann, shape):
    N, M, N = shape
    dX = np.zeros(N)
    dY = np.zeros(M)
    dZ = np.zeros(N)

    tmp_domain = domain.reshape((N, M, N, 3))
    middle_layer = int(np.floor(M/2))

    # dZ[1:] = np.cumsum(np.sqrt(
    #                     (tmp_domain[0, middle_layer, 1:, 0] - tmp_domain[0, middle_layer, :-1, 0])**2
    #                   + (tmp_domain[0, middle_layer, 1:, 2] - tmp_domain[0, middle_layer, :-1, 2])**2))
    dZ[1:] = np.cumsum(np.linalg.norm(
                        tmp_domain[0, middle_layer, 1:, 0::2] 
                      - tmp_domain[0, middle_layer, :-1, 0::2], axis=-1))
    
    dY[1:] = np.cumsum(tmp_domain[0, 1:, -1, 0] - tmp_domain[0, :-1, -1, 0])

    # dX[1:] = np.cumsum(np.sqrt(
    #                     (tmp_domain[1:, 0, -1, 0] - tmp_domain[:-1, 0, -1, 0])**2
    #                   + (tmp_domain[1:, 0, -1, 1] - tmp_domain[:-1, 0, -1, 1])**2))
    dX[1:] = np.cumsum(np.linalg.norm(
                        tmp_domain[1:, 0, -1, :-1] 
                      - tmp_domain[:-1, 0, -1, :-1], axis=-1))
    
    neumann_domain = neumann['coords'].reshape((N, N, 3))
    # exit(neumann_domain.shape)
    dX_neumann = np.zeros(N)
    dZ_neumann = np.zeros(N)

    # dZ_neumann[1:] = np.cumsum(np.sqrt(
    #                     (neumann_domain[0, 1:, 0] - neumann_domain[0, :-1, 0])**2
    #                   + (neumann_domain[0, 1:, 2] - neumann_domain[0, :-1, 2])**2))
    dZ_neumann[1:] = np.cumsum(np.linalg.norm(
                        neumann_domain[0, 1:, 0::2] 
                      - neumann_domain[0, :-1, 0::2], axis=-1))
    # dX_neumann[1:] = np.cumsum(np.sqrt(
    #                     (neumann_domain[1:, -1, 0] - neumann_domain[:-1, -1, 0])**2
    #                   + (neumann_domain[1:, -1, 1] - neumann_domain[:-1, -1, 1])**2))
    dX_neumann[1:] = np.cumsum(np.linalg.norm(
                        neumann_domain[1:, -1, :-1] 
                      - neumann_domain[:-1, -1, :-1], axis=-1))
    
    return dX, dY, dZ, dX_neumann, dZ_neumann

def plot_displacement(X, Z, X_cur, Z_cur, trainin_shape, axs, figname):
    N, M, N = trainin_shape
    N_test = len(X[:, 0, 0])
    M_test = len(X[0, :, 0])
    ax1, ax2, ax3 = axs
    # get indexes of line to be plotted
    k = int((N_test - 1) / 2)
    middle_layer = int(np.floor(M_test / 2))    
    # get line in reference configuration
    ref_x = X[k, middle_layer]
    ref_z = Z[k, middle_layer]
    # get line in current configuration
    cur_x = X_cur[k, middle_layer]
    cur_z = Z_cur[k, middle_layer]

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
    ax3.set_xlim((-5, 0))
    # ax3.set_xticks([-13, -9])
    # ax3.set_yticks([-27, -23])
    # ax3.set_yticks([-27, -23)

    plt.tight_layout()
    plt.savefig(f'figures/{figname}.pdf')

if __name__ == '__main__':
    N_test = 4; M_test = 3
    test_domain, _, _ = define_domain(N_test, M_test, n_cond=n_cond)
    test_domain = test_domain.reshape((N_test, M_test, N_test, 3))
    x_test = np.ascontiguousarray(test_domain[..., 0])
    y_test = np.ascontiguousarray(test_domain[..., 1])
    z_test = np.ascontiguousarray(test_domain[..., 2])

    # plt.style.use('seaborn-v0_8-darkgrid')
    # fig2, ax = plt.subplots()
    # fig = plt.figure()
    # plt.style.use('seaborn-v0_8-darkgrid')
    # ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    # ax2 = plt.subplot2grid((2,2), (0,1))
    # ax3 = plt.subplot2grid((2,2), (1,1))
    # for N, M in zip([30, 40, 40, 50, 60, 60, 80], [3, 3, 5, 5, 5, 9, 9]):
    for N, M in zip([100], [9]):
        middle_layer = int(np.floor(M/2))

        domain, dirichlet, neumann = define_domain(N, M, n_cond=n_cond)
        shape = [N, M, N]

        dX, dY, dZ, dX_neumann, dZ_neumann = generate_integration_line(domain, 
                                                                       neumann,
                                                                       shape)
        
        for lr, nn, nl in zip([0.05], [50], [3]):
            model = MultiLayerNet(3, *[nn]*nl, 3)
            # energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E3)
            energy = GuccioneEnergyModel(C, bf, bt, bfs, kappa=1E3)
            DemLV = DeepEnergyMethodLV(model, energy)
            DemLV.train_model(domain, dirichlet, neumann, 
                              shape=shape, dxdydz=[dX, dY, dZ, dX_neumann, dZ_neumann], 
                              LHD=np.zeros(3), neu_axis=[0, 2], lr=lr, epochs=300,
                              fb=np.array([[0, 0, 0]]),  ventricle_geometry=True)

            torch.save(DemLV.model.state_dict(), f'trained_models/run1/model_lr{lr}_nn{nn}_nl{nl}')
        # DemLV.model.load_state_dict(torch.load(f'trained_models/run1/model_{N}x{M}'))


        # U_pred = DemLV.evaluate_model(x_test, y_test, z_test)
        # write_vtk_LV(f'output/DemLV{N}x{M}', x_test, y_test, z_test, U_pred)

        # np.save(f'stored_arrays/DemLV{N}x{M}', np.asarray(U_pred))