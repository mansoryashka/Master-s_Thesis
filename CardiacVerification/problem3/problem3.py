import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350


import sys
sys.path.insert(0, "../problem2")
sys.path.insert(0, "../../")
from problem2 import DeepEnergyMethodLV, write_vtk_LV, define_domain, generate_integration_line
from DEM import MultiLayerNet, dev
from EnergyModels import GuccioneTransverseActiveEnergyModel

print(dev)
# plt.style.use('default')
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

C = 2
bf = 8
bt = 2
bfs = 4

rs_endo = 7
rl_endo = 17

rs_epi = 10
rl_epi = 20
n_cond = 15

def normalize(u):
    return u / np.linalg.norm(u, axis=0, keepdims=True)

def generate_fibers(N=15, M=3,
                    rs_endo=7,
                    rl_endo=17,
                    rs_epi=10,
                    rl_epi=20,
                    alpha_endo=90,
                    alpha_epi=-90):
    # fiber function from finsberg -> pulse

    t = np.linspace(0, 1, M)
    alpha = lambda t: (alpha_endo + (alpha_epi - alpha_endo) * t) * (np.pi / 180)
    r_long = lambda t: (rl_endo + (rl_epi - rl_endo) * t)
    r_short = lambda t: (rs_endo + (rs_epi - rs_endo) * t)

    rl = r_long(t).reshape((1, M, 1))
    rs = r_short(t).reshape((1, M, 1))
    al = alpha(t).reshape((1, M, 1))

    u = np.linspace(-np.pi, -np.arccos(5/rl), N).reshape((N, M, 1)).T
    v = np.linspace(-np.pi, np.pi, N).reshape((N, 1, 1))

    e_u = np.array(
        [
            rs * np.cos(u) * np.cos(v),
            rs * np.cos(u) * np.sin(v),
            -rl * np.ones_like(v) * np.sin(u),
        ],
    )
    e_u = normalize(e_u)

    e_v = np.array(
        [
            -rs * np.sin(u) * np.sin(v),
            rs * np.sin(u) * np.cos(v),
            np.zeros_like(e_u[0]),
        ],
    )
    e_v = normalize(e_v)

    f0 = np.sin(al) * e_u + np.cos(al) * e_v
    f0 = normalize(f0)

    n0 = np.cross(e_u, e_v, axis=0)
    n0 = normalize(n0)

    s0 = np.cross(f0, n0, axis=0)
    s0 = normalize(s0)

    # f0[-1, :] = 0.0
    # s0[-1, :] = 0.0
    # n0[-1, :] = 0.0
    # set apex to zero or one?
    # f0[..., 0] = 0
    # s0[..., 0] = 0
    # n0[..., 0] = 0
    # f0[..., -1] = 0
    # s0[..., -1] = 0
    # n0[..., -1] = 0
    return f0.reshape((3, -1, 1)), s0.reshape((3, -1, 1)), n0.reshape((3, -1, 1))

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
    ax3.set_ylim(bottom=-20)
    ax3.set_xlim((-5, 0))
    # ax3.set_xticks([-13, -9])
    # ax3.set_yticks([-27, -23])
    # ax3.set_yticks([-27, -23)

    plt.tight_layout()
    plt.savefig(f'figures/{figname}.pdf')

if __name__ == '__main__':
    N_test = 41; M_test = 9
    middle_layer = int(np.floor(M_test/2))
    test_domain, _, _ = define_domain(N_test, M_test, n_cond=15)
    test_domain = test_domain.reshape((N_test, M_test, N_test, 3))
    x_test = np.ascontiguousarray(test_domain[..., 0])
    y_test = np.ascontiguousarray(test_domain[..., 1])
    z_test = np.ascontiguousarray(test_domain[..., 2])
    # N = 80; M = 5
    N = 100; M = 9
    shape = [N, M, N]

    for lr, nn, nl in zip([0.1], [50], [3]):
        domain, dirichlet, neumann = define_domain(N, M, n_cond=15, plot=False)
        f0, s0, n0 = generate_fibers(N, M, alpha_endo=90, alpha_epi=-90)


        dX, dY, dZ, dX_neumann, dZ_neumann = generate_integration_line(domain, 
                                                                        neumann,
                                                                        shape)


        f0, s0, n0 = generate_fibers(N, M)
        f0 = torch.tensor(f0).to(dev)
        s0 = torch.tensor(s0).to(dev)
        n0 = torch.tensor(n0).to(dev)

        model = MultiLayerNet(3, *[nn]*nl, 3)
        energy = GuccioneTransverseActiveEnergyModel(C, bf, bt, bfs, kappa=1E3, Ta=60, f0=f0, s0=s0, n0=n0)
        DemLV = DeepEnergyMethodLV(model, energy)
        DemLV.train_model(domain, dirichlet, neumann, shape=shape, LHD=None, 
                          dxdydz=[dX, dY, dZ, dX_neumann, dZ_neumann], 
                          neu_axis=[0, 2], lr=lr, epochs=300, 
                          fb=np.array([[0, 0, 0]]), ventricle_geometry=True)
        torch.save(DemLV.model.state_dict(), f'trained_models/model_{lr}_nn{nn}_nl{nl}')
    
    # U_pred = DemLV.evaluate_model(x_test, y_test, z_test)
    # np.save('stored_arrays/U_pred', np.asarray(U_pred))
    # write_vtk_LV('output/DemLV', x_test, y_test, z_test, U_pred)
    # U_pred = np.load('stored_arrays/U_pred2.npy')
    # DemLV.model.load_state_dict(torch.load('trained_models/run1/model1'))
    # U_pred = DemLV.evaluate_model(x_test, y_test, z_test)

    # X = np.copy(x_test)
    # Y = np.copy(y_test)
    # Z = np.copy(z_test)
    # X_cur, Y_cur, Z_cur = X + U_pred[0], Y + U_pred[1], Z + U_pred[2]

    # # plt.style.use('seaborn-v0_8-darkgrid')
    # # fig2, ax = plt.subplots()
    # # fig = plt.figure()
    # # plt.style.use('seaborn-v0_8-darkgrid')
    # # ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    # # ax2 = plt.subplot2grid((2,2), (0,1))
    # # ax3 = plt.subplot2grid((2,2), (1,1))

    # # plot_displacement(X, Z, X_cur, Z_cur, shape, [ax1, ax2, ax3], 'fig1')

    # k = int((N_test - 1) / 2)
    # # k=0
    # middle_layer = int(np.floor(M_test / 2))    
    # # get line in reference configuration
    # ref_x = X[k, middle_layer]
    # ref_y = Y[k, middle_layer]
    # # get line in current configuration
    # cur_x = X_cur[k, middle_layer]
    # cur_y = Y_cur[k, middle_layer]

    # fig, ax = plt.subplots(figsize=(9, 3))
    # ax.plot(cur_x, cur_y)
    # ax.plot(ref_x, ref_y, linestyle='--')
    # plt.show()