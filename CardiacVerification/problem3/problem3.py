import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350


import sys
sys.path.insert(0, "../problem2")
sys.path.insert(0, "../../")
from problem2 import DeepEnergyMethodLV, write_vtk_v3
from DEM import MultiLayerNet, dev
from EnergyModels import GuccioneTransverseEnergyModel

print(dev)
plt.style.use('default')
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

C = 10E3
bf = bt = bfs = 1

rs_endo = 7
rl_endo = 17

rs_epi = 10
rl_epi = 20

def normalize(u):
    return u / np.linalg.norm(u, axis=0, keepdims=True)


def generate_fibers(N=15, M=3):
    t = np.linspace(0, 1, M)
    alpha_endo = 90
    alpha_epi = -90

    alpha = lambda t: (alpha_endo + (alpha_epi - alpha_endo) * t) * (np.pi / 180)
    r_long = lambda t: (rl_endo + (rl_epi - rl_endo) * t)
    r_short = lambda t: (rs_endo + (rs_epi - rs_endo) * t)

    rl = r_long(t).reshape((1, M, 1))
    rs = r_short(t).reshape((1, M, 1))
    al = alpha(t).reshape((1, M, 1))

    # drl_dt = rl_epi - rl_endo
    # drs_dt = rs_epi - rs_endo

    # a = np.sqrt(x**2 + y**2) / rs
    # b = z / rl

    # mu = -np.arctan2(a, b)
    # theta = np.pi - np.arctan(x, -y)
    # theta[mu > 1e-7] = 0.0

    mu = np.linspace(-np.pi, -np.arccos(5/20), N).reshape((1, 1, N))
    theta = np.linspace(-np.pi, np.pi, N).reshape((N, 1, 1))
    # print(theta)


    # e_t = np.array(
    #     [
    #         drs_dt * np.sin(mu) * np.cos(theta),
    #         drs_dt * np.sin(mu) * np.sin(theta),
    #         drl_dt * np.cos(mu),
    #     ],
    # )
    # e_t = normalize(e_t)

    e_mu = np.array(
        [
            rs * np.cos(mu) * np.cos(theta),
            rs * np.cos(mu) * np.sin(theta),
            -rl * np.ones_like(theta) * np.sin(mu),
        ],
    )
    e_mu = normalize(e_mu)

    e_theta = np.array(
        [
            -rs * np.sin(mu) * np.sin(theta),
            rs * np.sin(mu) * np.cos(theta),
            np.zeros_like(e_mu[0]),
        ],
    )
    e_theta = normalize(e_theta)

    f0 = np.sin(al) * e_mu + np.cos(al) * e_theta
    f0 = normalize(f0)

    n0 = np.cross(e_mu, e_theta, axis=0)
    n0 = normalize(n0)

    s0 = np.cross(f0, n0, axis=0)
    s0 = normalize(s0)

    # f0[:, -1] = 0.0
    # s0[:, -1] = 0.0
    # n0[:, -1] = 0.0
    # set apex to zero or one?
    # f0[..., 0] = 0
    # s0[..., 0] = 0
    # n0[..., 0] = 0
    # f0[..., -1] = 0
    # s0[..., -1] = 0
    # n0[..., -1] = 0
    # print(f0.shape, s0.shape, n0.shape)
    # print(np.max(f0), np.min(f0))
    # print(np.max(s0), np.min(s0))
    # print(np.max(n0), np.min(n0))
    # return f0.reshape((3, -1, 1)), s0.reshape((3, -1, 1)), n0.reshape((3, -1, 1))
    return f0, s0, n0

def define_domain(N=15, M=5):
    # N=N+1
    middle = int(N/2 + 1)
    print(middle)
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

    u = np.linspace(u_endo, u_epi, N)
    u = u.T[:, middle]

    v = np.linspace(-np.pi, np.pi, N)
    rs = np.linspace(rs_endo, rs_epi, M)
    rl = np.linspace(rl_endo, rl_epi, M)

    RS = np.ones((N, M, N))
    RL = np.ones((N, M, N))
    for i in range(M):
        # print(f'fra: {int(N*K/M)*i} til: {int(N*K/M)*(i+1)-1, 1}')
        RS[:, i] = rs[i]
        RL[:, i] = rl[i]

    x = RS*np.expand_dims(np.outer(np.cos(v), np.sin(u)), 1)
    y = RS*np.expand_dims(np.outer(np.sin(v), np.sin(u)), 1)
    z = RL*np.expand_dims(np.outer(np.ones(np.size(v)), np.cos(u)), 1)

    f, _, _ = generate_fibers(N, M)
    # print(x.shape)
    """ Finn ut hvorfor max(z) = 5.10!!! """
    # set z_max to 5
    # z = np.where(np.abs(z - 5) < .5, 5, z)
    # z = np.where(np.abs(z - 5E-3) < 1E-3, 5E-3, z)

    # define Dirichlet and Neumann BCs
    dir_BC = 5.0
    # dir_BC = lambda z: np.abs(z - 5E-3) < 5E-4
    neu_BC = RS == rs_endo

    # define inner points
    x0 = np.copy(x)
    y0 = np.copy(y)
    z0 = np.copy(z)

    z0[..., -1] = dir_BC

    # define points on Dirichlet boundary
    x1 = x0[..., -1]
    y1 = y0[..., -1]
    z1 = z0[..., -1]

    # define points on Neumann boundary
    x2 = x0[neu_BC]
    y2 = y0[neu_BC]
    z2 = z0[neu_BC]


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
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    # ax.scatter(x0, y0, z0, s=.1, alpha=0.05, c='tab:blue')
    # ax.scatter(x1, y1, z1, s=5, c='tab:green')
    # ax.scatter(x2, y2, z2, s=5, c='tab:red')

    ax.quiver(x0, y0, z0, 
              f[0], 
              f[1], 
              f[2])


    # ax.scatter(x0[:5, :, :], y0[:5, :, :], z0[:5, :, :], s=1, c='tab:blue')
    # ax.quiver(x0[0, :, :], y0[0, :, :], z0[0, :, :], 
    #           f[0][0, :, :], 
    #           f[1][0, :, :], 
    #           f[2][0, :, :])
    # ax.quiver(x0[-1, :, :], y0[-1, :, :], z0[-1, :, :], 
    #           f[0][-1, :, :], 
    #           f[1][-1, :, :], 
    #           f[2][-1, :, :])

    

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

if __name__ == '__main__':
    N = 51; M = 5
    shape = [N, M, N]

    rs_endo =  7
    rl_endo = 17
    rs_epi =  10
    rl_epi =  20

    u_endo = np.linspace(-np.pi, -np.arccos(5/17), N)
    u_epi = np.linspace(-np.pi, -np.arccos(5/20), N)
    middle = int(N/2 + 1)
    u = np.linspace(u_endo, u_epi, N)
    u = u.T[:, middle]
    v = np.linspace(-np.pi, np.pi, N)
    rs = np.linspace(rs_endo, rs_epi, M).reshape(M)
    rl = np.linspace(rl_endo, rl_epi, M).reshape(M)

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

    dx = rs_endo * (v[1] - v[0])
    dy = rs[1] - rs[0]
    dz = ((rl_epi + rs_epi) / 2 + (rl_endo + rs_endo) / 2) / 2 * (u[1] - u[0])
    dxdydz = np.asarray([dx, dy, dz])

    domain, dirichlet, neumann = define_domain(N, M)
    # f0, s0, n0 = generate_fibers(N, M)
    # f0 = torch.tensor(f0.reshape((3, -1, 1))).to(dev)
    # s0 = torch.tensor(s0.reshape((3, -1, 1))).to(dev)
    # n0 = torch.tensor(n0.reshape((3, -1, 1))).to(dev)
    f0 = torch.tensor([1, 0, 0]).to(dev)
    s0 = torch.tensor([0, 1, 0]).to(dev)
    n0 = torch.tensor([0, 0, 1]).to(dev)

    model = MultiLayerNet(3, 60, 60, 60, 60, 3)
    energy = GuccioneTransverseEnergyModel(C, bf, bt, bfs , f0=f0, s0=s0, n0=n0)
    DemLV = DeepEnergyMethodLV(model, energy)
    DemLV.train_model(domain, dirichlet, neumann, shape=shape, LHD=None, dxdydz=dxdydz, neu_axis=[0, 2], lr=.5, epochs=20, fb=np.array([[0, 0, 0]]))
    U_pred = DemLV.evaluate_model(x, y, z)
    write_vtk_v3('output/DemLV', x, y, z, U_pred)