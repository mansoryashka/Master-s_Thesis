import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
import scipy.integrate as sp
from pyevtk.hl import gridToVTK

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, MultiLayerNet, dev

# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# code to run on ML node with no hangup :
# CUDA_VISIBLE_DEVICES=x TMP=./tmp nohup python yourscript.py > out1.log 2> error1.log &

N_test = 25
x0 = 0
E = 1000
nu = 0.3

L = 4
H = 1
D = 1
LHD = [L, H, D]

dx = L/(4*N_test)
dy = H/N_test
dz = D/N_test

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = L
n_cond = [0, -5, 0]

def define_domain(L, H, D, N=25):
    x = np.linspace(0, L, int(4*N))
    y = np.linspace(0, H, N)
    z = np.linspace(0, D, N)

    Xm, Ym, Zm = np.meshgrid(x, y, z) 
    Xm = np.expand_dims(Xm.flatten(), 1)
    Ym = np.expand_dims(Ym.flatten(), 1)
    Zm = np.expand_dims(Zm.flatten(), 1)
    domain = np.concatenate((Xm, Ym, Zm), axis=-1)

    db_idx = np.where(Xm == d_boundary)[0]
    db_pts = domain[db_idx, :]
    db_vals = np.ones(np.shape(db_pts)) * d_cond

    nb_idx = np.where(Xm == n_boundary)[0]
    nb_pts = domain[nb_idx, :]
    nb_vals = np.ones(np.shape(nb_pts)) * n_cond

    db_pts_x, db_pts_y, db_pts_z = db_pts.T
    nb_pts_x, nb_pts_y, nb_pts_z = nb_pts.T

    fig = plt.figure(figsize=(5,3))
    # fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xm, Ym, Zm, facecolor='tab:blue', s=0.005)
    ax.scatter(db_pts_x, db_pts_y, db_pts_z, facecolor='tab:green', s=0.5)
    ax.scatter(nb_pts_x, nb_pts_y, nb_pts_z, facecolor='tab:red', s=0.5)
    ax.set_box_aspect((4,1,1))
    ax.set_xticks([0.0, 2.0, 4.0])
    ax.set_zticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=20, azim=-75)
    # plt.show()
    plt.close()
    # exit()

    dirichlet = {
        'coords': db_pts,
        'values': db_vals
    }

    neumann = {
        'coords': nb_pts,
        'values': nb_vals
    }

    return domain, dirichlet, neumann

lmbd =  E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))

def energy(u, x):
    ### energy frunction from DEM paper ### 
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
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2
    strainEnergy = 0.5 * lmbd * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
    return strainEnergy


""" Simspson's method to be implemented later

def basic_simps(f, dx=1, axis=-1):
    def _sliced_tuple(x, offset=0):
        "Slice desired dimension of tensor"
        x = np.array(x)
        x[axis] = slice(0, f.shape[axis] - offset, 2)
        x = tuple(x)
        return x
    
    n_dims = len(f.shape)

    slice_all = (slice(None),) * n_dims
    s0 = slice_all
    s1 = slice_all
    s2 = slice_all

    s0 = _sliced_tuple(s0, 0)
    s1 = _sliced_tuple(s1, 1)
    s2 = _sliced_tuple(s2, 2)

    return np.sum(1/3 * dx * (f[s0] + 4 * f[s1] + f[s2]), axis)
"""

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

def L2norm(U):
    Ux = np.expand_dims(U[0].flatten(), 1)
    Uy = np.expand_dims(U[1].flatten(), 1)
    Uz = np.expand_dims(U[2].flatten(), 1)
    Uxyz = np.concatenate((Ux, Uy, Uz), axis=1)
    n = Ux.shape[0]
    udotu = np.zeros(n)
    for i in range(n):
        udotu[i] = np.dot(Uxyz[i,:], Uxyz[i,:].T)
    udotu = udotu.reshape(4*N_test, N_test, N_test)
    L2norm = np.sqrt(sp.simps(sp.simps(sp.simps(udotu, dx=dz), dx=dy), dx=dx))
    return L2norm

def train_and_evaluate(Ns=20, lrs=0.1, num_neurons=20, num_layers=2, num_epochs=40, max_it=20):
    if isinstance(Ns, (list, tuple)):
        u_norms = np.zeros(len(Ns))
        for i, N in enumerate(Ns):
            # define model, DEM and domain
            model = MultiLayerNet(3, [num_neurons]*num_layers, 3)
            DemBeam = DeepEnergyMethod(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lrs, max_it=max_it, num_epochs=num_epochs)
            # evaluate model
            U_pred = DemBeam.evaluate_model(x, y, z)
            # calculate L2norm
            u_norms[i] = L2norm(U)
    elif isinstance((lrs and num_neurons), (list, tuple)):
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        pass
    elif isinstance((lrs and num_layers), (list, tuple)):
        u_norms = np.zeros((len(lrs), len(num_layers)))
        pass
    elif isinstance((num_neurons and num_layers), (list, tuple)):
        u_norms = np.zeros((len(num_neurons), len(num_layers)))
        pass
    else:
        raise Exception('You have to provide a list of N values or one of the following:\n' + 
                        '\t- lrs AND num_neurons\n\t- lrs AND num_layers\n\t- num_neurons AND num_layers')
    return u_norms

if __name__ == '__main__':
    u_fem = np.load('stored_arrays/u_fem.npy')
    # print(u_fem.shape)
    print(f'FEM: {L2norm(u_fem):8.5f} \n')
    # exit()

    # x = rng.random(size=4*N_test)
    # y = rng.random(size=N_test)
    # z = rng.random(size=N_test)
    # x = L*np.sort(x); y = H*np.sort(y); z = D*np.sort(z)
    x = np.linspace(0, L, 4*N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    x = np.linspace(0, H, N_test + 2)[1:-1]
    Ns = np.array([10, 20, 30])

    U_norms = train_and_evaluate(Ns, num_epochs=2)
    print(U_norms)



    exit()
    hidden_dim = np.array([30, 50])#, 30, 40])
    max_epoch = 10
    losses = np.zeros((len(N_ar), len(hidden_dim)))
    L2norms = np.zeros((len(N_ar), len(hidden_dim)))
    tot_losses = []
    best_norm = np.inf

    import time
    num_experiments = 1
    for i in range(len(N_ar)):
        for j in range(len(hidden_dim)):
            for _ in range(num_experiments):
                print(dev)
                dom, dirichlet, neumann = domain(l, h ,d, N_ar[i])

                model = MultiLayerNet(3, hidden_dim[j], hidden_dim[j], hidden_dim[j], 3)
                DemBeam = DeepEnergyMethod(model, energy)
                start = time.time()
                DemBeam.train_model(dom, dirichlet, neumann, [l, h, d], epochs=max_epoch)
                print(time.time()-start)
                
                # print(DemBeam.losses[max_epoch])
                U = DemBeam.evaluate_model(x, y, z)

                Udem = np.array(U).copy()
                np.save(f'stored_arrays/u_dem_{i}{j}', Udem)

                error_norm = (L2norm(Udem) - L2norm(u_fem))/L2norm(u_fem)

                if abs(error_norm) < abs(best_norm):
                    best_norm = error_norm
                    Ubest = U
                    print(f"Best N: {N_ar[i]:2d}, nr. neurons: {hidden_dim[j]:2d}")

                tot_losses.append(DemBeam.losses)
                losses[i, j] += DemBeam.losses[max_epoch]
                L2norms[i, j] += error_norm

            print(f"N: {N_ar[i]}, nr. hidden neurons: {hidden_dim[j]}")
            print(f'\nDEM: {L2norm(Udem):8.5f}   FEM: {L2norm(u_fem):8.5f}')
            print(f'L2norm = {error_norm:.5f}\n')

    losses /= num_experiments
    L2norms /= num_experiments

    print('losses')
    print(losses)
    print('L2norms')
    print(L2norms)
    np.save('stored_arrays/3Dlosses', losses)
    np.save('stored_arrays/L2norms', L2norms)
    tot_losses = np.array(tot_losses)
    np.save('stored_arrays/tot_losses', tot_losses)
    write_vtk_v2('output/NeoHook3D', x, y, z, Ubest)


    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(u_fem[0], u_fem[1], u_fem[2], s=0.002)
    # ax.scatter(U[0], U[1], U[2], s=0.002, c='tab:red')
    # plt.show()
    