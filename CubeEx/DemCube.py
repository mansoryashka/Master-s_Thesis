import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from pyevtk.hl import gridToVTK
from pathlib import Path

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, dev, MultiLayerNet, L2norm3D

torch.manual_seed(2023)
rng = np.random.default_rng(2023)
# dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'

N_test = 20
L = H = D = 1.0
LHD = [L, H, D]
dx = dy = dz = L/N_test

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = L
n_cond = [-0.5, 0, 0]

def define_domain(L, H, D, N=25):
    x = np.linspace(0, L, N)
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
    # ax.set_box_aspect((4,1,1))
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=20, azim=-75)
    plt.close()
    # plt.show()
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

class DeepEnergyMethodCube(DeepEnergyMethod):
    def evaluate_model(self, x, y, z):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        z1D = zGrid.flatten()
        xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
        xyz_tensor = torch.from_numpy(xyz).float().to(dev)
        xyz_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xyz_tensor)
        u_pred_torch = self.getU(self.model, xyz_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
        surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
        surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        return U

E = 1000
nu = 0.3
mu = E / (2*(1 + nu))

# mu = 15
def energy(u, x):
    kappa = 1e3
    Ta = 1.0

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

    J = detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

    compressibility = kappa * (J - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / J * (Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)

    return compressibility + neo_hookean + active_stress_energy

def energy2(u, x):
    kappa = 1e3
    Ta = 1.0

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

    J = detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

    compressibility = kappa * (J - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / J * (Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)

    return compressibility + neo_hookean + active_stress_energy


### Skrive blokkene til en egen funksjon? Kalles på helt likt inne i loopene ###
def train_and_evaluate(Ns=20, lrs=0.1, num_neurons=20, num_layers=2, num_epochs=40, max_it=20, shape=[20, 20, 20], eval_data=None, k=5):
    num_losses = int(num_epochs / k) + 1
    if eval_data:
        nr_losses = 2
    else:
        nr_losses = 1
    # train on many N values
    if isinstance((Ns and not lrs), (list, tuple)):
        # print('Ns')
        u_norms = np.zeros(len(Ns))
        losses = np.zeros((nr_losses, num_losses, len(Ns)))
        for i, N in enumerate(Ns):
            # define model, DEM and domain
            model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
            DemBeam = DeepEnergyMethod(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lrs, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k)
            # evaluate model
            U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
            VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
            # store solution
            write_vtk_v2(f'output/DemBeam_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
            # calculate L2norm
            u_norms[i] = L2norm3D(U_pred - u_fem20, 4*N_test, N_test, N_test, dx, dy, dz)
            losses[:, :, i] = np.array(DemBeam.losses).T
    # train on many learning rates and number of neurons in hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_neurons, (list, tuple)):
        print('lrs, num_n')
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(num_neurons)))
        for i, lr in enumerate(lrs):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*num_layers), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)

                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k)
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)

                # store solution
                write_vtk_v2(f'output/DemBeam_lr{lr}_nn{n}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, 4*N_test, N_test, N_test, dx, dy, dz)
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on many learning rates and number of hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('lrs, num_l')
        u_norms = np.zeros((len(lrs), len(num_layers)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(num_layers)))
        for i, lr in enumerate(lrs):
            for j, l in enumerate(num_layers):
                model = MultiLayerNet(3, *([num_neurons]*l), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs,  eval_data=eval_data, k=k)
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                write_vtk_v2(f'output/DemBeam_lr{lr}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})

                u_norms[i, j] = L2norm3D(U_pred - u_fem20, 4*N_test, N_test, N_test, dx, dy, dz)
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on number of neurons in hidden layers and number of hidden layers
    elif isinstance(num_neurons, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('num_n, num_l')
        u_norms = np.zeros((len(num_layers), len(num_neurons)))
        losses = np.zeros((nr_losses, num_losses, len(num_layers), len(num_neurons)))
        for i, l in enumerate(num_layers):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*l), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lrs, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k)
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                write_vtk_v2(f'output/DemBeam_nn{n}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, 4*N_test, N_test, N_test, dx, dy, dz)
                
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on many N values and learning rates
    elif isinstance(Ns, (list, tuple)) and isinstance(lrs, (list, tuple)):
        # print(type(Ns), type(lrs), isinstance((Ns and lrs), list), Ns); exit()
        print('Ns and lrs')
        u_norms = np.zeros((len(lrs), len(Ns)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(Ns)))
        for j, N in enumerate(Ns):
            shape = [4*N, N, N]
            for i, lr in enumerate(lrs):
                # define model, DEM and domain
                model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=N)
                # train model
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k)
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                write_vtk_v2(f'output/DemBeam_lr{lr}_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # calculate L2norm
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, 4*N_test, N_test, N_test, dx, dy, dz)
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    else:
        raise Exception('You have to provide a list of N values or one of the following:\n' + 
                        '\t- lrs AND num_neurons\n\t- lrs AND num_layers\n\t- num_neurons AND num_layers')
    return u_norms, losses

def plot_heatmap(data, xparameter, yparameter, title, xlabel, ylabel, figname, cmap='cividis', data_max=1):
    fig, ax = plt.subplots(figsize=(5,5))
    xticks = [str(i) for i in xparameter]
    yticks = [str(j) for j in yparameter]
    sns.heatmap(data, annot=True, ax=ax, cmap=cmap,
                xticklabels=xticks, yticklabels=yticks,
                cbar=False, vmax=np.max(data[data < data_max]))
    ### skriv tester for om title, labels og filnavn blir sendt inn!!! ###
    ax.set_title(title)
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)
    fig.savefig(figures_path / Path(figname + '.pdf'))

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

if __name__ == '__main__':
    u_fem20 = np.load(arrays_path / 'u_fem20.npy')
    print(f'FEM: {L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)}')

    x = np.linspace(0, L, N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]

    N = 20
    lrs = 0.9
    num_layers = [2, 3, 4, 5]
    num_neurons = [20, 30, 40, 50]
    num_expreriments = 3
    U_norms = 0
    for i in range(num_expreriments):
        U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=5)
    U_norms /= num_expreriments
    e_norms = (U_norms - L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)) / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)
    # plot_heatmap(e_norms, num_neurons, num_layers, rf'$L^2$ error norm with N={N} and $\eta$ = {lrs}', 'Number of hidden neurons', 'Number of hidden layers', 'cube_heatmap_nn_nl')
    # plot_heatmap(np.abs(e_norms), num_neurons, lrs, rf'$L^2$ error norm with N={N} and $\eta$ = {lrs}', 'Number of layers', r'$\eta$', 'cube_heatmap_nn_nlABS')
    # print(U_norms)
    # print(e_norms)

    # N = 20
    # lrs = [.05, .1, .5, .9]
    # num_layers = 3
    # num_neurons = [20, 30, 40, 50]
    # num_expreriments = 30
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=60)
    # U_norms /= num_expreriments
    # e_norms = (U_norms - L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)) / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)
    # plot_heatmap(e_norms, num_neurons, lrs, rf'$L^2$ error norm with N={N} and {num_layers} hidden layers', 'Number of neurons in hidden layers', r'$\eta$', 'cube_heatmap_lrs_num_neurons')
    # plot_heatmap(np.abs(e_norms), num_neurons, lrs, rf'$L^2$ error norm with N={N} and {num_layers} hidden layers', 'Number of neurons in hidden layers', r'$\eta$', 'cube_heatmap_lrs_num_neuronsABS')
    # print(U_norms)
    # print(e_norms)

    # N = [20, 30, 40, 50]
    # lr = 0.5
    # num_layers = 3
    # num_neurons = [10, 20, 30, 40, 50]
    # num_expreriments = 30
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=N, lrs=lr, num_neurons=num_neurons, num_layers=num_layers, num_epochs=60)
    # U_norms /= num_expreriments
    # e_norms = (U_norms - L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)) / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)
    # plot_heatmap(e_norms, num_neurons, N, rf'$L^2$ error norm with $\eta$ = {lr} and {num_layers} hidden layers', 'Number of neurons in hidden layers', 'N', 'cube_heatmap_N_num_neurons')
    # plot_heatmap(np.abs(e_norms), num_neurons, N, rf'$L^2$ error norm with $\eta$ = {lr} and {num_layers} hidden layers', 'Number of neurons in hidden layers', 'N', 'cube_heatmap_N_num_neuronsABS')