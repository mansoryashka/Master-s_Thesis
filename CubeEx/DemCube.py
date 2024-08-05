import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from pathlib import Path

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, dev, MultiLayerNet, L2norm3D, write_vtk_v2

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
    def evaluate_model(self, x, y, z, return_pred_tensor=False):
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

        if return_pred_tensor:
            return U, u_pred_torch, xyz_tensor
        return U

E = 1000
nu = 0.3
mu = E / (2*(1 + nu))

# mu = 15
def energy(u, x, J=False):
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

    detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

    compressibility = kappa * (detF - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / detF * (Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)

    if J:
        return compressibility + neo_hookean + active_stress_energy, detF
    return compressibility + neo_hookean + active_stress_energy

def energy2(u, x, J=False, f0=torch.tensor([1, 0, 0]).to(dev)):
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

    Ff0x = Fxx * f0[0] + Fxy * f0[1] + Fxz * f0[2]
    Ff0y = Fyx * f0[0] + Fyy * f0[1] + Fyz * f0[2]
    Ff0z = Fzx * f0[0] + Fzy * f0[1] + Fzz * f0[2]

    detF  = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2
    I4f = Ff0x * Ff0x + Ff0y * Ff0y + Ff0z * Ff0z

    compressibility = kappa * (detF - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / detF * (I4f - 1)

    return compressibility + neo_hookean + active_stress_energy

""" IMPLEMENT VONMISES STRESS!!!! """
# def VonMises_stress(u, x, f0=torch.tensor([1, 0, 0]).to(dev)):
#     Nx = N_test; Ny = N_test; Nz = N_test

#     duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
#     duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
#     duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.shape[0], 1, device=dev), create_graph=True, retain_graph=True)[0]

#     Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
#     Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
#     Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
#     Fyx = duydxyz[:, 0].unsqueeze(1) + 0
#     Fyy = duydxyz[:, 1].unsqueeze(1) + 1
#     Fyz = duydxyz[:, 2].unsqueeze(1) + 0
#     Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
#     Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
#     Fzz = duzdxyz[:, 2].unsqueeze(1) + 1

#     detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
#     invF11 = (Fyy * Fzz - Fyz * Fzy) / detF
#     invF12 = -(Fxy * Fzz - Fxz * Fzy) / detF
#     invF13 = (Fxy * Fyz - Fxz * Fyy) / detF
#     invF21 = -(Fyx * Fzz - Fyz * Fzx) / detF
#     invF22 = (Fxx * Fzz - Fxz * Fzy) / detF
#     invF23 = -(Fxx * Fyz - Fxz * Fyx) / detF
#     invF31 = (Fyx * Fzy - Fyy * Fzy) / detF
#     invF32 = -(Fxx * Fzy - Fxy * Fzx) / detF
#     invF33 = (Fxx * Fyy - Fxy * Fyx) / detF

#     P11 = mu * Fxx + (lmbd * torch.log(detF) - mu) * invF11
#     P12 = mu * Fxy + (lmbd * torch.log(detF) - mu) * invF21
#     P13 = mu * Fxz + (lmbd * torch.log(detF) - mu) * invF31
#     P21 = mu * Fyx + (lmbd * torch.log(detF) - mu) * invF12
#     P22 = mu * Fyy + (lmbd * torch.log(detF) - mu) * invF22
#     P23 = mu * Fyz + (lmbd * torch.log(detF) - mu) * invF32
#     P31 = mu * Fzx + (lmbd * torch.log(detF) - mu) * invF13
#     P32 = mu * Fzy + (lmbd * torch.log(detF) - mu) * invF23
#     P33 = mu * Fzz + (lmbd * torch.log(detF) - mu) * invF33
    
#     S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
#     S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
#     S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
#     S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
#     S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
#     S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
#     S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
#     S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
#     S33 = invF31 * P13 + invF32 * P23 + invF33 * P33
    
#     S11_pred = S11.detach().cpu().numpy()
#     S12_pred = S12.detach().cpu().numpy()
#     S13_pred = S13.detach().cpu().numpy()
#     S21_pred = S21.detach().cpu().numpy()
#     S22_pred = S22.detach().cpu().numpy()
#     S23_pred = S23.detach().cpu().numpy()
#     S31_pred = S31.detach().cpu().numpy()
#     S32_pred = S32.detach().cpu().numpy()
#     S33_pred = S33.detach().cpu().numpy()
    
#     surS11 = S11_pred.reshape(Ny, Nx, Nz)
#     surS12 = S12_pred.reshape(Ny, Nx, Nz)
#     surS13 = S13_pred.reshape(Ny, Nx, Nz)
#     surS21 = S21_pred.reshape(Ny, Nx, Nz)
#     surS22 = S22_pred.reshape(Ny, Nx, Nz)
#     surS23 = S23_pred.reshape(Ny, Nx, Nz)
#     surS31 = S31_pred.reshape(Ny, Nx, Nz)
#     surS32 = S32_pred.reshape(Ny, Nx, Nz)
#     surS33 = S33_pred.reshape(Ny, Nx, Nz)

#     SVonMises = np.float64(
#                 np.sqrt(0.5 * ((surS11 - surS22) ** 2 
#                             + (surS22 - surS33) ** 2 
#                             + (surS33 - surS11) ** 2 
#                     + 6 * (surS12 ** 2 + surS23 ** 2 + surS31 ** 2)))
#                 )
#     return SVonMises


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
            DemBeam = DeepEnergyMethodCube(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lrs, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k, fb=np.asarray([[0, 0, 0]]))
            # evaluate model
            U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
            # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
            # store solution
            # write_vtk_v2(f'output/DemBeam_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
            write_vtk_v2(f'output/DemBeam_N{N}', x, y, z, U_pred)
            # calculate L2norm
            u_norms[i] = L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
            losses[:, :, i] = np.array(DemBeam.losses).T
    # train on many learning rates and number of neurons in hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_neurons, (list, tuple)):
        print('lrs, num_n')
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(num_neurons)))
        for i, lr in enumerate(lrs):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*num_layers), 3)
                DemBeam = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)

                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k, fb=np.asarray([[0, 0, 0]]))
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)

                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_nn{n}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                write_vtk_v2(f'output/DemBeam_lr{lr}_nn{n}', x, y, z, U_pred)
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on many learning rates and number of hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('lrs, num_l')
        u_norms = np.zeros((len(lrs), len(num_layers)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(num_layers)))
        for i, lr in enumerate(lrs):
            for j, l in enumerate(num_layers):
                model = MultiLayerNet(3, *([num_neurons]*l), 3)
                DemBeam = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs,  eval_data=eval_data, k=k, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                write_vtk_v2(f'output/DemBeam_lr{lr}_nl{l}', x, y, z, U_pred)

                u_norms[i, j] = L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on number of neurons in hidden layers and number of hidden layers
    elif isinstance(num_neurons, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('num_n, num_l')
        u_norms = np.zeros((len(num_layers), len(num_neurons)))
        losses = np.zeros((nr_losses, num_losses, len(num_layers), len(num_neurons)))
        for i, l in enumerate(num_layers):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*l), 3)
                DemBeam = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lrs, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_nn{n}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                write_vtk_v2(f'output/DemBeam_nn{n}_nl{l}', x, y, z, U_pred)
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                
                losses[:, :, i, j] = np.array(DemBeam.losses).T
    # train on many N values and learning rates
    elif isinstance(Ns, (list, tuple)) and isinstance(lrs, (list, tuple)):
        # print(type(Ns), type(lrs), isinstance((Ns and lrs), list), Ns); exit()
        print('Ns and lrs')
        u_norms = np.zeros((len(lrs), len(Ns)))
        losses = np.zeros((nr_losses, num_losses, len(lrs), len(Ns)))
        for j, N in enumerate(Ns):
            shape = [N, N, N]
            for i, lr in enumerate(lrs):
                # define model, DEM and domain
                model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
                DemBeam = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=N)
                # train model
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, lr=lr, max_it=max_it, epochs=num_epochs, eval_data=eval_data, k=k, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemBeam.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                write_vtk_v2(f'output/DemBeam_lr{lr}_N{N}', x, y, z, U_pred)
                # calculate L2norm
                u_norms[i, j] = L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
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
                xticklabels=xticks, yticklabels=yticks, cbar=False,
                vmax=np.max(data[~np.isnan(data)])
                )
    ### skriv tester for om title, labels og filnavn blir sendt inn!!! ###
    ax.set_title(title)
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)
    fig.savefig(figures_path / Path(figname + '.pdf'))

if __name__ == '__main__':
    u_fem20 = np.load(arrays_path / 'u_fem20.npy')
    # print(f'FEM: {L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)}')

    x = np.linspace(0, L, N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]

    x_eval = np.linspace(0, L, N_test + 4)[1:-1]
    y_eval = np.linspace(0, D, N_test + 4)[1:-1]
    z_eval = np.linspace(0, H, N_test + 4)[1:-1]

    # N = 20
    # lr = 0.1
    # shape = [N_test, N_test, N_test]
    # num_layers = [2, 3, 4, 5]
    # num_neurons = [20, 30, 40, 50]
    # num_expreriments = 20
    # num_epochs = 500
    # U_norms = 0
    # losses = 0
    # start = time.time()
    # for i in range(num_expreriments):
    #     U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lr, num_neurons=num_neurons, num_layers=num_layers, num_epochs=num_epochs, shape=shape, eval_data=[x_eval, y_eval, z_eval], k=1)
    #     U_norms += U_norms_i
    #     losses += losses_i
    # # # losses = np.asarray(losses)
    # U_norms /= num_expreriments
    # losses /= num_expreriments
    # print(U_norms)
    # np.save(arrays_path / 'losses_nl_nn', losses)
    # plot_heatmap(U_norms, num_neurons, num_layers, rf'$L^2$ norm of error with N={N} and $\eta$ = {lr}', 'Number of hidden neurons', 'Number of hidden layers', 'cube_heatmap_num_neurons_layers')
    # tid = time.time() - start
    # print(f'tid: {tid:.2f}s')
    # print(f'tid: {tid/60:.2f}m')
    # print(f'tid: {tid/3600:.2f}t')

    N = 20
    shape = [N_test, N_test, N_test]
    lrs = [.01, .05, .1, .5]
    num_neurons = [20, 30, 40, 50]
    num_layers = 3
    num_expreriments = 20
    num_epochs = 500
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=num_epochs, shape=shape, eval_data=[x_eval, y_eval, z_eval], k=1)
        U_norms += U_norms_i
        losses += losses_i
        print(i, U_norms_i)
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')
    U_norms /= num_expreriments
    losses /= num_expreriments
    print(U_norms)
    plot_heatmap(U_norms, num_neurons, lrs, rf'$L^2$ norm of error with N={N} and {num_layers} hidden layers', 'Number of neurons in hidden layers', r'$\eta$', 'cube_heatmap_lrs_num_neurons')
    np.save(arrays_path / 'losses_lrs_nn', losses)

    # shape = [N_test, N_test, N_test]
    # Ns = [10, 20, 30, 40]
    # lrs = [.005, .01, .05, .1, .5]
    # num_neurons = 20
    # num_layers = 3
    # num_expreriments = 20
    # num_epochs = 500
    # U_norms = 0
    # losses = 0
    # start = time.time()
    # for i in range(num_expreriments):
    #     U_norms_i, losses_i = train_and_evaluate(Ns=Ns, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=num_epochs, eval_data=[x_eval, y_eval, z_eval], k=1)
    #     U_norms += U_norms_i
    #     losses += losses_i
    # U_norms /= num_expreriments
    # losses /= num_expreriments
    # np.save(arrays_path / 'losses_lrs_N', losses)
    # plot_heatmap(U_norms, Ns, lrs, rf'$L^2$ norm of error with {num_neurons} hidden neurons and {num_layers} hidden layers', 'N', r'$\eta$', 'cube_heatmap_lrs_N')
    # tid = time.time() - start
    # print(f'tid: {tid:.2f}s')
    # print(f'tid: {tid/60:.2f}m')
    # print(f'tid: {tid/3600:.2f}t')