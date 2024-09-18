import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, MultiLayerNet, L2norm3D, dev, write_vtk_v2
from EnergyModels import NeoHookeanEnergyModel

# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'
# code to run on ML node with no hangup:
# CUDA_VISIBLE_DEVICES=x TMP=./tmp nohup python yourscript.py > out1.log 2> error1.log &


# setup
E = 1000
nu = 0.3

lmbda =  E * nu / ((1 + nu) * (1 - 2*nu))
mu = E / (2 * (1 + nu))

L = 4
H = 1
D = 1
LHD = [L, H, D]

N_test = 30
x0 = 0

dx = L / (4*N_test - 1)
dy = H / (N_test - 1)
dz = D / (N_test - 1)

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = L
n_cond = [0, 0, 0]

dir_bcs = {d_boundary: d_cond}
neu_bcs = {n_boundary: n_cond}

def define_domain(L, H, D, N=25):
    x = np.linspace(0, L, 4*N)
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

    nb_idx = np.where(Xm == n_boundary)[0]
    nb_pts = domain[nb_idx, :]
    nb_vals = np.ones(np.shape(nb_pts)) * n_cond

    db_pts_x, db_pts_y, db_pts_z = db_pts.T
    nb_pts_x, nb_pts_y, nb_pts_z = nb_pts.T

    
    if not Path(current_path / 'domain.png').exists():
    # if True:
        fig = plt.figure(figsize=(5, 3))
        fig.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(Xm, Ym, Zm, facecolor='tab:blue', s=0.5, alpha=0.1)
        ax.scatter(db_pts_x, db_pts_y, db_pts_z, facecolor='tab:green', s=0.5)
        # ax.scatter(nb_pts_x, nb_pts_y, nb_pts_z, facecolor='tab:red', s=0.5)
        ax.set_box_aspect((4, 1, 1))
        ax.set_xticks([0.0, 4.0])
        ax.set_zticks([0.0, 1.0])
        ax.set_yticks([0.0, 1.0])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.view_init(elev=25, azim=-55)

        fig.savefig('domain.png')
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

class DeepEnergyMethodBeam(DeepEnergyMethod):
    def evaluate_model(self, x, y, z, return_pred_tensor=False):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)

        x1D = np.expand_dims(xGrid.flatten(), 1)
        y1D = np.expand_dims(yGrid.flatten(), 1)
        z1D = np.expand_dims(zGrid.flatten(), 1)
        xyz = np.concatenate((x1D, y1D, z1D), axis=-1)

        xyz_tensor = torch.from_numpy(xyz).float().to(dev)
        xyz_tensor.requires_grad_(True)

        u_pred_torch = self(self.model, xyz_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()

        surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
        surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
        surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)

        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        # print(np.asarray(U).shape); exit()
        if return_pred_tensor:
            return U, u_pred_torch, xyz_tensor
        return U

def VonMises_stress(u, x):
    Nx = 4*N_test; Ny = N_test; Nz = N_test
    duxdxyz = grad(u[:, 0].unsqueeze(1), x, 
                    torch.ones(x.shape[0], 1, device=dev), 
                    create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, 
                    torch.ones(x.shape[0], 1, device=dev), 
                    create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, 
                    torch.ones(x.shape[0], 1, device=dev), 
                    create_graph=True, retain_graph=True)[0]

    Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
    Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
    Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
    Fyx = duydxyz[:, 0].unsqueeze(1) + 0
    Fyy = duydxyz[:, 1].unsqueeze(1) + 1
    Fyz = duydxyz[:, 2].unsqueeze(1) + 0
    Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
    Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
    Fzz = duzdxyz[:, 2].unsqueeze(1) + 1

    detF = (Fxx * (Fyy * Fzz - Fyz * Fzy) 
          - Fxy * (Fyx * Fzz - Fyz * Fzx) 
          + Fxz * (Fyx * Fzy - Fyy * Fzx))
    
    invF11 =  (Fyy * Fzz - Fyz * Fzy) / detF
    invF12 = -(Fxy * Fzz - Fxz * Fzy) / detF
    invF13 =  (Fxy * Fyz - Fxz * Fyy) / detF
    invF21 = -(Fyx * Fzz - Fyz * Fzx) / detF
    invF22 =  (Fxx * Fzz - Fxz * Fzy) / detF
    invF23 = -(Fxx * Fyz - Fxz * Fyx) / detF
    invF31 =  (Fyx * Fzy - Fyy * Fzy) / detF
    invF32 = -(Fxx * Fzy - Fxy * Fzx) / detF
    invF33 =  (Fxx * Fyy - Fxy * Fyx) / detF

    P11 = mu * Fxx + (lmbda * torch.log(detF) - mu) * invF11
    P12 = mu * Fxy + (lmbda * torch.log(detF) - mu) * invF21
    P13 = mu * Fxz + (lmbda * torch.log(detF) - mu) * invF31
    P21 = mu * Fyx + (lmbda * torch.log(detF) - mu) * invF12
    P22 = mu * Fyy + (lmbda * torch.log(detF) - mu) * invF22
    P23 = mu * Fyz + (lmbda * torch.log(detF) - mu) * invF32
    P31 = mu * Fzx + (lmbda * torch.log(detF) - mu) * invF13
    P32 = mu * Fzy + (lmbda * torch.log(detF) - mu) * invF23
    P33 = mu * Fzz + (lmbda * torch.log(detF) - mu) * invF33
    
    S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
    S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
    S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
    S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
    S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
    S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
    S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
    S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
    S33 = invF31 * P13 + invF32 * P23 + invF33 * P33
    
    S11_pred = S11.detach().cpu().numpy()
    S12_pred = S12.detach().cpu().numpy()
    S13_pred = S13.detach().cpu().numpy()
    S21_pred = S21.detach().cpu().numpy()
    S22_pred = S22.detach().cpu().numpy()
    S23_pred = S23.detach().cpu().numpy()
    S31_pred = S31.detach().cpu().numpy()
    S32_pred = S32.detach().cpu().numpy()
    S33_pred = S33.detach().cpu().numpy()
    
    surS11 = S11_pred.reshape(Ny, Nx, Nz)
    surS12 = S12_pred.reshape(Ny, Nx, Nz)
    surS13 = S13_pred.reshape(Ny, Nx, Nz)
    surS21 = S21_pred.reshape(Ny, Nx, Nz)
    surS22 = S22_pred.reshape(Ny, Nx, Nz)
    surS23 = S23_pred.reshape(Ny, Nx, Nz)
    surS31 = S31_pred.reshape(Ny, Nx, Nz)
    surS32 = S32_pred.reshape(Ny, Nx, Nz)
    surS33 = S33_pred.reshape(Ny, Nx, Nz)

    SVonMises = np.float64(np.sqrt(0.5 * 
                             ((surS11 - surS22) ** 2 
                            + (surS22 - surS33) ** 2 
                            + (surS33 - surS11) ** 2 
                    + 6 * (surS12 ** 2 + surS23 ** 2 
                           + surS31 ** 2)))
                )
    return SVonMises

### Skrive blokkene til en egen funksjon? Kalles på helt likt inne i loopene ###
def train_and_evaluate(Ns=20, lrs=0.1, num_neurons=20, num_layers=2, num_epochs=40, shape=[20, 20, 20]):
    # if eval_data:
    #     nr_losses = 2
    # else:
        # nr_losses = 1
    nr_losses = 1
    energy = NeoHookeanEnergyModel(lmbda, mu)
    # train on many N values
    if isinstance((Ns and not lrs), (list, tuple)):
        # print('Ns')
        u_norms = np.zeros(len(Ns))
        losses = torch.zeros((nr_losses, num_epochs, len(Ns)))
        for i, N in enumerate(Ns):
            # define model, DEM and domain
            model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
            DemBeam = DeepEnergyMethodBeam(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lrs, epochs=num_epochs)
            # evaluate model
            U_pred = DemBeam.evaluate_model(x, y, z)
            # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
            # store solution
            # write_vtk_v2(f'output/DemBeam_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
            # calculate L2norm

            u_norms[i] = (L2norm3D(U_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
                        / L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))
            # losses[:, :, i] = np.array(DemBeam.losses.detach().numpy()).T
            losses[:, :, i] = DemBeam.losses
            del model
    # train on many learning rates and number of neurons in hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_neurons, (list, tuple)):
        print('lrs, num_n')
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        losses = torch.zeros((nr_losses, num_epochs, len(lrs), len(num_neurons)))
        for i, lr in enumerate(lrs):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*num_layers), 3)
                DemBeam = DeepEnergyMethodBeam(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)

                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs)
                U_pred = DemBeam.evaluate_model(x, y, z)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)

                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_nn{n}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                
                u_norms[i, j] = (L2norm3D(U_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))
                # losses[:, :, i, j] = np.array(DemBeam.losses.detach().numpy()).T
                losses[:, :, i, j] = DemBeam.losses
                del model
    # train on many learning rates and number of hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('lrs, num_l')
        u_norms = np.zeros((len(lrs), len(num_layers)))
        losses = torch.zeros((nr_losses, num_epochs, len(lrs), len(num_layers)))
        for i, lr in enumerate(lrs):
            for j, l in enumerate(num_layers):
                model = MultiLayerNet(3, *([num_neurons]*l), 3)
                DemBeam = DeepEnergyMethodBeam(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})

                
                u_norms[i, j] = (L2norm3D(U_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))
                # losses[:, :, i, j] = np.array(DemBeam.losses.detach().numpy()).T
                losses[:, :, i, j] = DemBeam.losses
                del model
    # train on number of neurons in hidden layers and number of hidden layers
    elif isinstance(num_neurons, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('num_n, num_l')
        u_norms = np.zeros((len(num_layers), len(num_neurons)))
        losses = torch.zeros((nr_losses, num_epochs, len(num_layers), len(num_neurons)))
        for i, l in enumerate(num_layers):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*l), 3)
                # print(id(model))
                DemBeam = DeepEnergyMethodBeam(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lrs, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_nn{n}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                
                u_norms[i, j] = (L2norm3D(U_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))

                # losses[:, :, i, j] = np.array(DemBeam.losses.detach().numpy()).T
                losses[:, :, i, j] = DemBeam.losses
                del model
    # train on many N values and learning rates
    elif isinstance(Ns, (list, tuple)) and isinstance(lrs, (list, tuple)):
        # print(type(Ns), type(lrs), isinstance((Ns and lrs), list), Ns); exit()
        print('Ns and lrs')
        u_norms = np.zeros((len(lrs), len(Ns)))
        losses = torch.zeros((nr_losses, num_epochs, len(lrs), len(Ns)))
        for j, N in enumerate(Ns):
            shape = [4*N, N, N]
            for i, lr in enumerate(lrs):
                # define model, DEM and domain
                model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
                DemBeam = DeepEnergyMethodBeam(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=N)
                # train model
                DemBeam.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemBeam_lr{lr}_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # calculate L2norm
                
                u_norms[i, j] = (L2norm3D(U_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))
                # losses[:, :, i, j] = np.array(DemBeam.losses.detach().numpy()).T
                losses[:, :, i, j] = DemBeam.losses
                del model
    else:
        raise Exception('You have to provide a list of N values or one of the following:\n' + 
                        '\t- lrs AND num_neurons\n\t- lrs AND num_layers\n\t- num_neurons AND num_layers')
    return u_norms, losses


def plot_heatmap(data, xparameter, yparameter, title, xlabel, ylabel, figname, cmap='cividis'):
    fig, ax = plt.subplots(figsize=(5,5))
    xticks = [str(i) for i in xparameter]
    yticks = [str(j) for j in yparameter]
    sns.heatmap(data, annot=True, ax=ax, cmap=cmap,
                xticklabels=xticks, yticklabels=yticks, cbar=False,
                vmax=np.max(data[~np.isnan(data)])
                )
    ### skriv tester for om title, labels og filnavn blir sendt inn!!! ###
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(figures_path / Path(figname + '.pdf'))


def plot_losses(losses, parameter1, parameter2, dim1, dim2, num_epochs, title, filename):
# def plot_losses(losses, dim1, dim2, num_epochs, title, filename):
    ### find out how to use different markers for each line ###
    losses += np.abs(np.min(losses[~np.isnan(losses)])) + 1
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(np.arange(300, num_epochs+1, 1), losses[300:, dim1, :])
    ax2.semilogy(np.arange(300, num_epochs+1, 1), losses[300:, :, dim2])
    ax1.legend([f'{parameter1[0]} = {nn}' for nn in parameter1[1]])
    ax2.legend([f'{parameter2[0]} = {lr}' for lr in parameter2[1]])
    # ax1.set_ylim(bottom=0.0001)
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel(r'$L^2$-error norm')
    ax2.set_ylabel(r'$L^2$-error norm')
    ax1.set_title(rf'$L^2$-error norm with {title[0]}')
    ax2.set_title(rf'$L^2$-error norm with {title[1]}')
    fig.savefig(figures_path / Path(filename + '.pdf'))

def run1():
    N = 30
    lr = .1
    shape = [4*N, N, N]
    num_layers = [2, 3, 4, 5]
    num_neurons = [30, 40, 50, 60]
    num_expreriments = 10
    num_epochs = 300
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print(f'Experiment nr. {i}')
        U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lr,
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs, 
                                                 shape=shape)
        print('U_norms_i')
        print(U_norms_i)
        U_norms += U_norms_i
        losses += losses_i
    print('U_norms')
    print(U_norms)
    U_norms /= num_expreriments
    losses /= num_expreriments
    losses = losses.detach().numpy()
    np.save(arrays_path / 'losses_nl_nn', losses)
    plot_heatmap(U_norms, num_neurons, num_layers, 
                 rf'$L^2$ norm of error with N={N} and $\eta$ = {lr}', 
                 'Number of hidden neurons', 'Number of hidden layers', 
                 'beam_heatmap_num_neurons_layers')
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')

def run2():
    N = 30
    shape = [4*N, N, N]
    lrs = [0.01, 0.05, 0.1, 0.5]
    num_layers = [2, 3, 4, 5]
    num_neurons = 40
    num_expreriments = 10
    num_epochs = 300
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print(f'Experiment nr. {i}')
        U_norms_i, losses_i= train_and_evaluate(Ns=N, lrs=lrs, 
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs, 
                                                 shape=shape)
        print('U_norms_i')
        print(U_norms_i)
        U_norms += U_norms_i
        losses += losses_i
        
    print('U_norms')
    print(U_norms)
    print('U_norms')
    print(U_norms)
    U_norms /= num_expreriments
    losses /= num_expreriments
    losses = losses.detach().numpy()
    np.save(arrays_path / 'losses_lrs_nl', losses)
    plot_heatmap(U_norms, num_layers, lrs, 
                 rf'$L^2$ norm of error with N={N} and {num_neurons} hidden neurons', 
                 'Number of layers', r'$\eta$', f'beam_heatmap_lrs_num_layers2')
    # print(U_norms)
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')

def run3():
    N = 30
    shape = [4*N, N, N]
    lrs = [0.01, 0.05, 0.1, 0.5]
    num_neurons = [30, 40, 50, 60]
    num_layers = 3
    num_expreriments = 10
    num_epochs = 300
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print('Experiment nr. ', i)
        U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lrs, 
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs, 
                                                 shape=shape)
        U_norms += U_norms_i
        print('U_norms_i')
        print(U_norms_i)
        losses += losses_i
    print('U_norms')
    print(U_norms)
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')
    U_norms /= num_expreriments
    losses /= num_expreriments
    losses = losses.detach().numpy()
    plot_heatmap(U_norms, num_neurons, lrs, 
                 rf'$L^2$ norm of error with N={N} and {num_layers} hidden layers', 
                 'Number of neurons in hidden layers', r'$\eta$', 
                 'beam_heatmap_lrs_num_neurons')
    np.save(arrays_path / 'losses_lrs_nn', losses)

def run4():
    Ns = [30, 40, 50, 60]
    lrs = [0.01, 0.05, 0.1, 0.5]
    num_layers = 3
    num_neurons = 50
    num_expreriments = 2
    num_epochs = 300
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print('Experiment nr. ', i)
        U_norms_i, losses_i = train_and_evaluate(Ns=Ns, lrs=lrs,
                                                  num_neurons=num_neurons, 
                                                  num_layers=num_layers, 
                                                  num_epochs=num_epochs)
        
        print('U_norms_i')
        print(U_norms_i)
        U_norms += U_norms_i
        losses += losses_i
    print('U_norms')
    print(U_norms)
    U_norms /= num_expreriments
    losses /= num_expreriments
    losses = losses.detach().numpy()
    np.save(arrays_path / 'losses_lrs_N', losses)
    plot_heatmap(U_norms, Ns, lrs, 
                 rf'$L^2$ norm of error with {num_neurons}' 
                 + rf'hidden neurons and {num_layers} hidden layers', 
                 'N', r'$\eta$', 'beam_heatmap_lrs_N')
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')

if __name__ == '__main__':
    # u_fem20 = np.load('stored_arrays/u_fem_N20.npy')
    u_fem30 = np.load(arrays_path / 'u_fem_N30.npy')
    # exit(L2norm3D(u_fem30, 4*N_test, N_test, N_test, dx, dy, dz))

    x = np.linspace(0, L, 4*N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]

    x_eval = np.linspace(0, L, 4*N_test + 4)[2:-2]
    y_eval = np.linspace(0, D, N_test + 4)[2:-2]
    z_eval = np.linspace(0, H, N_test + 4)[2:-2]
    # print(x, '\n', y, '\n', z); exit()

    "______________________________________________"

    # run1()
    run2()
    # run3()
    # run4()

    # N = 30
    # shape = [120, 30, 30]
    # LHD = [4, 1, 1]
    # domain, dirichlet, neumann = define_domain(L, H, D, N=N)
    # for lr, nl in zip([0.5, 0.1, 0.05, 0.05], [2, 3, 3, 4]):
    #     model = MultiLayerNet(3, *[40]*nl, 3)
    #     energy = NeoHookeanEnergyModel(lmbda, mu)
    #     DemBeam = DeepEnergyMethodBeam(model, energy)
    #     DemBeam.train_model(domain, dirichlet, neumann, shape, neu_axis=[1, 2], LHD=LHD, epochs=300)
    #     torch.save(DemBeam.model.state_dict(), f'trained_models/model_lr{lr}_nl{nl}')