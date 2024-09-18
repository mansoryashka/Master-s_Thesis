import time
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from EnergyModels import NeoHookeanActiveEnergyModel
from DEM import DeepEnergyMethod, dev, MultiLayerNet, L2norm3D, write_vtk_v2, dev

torch.manual_seed(2023)
rng = np.random.default_rng(2023)

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'

E = 1000
nu = 0.9
mu = E / (2*(1 + nu))

N_test = 20
L = H = D = 1.0
LHD = [L, H, D]
dx = dy = dz = L / (N_test - 1)

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = L
n_cond = [-0.5, 0, 0]

def define_domain(L, H, D, N=20):
    x = np.linspace(0, L, N)
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
        u_pred_torch = self(self.model, xyz_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
        surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
        surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))

        if return_pred_tensor:
            return U, u_pred_torch, xyz_tensor
        return U

### Skrive blokkene til en egen funksjon? Kalles på helt likt inne i loopene ###
def train_and_evaluate(Ns=20, lrs=0.1, num_neurons=20, num_layers=2, num_epochs=40, shape=[20, 20, 20], eval_data=None):
    num_losses = int(num_epochs)
    if eval_data:
        nr_losses = 2
    else:
        nr_losses = 1
    energy = NeoHookeanActiveEnergyModel(mu)
    # train on many N values
    if isinstance((Ns and not lrs), (list, tuple)):
        # print('Ns')
        u_norms = np.zeros(len(Ns))
        losses = torch.zeros((nr_losses, num_losses, len(Ns)))
        for i, N in enumerate(Ns):
            # define model, DEM and domain
            model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
            DemCube = DeepEnergyMethodCube(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemCube.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lrs, epochs=num_epochs, fb=np.asarray([[0, 0, 0]]))
            # evaluate model
            U_pred, u_pred_torch, xyz_tensor = DemCube.evaluate_model(x, y, z, return_pred_tensor=True)
            # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
            # store solution
            # write_vtk_v2(f'output/DemCube_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
            # write_vtk_v2(f'output/DemCube_N{N}', x, y, z, U_pred)
            # calculate L2norm
            u_norms[i] = (L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                            / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz))
            losses[:, :, i] = DemCube.losses
            del model
    # train on many learning rates and number of neurons in hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_neurons, (list, tuple)):
        print('lrs, num_n')
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        losses = torch.zeros((nr_losses, num_losses, len(lrs), len(num_neurons)))
        for i, lr in enumerate(lrs):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*num_layers), 3)
                DemCube = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)

                DemCube.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs, fb=np.asarray([[0, 0, 0]]))
                U_pred, u_pred_torch, xyz_tensor = DemCube.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)

                # store solution
                # write_vtk_v2(f'output/DemCube_lr{lr}_nn{n}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # write_vtk_v2(f'output/DemCube_lr{lr}_nn{n}', x, y, z, U_pred)
                u_norms[i, j] = (L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz))
                losses[:, :, i, j] = DemCube.losses
                del model
    # train on many learning rates and number of hidden layers
    elif isinstance(lrs, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('lrs, num_l')
        u_norms = np.zeros((len(lrs), len(num_layers)))
        losses = torch.zeros((nr_losses, num_losses, len(lrs), len(num_layers)))
        for i, lr in enumerate(lrs):
            for j, l in enumerate(num_layers):
                model = MultiLayerNet(3, *([num_neurons]*l), 3)
                DemCube = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemCube.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemCube.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemCube_lr{lr}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # write_vtk_v2(f'output/DemCube_lr{lr}_nl{l}', x, y, z, U_pred)

                u_norms[i, j] = (L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz))
                losses[:, :, i, j] = DemCube.losses
                del model
    # train on number of neurons in hidden layers and number of hidden layers
    elif isinstance(num_neurons, (list, tuple)) and isinstance(num_layers, (list, tuple)):
        print('num_n, num_l')
        u_norms = np.zeros((len(num_layers), len(num_neurons)))
        losses = torch.zeros((nr_losses, num_losses, len(num_layers), len(num_neurons)))
        for i, l in enumerate(num_layers):
            for j, n in enumerate(num_neurons):
                model = MultiLayerNet(3, *([n]*l), 3)
                DemCube = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemCube.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lrs, epochs=num_epochs, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemCube.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemCube_nn{n}_nl{l}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # write_vtk_v2(f'output/DemCube_nn{n}_nl{l}', x, y, z, U_pred)
                u_norms[i, j] = (L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz))
                
                losses[:, :, i, j] = DemCube.losses
                del model
    # train on N values and learning rates
    elif isinstance(Ns, (list, tuple)) and isinstance(lrs, (list, tuple)):
        # print(type(Ns), type(lrs), isinstance((Ns and lrs), list), Ns); exit()
        print('Ns and lrs')
        u_norms = np.zeros((len(lrs), len(Ns)))
        losses = torch.zeros((nr_losses, num_losses, len(lrs), len(Ns)))
        for j, N in enumerate(Ns):
            shape = [N, N, N]
            for i, lr in enumerate(lrs):
                # define model, DEM and domain
                model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
                DemCube = DeepEnergyMethodCube(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=N)
                # train model
                DemCube.train_model(domain, dirichlet, neumann, shape, LHD, neu_axis=[1, 2], lr=lr, epochs=num_epochs, fb=np.asarray([[0, 0, 0]]))
                # evaluate model
                U_pred, u_pred_torch, xyz_tensor = DemCube.evaluate_model(x, y, z, return_pred_tensor=True)
                # VonMises_pred = VonMises_stress(u_pred_torch, xyz_tensor)
                # store solution
                # write_vtk_v2(f'output/DemCube_lr{lr}_N{N}', x, y, z, {'Displacement': U_pred, 'vonMises stress': VonMises_pred})
                # write_vtk_v2(f'output/DemCube_lr{lr}_N{N}', x, y, z, U_pred)
                # calculate L2norm
                u_norms[i, j] = (L2norm3D(U_pred - u_fem20, N_test, N_test, N_test, dx, dy, dz)
                                / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz))
                losses[:, :, i, j] = DemCube.losses
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

def run1():
    N = 20
    lr = 0.1
    shape = [N, N, N]
    num_layers = [2, 3, 4, 5]
    num_neurons = [20, 30, 40, 50]
    num_expreriments = 10
    num_epochs = 100
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print('Experiment: ', i)
        U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lr, 
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs, shape=shape) 
        U_norms += U_norms_i
        losses += losses_i
    losses = losses.detach().numpy()
    U_norms /= num_expreriments
    losses /= num_expreriments
    print(U_norms)
    np.save(arrays_path / 'losses_nl_nn', losses)
    plot_heatmap(U_norms, num_neurons, num_layers, 
                 rf'$L^2$ norm of error with N={N} and $\eta$ = {lr}', 
                 'Number of hidden neurons', 'Number of hidden layers', 
                 'cube_heatmap_num_neurons_layers120')
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')

def run2():
    N = 40
    shape = [N, N, N]
    lrs = [.01, .05, .1, .5]
    num_neurons = [20, 30, 40, 50]
    num_layers = 5
    num_expreriments = 10
    num_epochs = 100
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print('Experiment: ', i)
        U_norms_i, losses_i = train_and_evaluate(Ns=N, lrs=lrs, 
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs, shape=shape)
        U_norms += U_norms_i
        losses += losses_i
        print(i, U_norms_i)
    losses = losses.detach().numpy()
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')
    U_norms /= num_expreriments
    losses /= num_expreriments
    print(U_norms)
    plot_heatmap(U_norms, num_neurons, lrs, 
                 rf'$L^2$ norm of error with N={N} and {num_layers} hidden layers', 
                 'Number of neurons in hidden layers', r'$\eta$', 
                 'cube_heatmap_lrs_num_neurons1')
    np.save(arrays_path / 'losses_lrs_nn', losses)

def run3():
    Ns = [20, 30, 40, 50]
    lrs = [0.01, 0.05, 0.1, 0.5]
    num_neurons = 20
    num_layers = 3
    num_expreriments = 10
    num_epochs = 100
    U_norms = 0
    losses = 0
    start = time.time()
    for i in range(num_expreriments):
        print('Experiment: ', i)
        U_norms_i, losses_i = train_and_evaluate(Ns=Ns, lrs=lrs, 
                                                 num_neurons=num_neurons, 
                                                 num_layers=num_layers, 
                                                 num_epochs=num_epochs)
        U_norms += U_norms_i
        losses += losses_i
    losses = losses.detach().numpy()
    U_norms /= num_expreriments
    losses /= num_expreriments
    np.save(arrays_path / 'losses_lrs_N', losses)
    plot_heatmap(U_norms, Ns, lrs, 
                 rf'$L^2$ norm of error with {num_neurons} hidden neurons and {num_layers} hidden layers', 
                 'N', r'$\eta$', 'cube_heatmap_lrs_N')
    tid = time.time() - start
    print(f'tid: {tid:.2f}s')
    print(f'tid: {tid/60:.2f}m')
    print(f'tid: {tid/3600:.2f}t')

if __name__ == '__main__':
    u_fem20 = np.load(arrays_path / 'u_fem20.npy')
    # exit(f'FEM: {L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)}')


    x = np.linspace(0, L, N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]

    x_eval = np.linspace(0, L, N_test + 4)[1:-1]
    y_eval = np.linspace(0, D, N_test + 4)[1:-1]
    z_eval = np.linspace(0, H, N_test + 4)[1:-1]

    # run1()
    # run2()
    # run3()

    N = 20
    shape = [N, N, N]
    LHD = [1, 1, 1]
    domain, dirichlet, neumann = define_domain(L, H, D, N=N)
    for nn, nl in zip([20, 30, 40], [5, 3, 4]):
        model = MultiLayerNet(3, *[nn]*nl, 3)
        energy = NeoHookeanActiveEnergyModel(mu)
        DemBeam = DeepEnergyMethodCube(model, energy)
        DemBeam.train_model(domain, dirichlet, neumann, shape, neu_axis=[1, 2], LHD=LHD, epochs=100)
        torch.save(DemBeam.model.state_dict(), f'trained_models/model_nn{nn}_nl{nl}')