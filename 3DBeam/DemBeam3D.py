import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
import scipy.integrate as sp
from pyevtk.hl import gridToVTK
from pathlib import Path

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import seaborn as sns
sns.set()

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, MultiLayerNet, dev, L2norm3D

# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'
# code to run on ML node with no hangup :
# CUDA_VISIBLE_DEVICES=x TMP=./tmp nohup python yourscript.py > out1.log 2> error1.log &

N_test = 30
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

dir_bcs = {d_boundary: d_cond}
neu_bcs = {n_boundary: n_cond}

def define_domain(L, H, D, N=25, dir_bcs=None, neu_bcs=None):
    x = np.linspace(0, L, 4*N)
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
    fig.savefig('fig.png')
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
    ### function from DEM paper ###
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

### Skrive blokkene til en egen funksjon? Kalles på helt likt inne i loopene ###
def train_and_evaluate(Ns=20, lrs=0.1, num_neurons=20, num_layers=2, num_epochs=40, max_it=20):
    # train on many N values
    if isinstance((Ns and not lrs), (list, tuple)):
        print('Ns')
        u_norms = np.zeros(len(Ns))
        for i, N in enumerate(Ns):
            # define model, DEM and domain
            model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
            DemBeam = DeepEnergyMethod(model, energy)
            domain, dirichlet, neumann = define_domain(L, H, D, N=N)
            # train model
            DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lrs, max_it=max_it, epochs=num_epochs)
            # evaluate model
            U_pred = DemBeam.evaluate_model(x, y, z)
            # calculate L2norm
            u_norms[i] = L2norm3D(U_pred, 4*N_test, N_test, N_test, dx, dy, dz)
    # train on many learning rates and number of neurons in hidden layers
    elif isinstance((lrs and num_neurons), (list, tuple)):
        print('lrs, num_n')
        u_norms = np.zeros((len(lrs), len(num_neurons)))
        for j, n in enumerate(num_neurons):
            for i, lr in enumerate(lrs):
                model = MultiLayerNet(3, *([n]*num_layers), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)

                DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lr, max_it=max_it, epochs=num_epochs)
                U_pred = DemBeam.evaluate_model(x, y, z)

                u_norms[i, j] = L2norm3D(U_pred, 4*N_test, N_test, N_test, dx, dy, dz)
    # train on many learning rates and number of hidden layers
    elif isinstance((lrs and num_layers), (list, tuple)):
        print('lrs, num_l')
        u_norms = np.zeros((len(lrs), len(num_layers)))
        for j, l in enumerate(num_layers):
            for i, lr in enumerate(lrs):
                model = MultiLayerNet(3, *([num_neurons]*l), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lr, max_it=max_it, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)

                u_norms[i, j] = L2norm3D(U_pred, 4*N_test, N_test, N_test, dx, dy, dz)
    # train on number of neurons in hidden layers and number of hidden layers
    elif isinstance((num_neurons and num_layers), (list, tuple)):
        print('num_n, num_l')
        u_norms = np.zeros((len(num_neurons), len(num_layers)))
        for j, n in enumerate(num_neurons):
            for i, l in enumerate(num_layers):
                model = MultiLayerNet(3, *([n]*l), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=Ns)
                DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lrs, max_it=max_it, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)

                u_norms[i, j] = L2norm3D(U_pred, 4*N_test, N_test, N_test, dx, dy, dz)
    # train on many N values and learning rates
    elif isinstance((Ns and lrs), (list, tuple)):
        # print(type(Ns), type(lrs), isinstance((Ns and lrs), list), Ns); exit()
        print('Ns and lrs')
        u_norms = np.zeros((len(lrs), len(Ns)))
        for j, N in enumerate(Ns):
            for i, lr in enumerate(lrs):
                # define model, DEM and domain
                model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
                DemBeam = DeepEnergyMethod(model, energy)
                domain, dirichlet, neumann = define_domain(L, H, D, N=N)
                # train model
                DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lr, max_it=max_it, epochs=num_epochs)
                # evaluate model
                U_pred = DemBeam.evaluate_model(x, y, z)
                # calculate L2norm
                u_norms[i, j] = L2norm3D(U_pred, 4*N_test, N_test, N_test, dx, dy, dz)
    else:
        raise Exception('You have to provide a list of N values or one of the following:\n' + 
                        '\t- lrs AND num_neurons\n\t- lrs AND num_layers\n\t- num_neurons AND num_layers')
    return u_norms

### implementere med define_domain som input? 
### kan bruke samme funksjon i alle 3D filene
def function_to_be_called_inside_train_and_eval(define_domain, N, lr, num_neurons, num_layers, num_epochs, max_it):
    # define model, DEM and domain
    model = MultiLayerNet(3, *([num_neurons]*num_layers), 3)
    DemBeam = DeepEnergyMethod(model, energy)
    domain, dirichlet, neumann = define_domain(L, H, D, N=N)
    # train model
    DemBeam.train_model(domain, dirichlet, neumann, LHD, lr=lr, max_it=max_it, epochs=num_epochs)
    # evaluate model
    return DemBeam.evaluate_model(x, y, z)

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

if __name__ == '__main__':
    u_fem30 = np.load('stored_arrays/u_fem_N=30.npy')
    print(f'FEM: {L2norm(u_fem30)} \n')


    x = np.linspace(0, L, 4*N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]
    # Ns = [20, 30, 40, 50]

    # U_norms = train_and_evaluate(Ns, num_epochs=20, num_layers=3)
    # print(U_norms)
    # print((U_norms - L2norm(u_fem30))/L2norm(u_fem30))

    # N = 30
    # lrs = [.05, .1, .5, .9]
    # num_layers = [2, 3, 4, 5]
    # num_neurons = 30
    # num_expreriments = 30
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=80)
    # U_norms /= num_expreriments
    # e_norms = (U_norms - L2norm(u_fem30)) / L2norm(u_fem30)
    # plot_heatmap(e_norms, num_layers, lrs, rf'$L^2$ error norm with N={N} and {num_neurons} hidden neurons', 'Number of layers', r'$\eta$', 'heatmap_lrs_num_layers80')


    N = 30
    lrs = [.05, .1, .5, 1]
    num_layers = 3
    num_neurons = [10, 20, 30, 40, 50]
    num_expreriments = 30
    U_norms = 0
    for i in range(num_expreriments):
        U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=80)
    U_norms /= num_expreriments
    e_norms = (U_norms - L2norm(u_fem30)) / L2norm(u_fem30)
    plot_heatmap(e_norms, num_neurons, lrs, rf'$L^2$ error norm with N={N} and {num_layers} hidden layers', 'Number of neurons in hidden layers', r'$\eta$', 'heatmap_lrs_num_neurons80')
    print(U_norms)
    print((U_norms - L2norm(u_fem30))/L2norm(u_fem30))


    # Ns = [20, 30, 40, 50, 60]
    # lrs = [.05, .1, .5, .9]
    # num_layers = 3
    # num_neurons = 30
    # num_expreriments = 30
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=Ns, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=40)
    # U_norms /= num_expreriments
    # print(U_norms)
    # e_norms = (U_norms - L2norm(u_fem30)) / L2norm(u_fem30)
    # plot_heatmap(e_norms, Ns, lrs, rf'$L^2$ error norm with {num_neurons} hidden neurons and {num_layers} hidden layers', 'N', r'$\eta$', 'heatmap_lrs_N')