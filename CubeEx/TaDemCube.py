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
from DEM import DeepEnergyMethod, dev, MultiLayerNet, L2norm3D, penalty, loss_squared_sum

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

def define_domain(l, h, d, N=25):
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

class DeepEnergyMethodCubeTa(DeepEnergyMethod):
    def train_model(self, data, Ta, dirichlet, neumann, LHD, lr=0.5, max_it=20, epochs=20):
        # data
        # print(data)
        x = torch.from_numpy(data).float().to(dev)
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=max_it)

        # boundary
        dirBC_coords = torch.from_numpy(dirichlet['coords']).float().to(dev)
        dirBC_coords.requires_grad_(True)
        dirBC_values = torch.from_numpy(dirichlet['values']).float().to(dev)

        neuBC_coords = torch.from_numpy(neumann['coords']).float().to(dev)
        neuBC_coords.requires_grad_(True)
        neuBC_values = torch.from_numpy(neumann['values']).float().to(dev)

        self.losses = {}
        start_time = time.time()
        for i in range(epochs):
            def closure():
                # internal loss
                u_pred = self.getU(self.model, x)
                u_pred.double()

                IntEnergy = self.energy(u_pred, x, Ta)
                internal_loss = LHD[0]*LHD[1]*LHD[2]*penalty(IntEnergy)

                # boundary loss
                dir_pred = self.getU(self.model, dirBC_coords)
                # print(torch.norm(dir_pred))
                bc_dir = LHD[1]*LHD[2]*loss_squared_sum(dir_pred, dirBC_values)
                boundary_loss = torch.sum(bc_dir)

                # external loss
                neu_pred = self.getU(self.model, neuBC_coords)
                bc_neu = torch.matmul((neu_pred + neuBC_coords[:,:-1]).unsqueeze(1), neuBC_values.unsqueeze(2))
                external_loss = LHD[1]*LHD[2]*penalty(bc_neu)
                # print(neu_pred.shape, neuBC_coords.shape, neuBC_values.shape)
                # print(bc_neu.shape); exit()
                energy_loss = internal_loss - torch.sum(external_loss)
                loss = internal_loss - torch.sum(external_loss) + boundary_loss

                optimizer.zero_grad()
                loss.backward()
                
                if self.losses.get(i+1):
                    self.losses[i+1] += loss.item() / max_it
                else:
                    self.losses[i+1] = loss.item() / max_it

                #       + f'loss: {loss.item():10.5f}')
                # print(f'Iter: {i+1:2d}, Energy: {energy_loss.item():10.5f}')
                # print(f'Iter: {i+1:2d}, Energy: {loss}')
                return loss

            optimizer.step(closure)

    def evaluate_model(self, x, y, z, Ta):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        z1D = zGrid.flatten()
        Ta_arr = np.full((x1D.shape[0], 1), Ta)
        xyzTa = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T, Ta_arr), axis=-1)
        xyz_tensor = torch.from_numpy(xyzTa).float().to(dev)
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
def energy(u, x, Ta=1):
    # f0 = torch.from_numpy(np.array([1, 0, 0]))
    kappa = 1e3

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

    # compressibility = kappa * (J * torch.log(J) - J + 1)
    compressibility = kappa * (J - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / J * (Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)

    # strainEnergy = 0.5 * lmbd * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
    return compressibility + neo_hookean + active_stress_energy

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
            u_norms[i] = L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz)
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

                u_norms[i, j] = L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz)
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

                u_norms[i, j] = L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz)
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

                u_norms[i, j] = L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz)
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
                u_norms[i, j] = L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz)
    else:
        raise Exception('You have to provide a list of N values or one of the following:\n' + 
                        '\t- lrs AND num_neurons\n\t- lrs AND num_layers\n\t- num_neurons AND num_layers')
    return u_norms

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

def ca_transient(t, tstart=0.05):
    tau1 = 0.05
    tau2 = 0.110

    ca_diast = 0.0
    ca_ampl = 1.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (
        -1 / (1 - tau2 / tau1)
    )
    # ca = np.zeros_like(t)

    # ca[t <= tstart] = ca_diast
    if 2*t % 1 <= tstart:
        ca = ca_diast
    else:
        ca = (ca_ampl - ca_diast) / beta * (
            np.exp(-(2*t % 1- tstart) / tau1)
            - np.exp(-(2*t % 1- tstart) / tau2)) + ca_diast

    return ca

if __name__ == '__main__':
    u_fem20 = np.load(arrays_path / 'u_fem20.npy')
    print(f'FEM: {L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)}')

    x = np.linspace(0, L, N_test + 2)[1:-1]
    y = np.linspace(0, D, N_test + 2)[1:-1]
    z = np.linspace(0, H, N_test + 2)[1:-1]

    # N = 20
    # lrs = [.05, .1, .5, .9]
    # num_layers = [2, 3, 4, 5]
    # num_neurons = 30
    # num_expreriments = 1
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=40)
    # U_norms /= num_expreriments
    # e_norms = (U_norms - L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)) / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)
    # plot_heatmap(e_norms, num_layers, lrs, rf'$L^2$ error norm with N={N} and {num_neurons} hidden neurons', 'Number of layers', r'$\eta$', 'cube_heatmap_lrs_num_layers')
    # print(U_norms)
    # print(e_norms)

    # N = 20
    # lrs = [.05, .1, .5, 1]
    # num_layers = 3
    # num_neurons = [10, 20, 30, 40, 50]
    # num_expreriments = 30
    # U_norms = 0
    # for i in range(num_expreriments):
    #     U_norms += train_and_evaluate(Ns=N, lrs=lrs, num_neurons=num_neurons, num_layers=num_layers, num_epochs=40)
    # U_norms /= num_expreriments
    # e_norms = (U_norms - L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)) / L2norm3D(u_fem20, N_test, N_test, N_test, dx, dy, dz)
    # plot_heatmap(e_norms, num_neurons, lrs, rf'$L^2$ error norm with N={N} and {num_layers} hidden layers', 'Number of neurons in hidden layers', r'$\eta$', 'cube_heatmap_lrs_num_neurons')
    # print(U_norms)
    # print(e_norms)

    T = 1.2
    dt = 0.05
    t = 0
    num_steps = int(T/dt + 1)
    Ta = ca_transient(t)

    N=20; lr=.5; num_neurons=30; num_layers=3
    train_domain, dirichlet, neumann = define_domain(L, H, D, N)
    print(dirichlet['coords'].shape)
    model = MultiLayerNet(4, *([num_neurons]*num_layers), 3)
    DemCubeTa = DeepEnergyMethodCubeTa(model, energy)
    Ta_arr = np.zeros((train_domain.shape[0], 1))

    Ta_arr[:] = Ta
    Ta_arr_for_bc = Ta_arr[:dirichlet['coords'].shape[0]]
    train_domain_wTa = np.concatenate((train_domain, Ta_arr), axis=1)
    dirichlet['coords'] = np.concatenate((dirichlet['coords'], Ta_arr_for_bc), axis=1)
    neumann['coords'] = np.concatenate((neumann['coords'], Ta_arr_for_bc), axis=1)

    import time
    for i in range(num_steps):
        start = time.perf_counter()

        Ta_arr[:] = Ta
        Ta_arr_for_bc = Ta_arr[:dirichlet['coords'].shape[0]]
        train_domain_wTa[:,-1] = Ta
        dirichlet['coords'][:, -1] = Ta
        neumann['coords'][:, -1] = Ta

        DemCubeTa.train_model(train_domain_wTa, Ta, dirichlet, neumann, LHD, lr, epochs=10)
        U_pred = DemCubeTa.evaluate_model(x, y, z, Ta)
        print(time.perf_counter() - start)

        t += dt
        Ta = ca_transient(t)
        print(L2norm3D(U_pred, N_test, N_test, N_test, dx, dy, dz))
        write_vtk_v2(f'output/CubeTa{i:02d}', x, y, z, U_pred)
