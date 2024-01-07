import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
import matplotlib
matplotlib.rcParams['figure.dpi'] = 350
# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)

N = 25
x0 = 0
E = 1000
nu = 0.3

l = 4
h = 1
d = 1

dx = l/(4*N)
dy = h/N
dz = d/N

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = l
n_cond = [0, -5, 0]

def domain(l, h, d, N=25):
    x = np.linspace(0, l, int(4*N))
    y = np.linspace(0, h, N)
    z = np.linspace(0, d, N)

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

# domain(l, h, d)

class MultiLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerNet, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = self.l4(x)
        return x

class DeepEnergyMethod:
    def __init__(self, model, energy, dim):
        self.model = model
        self.energy = energy
    
    def train_model(self, data, dirichlet, neumann, LHD, lr=0.5, max_it=20, epochs=20):
        # data
        x = torch.from_numpy(data).float()
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=max_it)

        # boundary
        dirBC_coords = torch.from_numpy(dirichlet['coords']).float()
        dirBC_coords.requires_grad_(True)
        dirBC_values = torch.from_numpy(dirichlet['values']).float()

        neuBC_coords = torch.from_numpy(neumann['coords']).float()
        neuBC_coords.requires_grad_(True)
        neuBC_values = torch.from_numpy(neumann['values']).float()

        self.losses = {}
        start_time = time.time()
        for i in range(epochs):
            def closure():
                # internal loss
                u_pred = self.getU(self.model, x)
                u_pred.double()

                IntEnergy = self.energy(u_pred, x)
                internal_loss = LHD[0]*LHD[1]*LHD[2]*penalty(IntEnergy)

                # boundary loss
                dir_pred = self.getU(self.model, dirBC_coords)
                # print(torch.norm(dir_pred))
                bc_dir = LHD[1]*LHD[2]*loss_squared_sum(dir_pred, dirBC_values)
                boundary_loss = torch.sum(bc_dir)

                # external loss
                neu_pred = self.getU(self.model, neuBC_coords)
                bc_neu = torch.matmul((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))
                external_loss = LHD[1]*LHD[2]*penalty(bc_neu)
                # print(neu_pred.shape, neuBC_coords.shape, neuBC_values.shape)
                # print(bc_neu.shape); exit()
                energy_loss = internal_loss - torch.sum(external_loss)
                loss = internal_loss - torch.sum(external_loss) + boundary_loss

                optimizer.zero_grad()
                loss.backward()
                
                if self.losses.get(i):
                    self.losses[i] += loss.item() / max_it
                else:
                    self.losses[i] = loss.item() / max_it

                #       + f'loss: {loss.item():10.5f}')
                print(f'Iter: {i+1:d}, Energy: {energy_loss.item():10.5f}')
                return loss

            optimizer.step(closure)
        # return self.model

    def getU(self, model, x):
        u = model(x)
        Ux, Uy, Uz = x[:, 0] * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred

    def evaluate_model(self, x, y, z):
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        z1D = zGrid.flatten()
        xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
        xyz_tensor = torch.from_numpy(xyz).float()
        xyz_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xyz_tensor)
        u_pred_torch = self.getU(self.model, xyz_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
        surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
        surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        return U

lmbd =  E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))

def Psi(u, x):
    duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.shape[0], 1), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.shape[0], 1), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.shape[0], 1), create_graph=True, retain_graph=True)[0]

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

def loss_squared_sum(input, target):
    return torch.sum((input - target)**2, dim=1) / input.shape[1]*input.data.nelement()

def penalty(input):
    return torch.sum(input) / input.data.nelement()

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

from pyevtk.hl import gridToVTK

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

import scipy.integrate as sp

def L2norm(U):
    Ux = np.expand_dims(U[0].flatten(), 1)
    Uy = np.expand_dims(U[1].flatten(), 1)
    Uz = np.expand_dims(U[2].flatten(), 1)
    Uxyz = np.concatenate((Ux, Uy, Uz), axis=1)
    n = Ux.shape[0]
    udotu = np.zeros(n)
    for i in range(n):
        udotu[i] = np.dot(Uxyz[i,:], Uxyz[i,:].T)
    udotu = udotu.reshape(4*N, N, N)
    L2norm = np.sqrt(sp.simps(sp.simps(sp.simps(udotu, dx=dz), dx=dy), dx=dx))
    return L2norm

if __name__ == '__main__':

    # x = np.linspace(0, l, int(4*N))
    # y = np.linspace(0, h, N)
    # z = np.linspace(0, d, N)
    
    u_fem = np.load('u_fem.npy')
    # print(u_fem.shape)
    # print(f'FEM: {L2norm(u_fem):8.5f} \n')
    # exit()
    x = rng.random(size=4*N)
    y = rng.random(size=N)
    z = rng.random(size=N)
    x = l*np.sort(x); y = h*np.sort(y); z = d*np.sort(z)
    

    # for N in range(10, 35, 5):
    domain, dirichlet, neumann = domain(l, h ,d)

    model = MultiLayerNet(3, 30, 3)
    DemBeam = DeepEnergyMethod(model, Psi, 3)

    DemBeam.train_model(domain, dirichlet, neumann, [l, h, d], epochs=4)
    print(DemBeam.losses)
    U = DemBeam.evaluate_model(x, y, z)
    
    Udem = np.array(U).copy()

    # np.save('u_dem', Udem)
    # write_vtk_v2('output/NeoHook3D', x, y, z, U)

    # print(f'\nDEM: {L2norm(Udem):8.5f}   FEM: {L2norm(u_fem):8.5f} \n')
    # print((L2norm(Udem) - L2norm(u_fem))/(L2norm(u_fem)))


    
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(u_fem[0], u_fem[1], u_fem[2], s=0.002)
    # ax.scatter(U[0], U[1], U[2], s=0.002, c='tab:red')
    # plt.show()