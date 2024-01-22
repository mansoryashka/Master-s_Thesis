import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod

torch.manual_seed(2023)
rng = np.random.default_rng(2023)

N = 20
L = H = D = 1.0
dx = dy = dz = L/N

d_boundary = 0.0
d_cond = [0, 0, 0]

n_boundary = L
n_cond = [-0.5, 0, 0]

def domain(l, h, d, N=25):
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

# class DeepEnergyMethod:
#     def __init__(self, model, energy, dim):
#         self.model = model
#         self.energy = energy
    
#     def train_model(self, data, dirichlet, neumann, LHD, lr=0.3, max_it=20):
#         # data
#         x = torch.from_numpy(data).float()
#         x.requires_grad_(True)

#         # boundary
#         dirBC_coords = torch.from_numpy(dirichlet['coords']).float()
#         dirBC_coords.requires_grad_(True)
#         dirBC_values = torch.from_numpy(dirichlet['values']).float()

#         neuBC_coords = torch.from_numpy(neumann['coords']).float()
#         neuBC_coords.requires_grad_(True)
#         neuBC_values = torch.from_numpy(neumann['values']).float()

#         optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
#         # loss = []
#         start_time = time.time()
#         for i in range(max_it):
#             def closure():
#                 # internal loss
#                 u_pred = self.getU(self.model, x)
#                 u_pred.double()

#                 IntEnergy = self.energy(u_pred, x)
#                 internal_loss = LHD[0]*LHD[1]*LHD[2]*penalty(IntEnergy)

#                 # boundary loss
#                 dir_pred = self.getU(self.model, dirBC_coords)
#                 # print(torch.norm(dir_pred))
#                 bc_dir = LHD[1]*LHD[2]*loss_squared_sum(dir_pred, dirBC_values)
#                 boundary_loss = torch.sum(bc_dir)

#                 # external loss
#                 neu_pred = self.getU(self.model, neuBC_coords)
#                 bc_neu = torch.matmul((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))
#                 external_loss = LHD[1]*LHD[2]*penalty(bc_neu)
#                 # print(neu_pred.shape, neuBC_coords.shape, neuBC_values.shape)
#                 # print(bc_neu.shape); exit()
#                 energy_loss = internal_loss - torch.sum(external_loss)
#                 loss = internal_loss - torch.sum(external_loss) + boundary_loss

#                 optimizer.zero_grad()
#                 loss.backward()

#                 print(f'Iter: {i+1:d}, Energy: {energy_loss.item():10.5f}')
#                 #       + f'loss: {loss.item():10.5f}')
#                 return loss

#             optimizer.step(closure)
#         # return self.model

#     def getU(self, model, x):
#         u = model(x)
#         Ux, Uy, Uz = x[:, 0] * u.T.unsqueeze(1)
#         u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
#         return u_pred

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

E = 1000
nu = 0.3
mu = E / (2*(1 + nu))

# mu = 15
def Psi(u, x):
    # f0 = torch.from_numpy(np.array([1, 0, 0]))
    kappa = 1e3
    Ta = 1.0

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

    J = detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
    trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

    # compressibility = kappa * (J * torch.log(J) - J + 1)
    compressibility = kappa * (J - 1)**2
    neo_hookean = 0.5 * mu * (trC - 3)
    active_stress_energy = 0.5 * Ta / J * (Fxx*Fxx + Fyx*Fyx + Fzx*Fzx - 1)

    # strainEnergy = 0.5 * lmbd * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
    return compressibility + neo_hookean + active_stress_energy


def loss_squared_sum(input, target):
    return torch.sum((input - target)**2, dim=1) / input.shape[1]*input.data.nelement()

def penalty(input):
    return torch.sum(input) / input.data.nelement()

from pyevtk.hl import gridToVTK

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

if __name__ == '__main__':
    domain, dirichlet, neumann = domain(L, H, D)


    model = MultiLayerNet(3, 30, 3)
    DemBeam = DeepEnergyMethodCube(model, Psi, 3)

    DemBeam.train_model(domain, dirichlet, neumann, [L, H, D], epochs=40)

    

    x = rng.random(size=N)
    y = rng.random(size=N)
    z = rng.random(size=N)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    x = L*np.sort(x); y = H*np.sort(y); z = D*np.sort(z)


    U = DemBeam.evaluate_model(x, y, z)
    Udem = np.array(U).copy()
    np.save('u_dem', Udem)
    write_vtk_v2('output/DemCube', x, y, z, U)

    