import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
# from scipy.integrate import simpson
# from scipy.integrate import trapezoid
from simps import simpson

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350
# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# dev = torch.device('cpu')

class MultiLayerNet(nn.Module):
    def __init__(self, *neurons):
        super(MultiLayerNet, self).__init__()
        #### throw error if depth < 3 ####
        self.linears = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))])

    def forward(self, x):
        for layer in self.linears:
            x = torch.tanh(layer(x))
        return x

class DeepEnergyMethod:
    def __init__(self, model, energy):
        self.model = model.to(dev)
        self.energy = energy
        
    def train_model(self, data, dirichlet, neumann, shape, LHD, lr=0.5, max_it=20, epochs=20, fb=np.array([[0, -5, 0]]), eval_data=None):
        # data
        # print(data)
        # N = np.array(LHD) / np.array(dxdydz)
        # N = list(map(int, N))
        dxdydz = np.array(LHD) / (np.array(shape) - 1)
        # print(shape)
        # print(LHD)
        # print(dxdydz); exit()
        # N = list(map(int, dxdydz))

        x = torch.from_numpy(data).float().to(dev)
        fb = torch.from_numpy(fb).float().to(dev)
        # fb2 = fb*torch.ones((np.prod(shape), 3)).to(dev)
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=max_it)
        self.x = x

        # boundary
        dirBC_coords = torch.from_numpy(dirichlet['coords']).float().to(dev)
        dirBC_coords.requires_grad_(True)
        dirBC_values = torch.from_numpy(dirichlet['values']).float().to(dev)

        neuBC_coords = torch.from_numpy(neumann['coords']).float().to(dev)
        neuBC_coords.requires_grad_(True)
        neuBC_values = torch.from_numpy(neumann['values']).float().to(dev)

        self.losses = []
        self.eval_losses = []
        start_time = time.time()
        for i in range(epochs+1):
            def closure():
                # internal loss
                u_pred = self.getU(self.model, x)
                u_pred.double()

                IntEnergy, J = self.energy(u_pred, x, J=True)

                # print('internal')
                internal_loss = simps3D(IntEnergy, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)

                # boundary loss
                dir_pred = self.getU(self.model, dirBC_coords)
                bc_dir = LHD[1]*LHD[2]*loss_squared_sum(dir_pred, dirBC_values)
                boundary_loss = torch.sum(bc_dir)

                # external loss
                neu_pred = self.getU(self.model, neuBC_coords)
                bc_neu = torch.bmm((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))

                body_f = torch.matmul(u_pred.unsqueeze(1), fb.unsqueeze(2))

                external_loss = simps3D(body_f, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)

                loss = internal_loss - external_loss + boundary_loss
                optimizer.zero_grad()
                loss.backward()

                self.internal_loss = internal_loss
                self.external_loss = external_loss
                self.energy_loss = loss

                self.current_loss = loss
                return loss
        
            optimizer.step(closure)

            if eval_data:
                eval_shape = [len(eval_data[0]), len(eval_data[1]), len(eval_data[2])]
                _, u_eval, xyz_eval = self.evaluate_model(eval_data[0], eval_data[1], eval_data[2], True)
                eval_internal = self.energy(u_eval, xyz_eval)
                eval_loss1 = simps3D(eval_internal, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)

                eval_BF = torch.matmul(u_eval.unsqueeze(1), fb.unsqueeze(2))
                eval_loss2 = simps3D(eval_BF, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)
                self.eval_loss = eval_loss1 - eval_loss2
            # print(eval_loss1, eval_loss2)

            if i % 5 == 0:
                if eval_data:
                    print(f'Iter: {i:3d}, Energy: {self.energy_loss.item():10.5f}, Int: {self.internal_loss:10.5f}, Ext: {self.external_loss:10.5f}, Eval loss: {self.eval_loss:10.5f}')
                    self.losses.append([self.current_loss.detach().cpu(), self.eval_loss.detach().cpu()])
                else:
                    print(f'Iter: {i:3d}, Energy: {self.energy_loss.item():10.5f}, Int: {self.internal_loss:10.5f}, Ext: {self.external_loss:10.5f}')
                    self.losses.append(self.current_loss.detach().cpu())
        # return self.modeld

    def getU(self, model, x):
        u = model(x).to(dev)
        Ux, Uy, Uz = x[:, 0] * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred

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

        u_pred_torch = self.getU(self.model, xyz_tensor)
        # self.u_pred_torch = u_pred_torch.double()
        # self.u_pred_torch.requires_grad_(True)
        u_pred = u_pred_torch.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
        surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
        surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)

        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))

        if return_pred_tensor:
            return U, u_pred_torch, xyz_tensor
        return U


def loss_squared_sum(input, target):
    return torch.sum((input - target)**2, dim=1) / input.shape[1]*input.data.nelement()

def penalty(input, LHD):
    return LHD[0]*LHD[1]*LHD[2]*torch.sum(input) / input.data.nelement()

def L2norm3D(U, Nx, Ny, Nz, dx, dy, dz):
    ### function from DEM paper ###
    Ux = np.expand_dims(U[0].flatten(), 1)
    Uy = np.expand_dims(U[1].flatten(), 1)
    Uz = np.expand_dims(U[2].flatten(), 1)
    Uxyz = np.concatenate((Ux, Uy, Uz), axis=1)
    n = Ux.shape[0]
    udotu = torch.zeros(n)
    for i in range(n):
        udotu[i] = np.dot(Uxyz[i,:], Uxyz[i,:].T)
    udotu = udotu.reshape(Nx, Ny, Nz)
    L2norm = np.sqrt(simpson(simpson(simpson(udotu, dx=dz), dx=dy), dx=dx))
    return L2norm

def simpsons2D(U, x=None, y=None, dx=None, dy=None, N=25):
    Nx = 4*N; Ny = N; Nz = N
    # U = U.cpu().detach().numpy()
    U = U.flatten().reshape(Ny, Nz)
    if (x and y):
        raise NotImplementedError('Not implemented yet. Please use dx and dy.')
    elif (dx and dy):
        return simpson(simpson(U, dx=dy), dx=dx)
    
# def simpsons3D(U, x=None, y=None, z=None, dx=None, dy=None, dz=None, N=25):
#     Nx = 4*N; Ny = N; Nz = N
#     # U = U.cpu().detach().numpy()
#     # print(U.flatten().shape); exit()
#     U3D = U.flatten().reshape(Nx, Ny, Nz)
#     # breakpoint()
#     if (x and y and z):
#         raise NotImplementedError('Not implemented yet. Please use dx and dy.')
#     elif (dx and dy and dz):
#         return simpson(simpson(simpson(U3D, dx=dz), dx=dy), dx=dx)
    
def simps3D(U, xyz=None, dx=None, dy=None, dz=None, shape=None):
    # print(shape[0]); exit()
    # print('Mansur')
    Nx = shape[0]
    Ny = shape[1]
    Nz = shape[2]
    # print(U.shape)
    U3D = U.flatten().reshape(Nx, Ny, Nz)
    # print(U3D.shape)
    return simpson(simpson(simpson(U3D, dx=dz), dx=dy), dx=dx)