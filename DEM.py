import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from pyevtk.hl import gridToVTK
from pathlib import Path
from simps import simpson
torch.autograd.set_detect_anomaly(True)

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350
# np.random.seed(2023)
torch.manual_seed(2023)
rng = np.random.default_rng(2023)
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# dev = torch.device('cpu')

current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models' / 'run3'
msg = "You have to run the files from their respective folders!"

assert figures_path.exists(), msg
assert arrays_path.exists(), msg
assert models_path.exists(), msg

class MultiLayerNet(nn.Module):
    def __init__(self, *neurons):
        super(MultiLayerNet, self).__init__()
        #### throw error if depth < 3 ####
        self.linears = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))])
    def forward(self, x):
        for layer in self.linears:
            x = torch.tanh(layer(x))

            # x = torch.tanh(torch.nn.functional.linear(x, layer.weight.clone(), layer.bias))
        return x

class DeepEnergyMethod:
    def __init__(self, model, energy):
        self.model = model.to(dev)
        self.energy = energy
        
    def train_model(self, data, dirichlet, neumann, shape, LHD, xyz=None, lr=0.5, max_it=20, epochs=20, fb=np.array([[0, -5, 0]]), eval_data=None):
        dxdydz = np.array(LHD) / (np.array(shape) - 1)
        # print(dxdydz); exit()

        x = torch.from_numpy(data).float().to(dev)
        fb = torch.from_numpy(fb).float().to(dev)
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
        neuBC_coords_i = neuBC_coords[:]
        neuBC_coords_i.requires_grad_(True)

        self.losses = []
        self.eval_losses = []
        prev_loss = torch.tensor([0.0]).to(dev)
        start_time = time.time()

        nn = self.model.linears[0].out_features
        nl = len(self.model.linears) - 1
        j = 0
        while Path(models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}').exists():
            j += 1
        
        for i in range(epochs+1):
            def closure():
                # internal loss
                u_pred = self.getU(self.model, x)
                u_pred.double()

                IntEnergy, J = self.energy(u_pred, x, J=True)

                # print('internal')
                # internal_loss = simps3D(IntEnergy, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)
                internal_loss = simps3D(IntEnergy, xyz=xyz, shape=shape)

                # boundary loss
                dir_pred = self.getU(self.model, dirBC_coords)

                bc_dir = LHD[1]*LHD[2]*loss_squared_sum(dir_pred, dirBC_values)
                boundary_loss = torch.sum(bc_dir)

                # external loss
                neu_pred = self.getU(self.model, neuBC_coords)
                bc_neu = torch.bmm((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))
                # neu_pred = self.getU(self.model, neuBC_coords_i)
                # bc_neu = torch.bmm((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))
                # self.neu_pred = neu_pred
                body_f = torch.matmul(u_pred.unsqueeze(1), fb.unsqueeze(2))
                # print(body_f.shape)
                # external_loss = simps3D(body_f, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape) + simps2D(bc_neu, dx=dxdydz[1], dy=dxdydz[2], shape=[shape[1], shape[2]])
                external_loss = simps3D(body_f, xyz=xyz, shape=shape) + simps2D(bc_neu, xy=xyz[1:], shape=[shape[0], shape[2]])

                loss = internal_loss - external_loss + boundary_loss
                optimizer.zero_grad()
                loss.backward()
                # loss.backward(retain_graph=True)

                self.internal_loss = internal_loss
                self.external_loss = external_loss
                self.energy_loss = loss

                self.current_loss = loss
                return loss

            optimizer.step(closure)
            # neuBC_coords_i = self.neu_pred + neuBC_coords

            loss_change = torch.abs(self.current_loss - prev_loss)
            prev_loss = self.current_loss

            # if i == 100:
            #     best_change = loss_change
            #     best_epoch = i
            #     torch.save(self.model.state_dict(), 
            #                 models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}')
            # elif i > 100:
            #     if loss_change <= best_change:
            #         best_change = loss_change
            #         best_epoch = i
            #         torch.save(self.model.state_dict(), 
            #                    models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}')
            if eval_data:
                eval_shape = [len(eval_data[0]), len(eval_data[1]), len(eval_data[2])]
                _, u_eval, xyz_eval = self.evaluate_model(eval_data[0], eval_data[1], eval_data[2], True)
                eval_internal = self.energy(u_eval, xyz_eval)
                eval_loss1 = simps3D(eval_internal, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)

                eval_BF = torch.matmul(u_eval.unsqueeze(1), fb.unsqueeze(2))
                eval_loss2 = simps3D(eval_BF, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)
                self.eval_loss = eval_loss1 - eval_loss2

                print(f'Iter: {i:3d}, Energy: {self.energy_loss.item():10.5f}, Int: {self.internal_loss:10.5f}, Ext: {self.external_loss:10.5f}, Eval loss: {self.eval_loss:10.5f}, Loss_change: {loss_change.item():8.5f}')
                self.losses.append([self.current_loss.detach().cpu(), self.eval_loss.detach().cpu()])
            else:
                print(f'Iter: {i:3d}, Energy: {self.energy_loss.item():10.5f}, Int: {self.internal_loss:10.5f}, Ext: {self.external_loss:10.5f}')
                self.losses.append(self.current_loss.detach().cpu())
        # print(best_change, best_epoch)
        # return self.model

    def getU(self, model, x):
        u = model(x).to(dev)
        Ux, Uy, Uz = x[:, 0] * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred

    def evaluate_model(self, x, y, z, return_pred_tensor=False):
        raise NotImplementedError("You need to implement an 'evaluate_model' method in the subclass!")


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

def simps2D(U, xy=None, dx=None, dy=None, shape=None):
    Nx = shape[0]
    Ny = shape[1]
    U = U.flatten().reshape(Nx, Ny)
    if xy:
        return simpson(simpson(U, x=xy[1]), x=xy[0])
    else:
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
    Nx = shape[0]
    Ny = shape[1]
    Nz = shape[2]
    U3D = U.flatten().reshape(Nx, Ny, Nz)
    if xyz is not None:
        print('fikk xyz')
        return simpson(simpson(simpson(U3D, x=xyz[2]), x=xyz[1]), x=xyz[0])
    return simpson(simpson(simpson(U3D, dx=dz), dx=dy), dx=dx)

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    if isinstance(U, dict):
        gridToVTK(filename, xx, yy, zz, pointData=U)
    else:
        gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})