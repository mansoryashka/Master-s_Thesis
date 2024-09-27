import time
import numpy as np
from pathlib import Path
import torch
from torch import nn
from pyevtk.hl import gridToVTK
from simps import simpson
torch.autograd.set_detect_anomaly(True)

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350
torch.manual_seed(2023)
rng = np.random.default_rng(2023)

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(dev)
# dev = torch.device('cpu')
print('Running on GPU') if dev == torch.device('cuda') else print('Running on CPU')


current_path = Path.cwd().resolve()
figures_path = current_path / 'figures'
arrays_path = current_path / 'stored_arrays'
models_path = current_path / 'trained_models'
msg = "You have to run the files from their respective folders!"

assert figures_path.exists(), msg
assert arrays_path.exists(), msg
assert models_path.exists(), msg

class MultiLayerNet(nn.Module):
    def __init__(self, *neurons):
        super(MultiLayerNet, self).__init__()
        if len(neurons) < 3:
            raise Exception('You have to provide at least three layes!')
        self.linears = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))])
    def forward(self, x):
        for layer in self.linears:
            x = torch.tanh(layer(x))
        return x

class DeepEnergyMethod:
    def __init__(self, model, energy):
        self.model = model.to(dev)
        self.energy = energy
        
    def train_model(self, data, dirichlet, neumann,
                    shape, LHD, dxdydz=None, 
                    neu_axis=None, lr=0.5, epochs=20,
                    fb=np.array([[0, 0, 0]]),
                    ventricle_geometry=False):
        dxdydz = dxdydz if dxdydz is not None else np.array(LHD) / (np.array(shape) - 1)
        if len(dxdydz) == 3:
            dxdydz_neu = (torch.tensor(dxdydz[neu_axis[0]]).double().to(dev), 
                          torch.tensor(dxdydz[neu_axis[1]]).double().to(dev))
        elif len(dxdydz) == 5:
            dxdydz_neu = (torch.from_numpy(dxdydz[3]).double().to(dev),
                          torch.from_numpy(dxdydz[4]).double().to(dev))
        else:
            raise Exception('')
        dxdydz_dom = (torch.tensor(dxdydz[0]).double().to(dev),
                      torch.tensor(dxdydz[1]).double().to(dev),
                      torch.tensor(dxdydz[2]).double().to(dev))

        self.x = torch.from_numpy(data).float().to(dev)
        self.x.requires_grad_(True)
        fb = torch.from_numpy(fb).float().to(dev)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
    

        # boundary
        dirBC_coords = torch.from_numpy(dirichlet['coords']).float().to(dev)
        dirBC_coords.requires_grad_(True)
        dirBC_values = torch.from_numpy(dirichlet['values']).float().to(dev)

        neuBC_coords = torch.from_numpy(neumann['coords']).float().to(dev)
        neuBC_coords.requires_grad_(True)
        neuBC_values = torch.from_numpy(neumann['values']).float().to(dev)

        # use different integration functions based on geometry 
        integral2D = simps2D_LV if ventricle_geometry else simps2D
        integral3D = simps3D_LV if ventricle_geometry else simps3D        
        # integral2D = simps2D
        # integral3D = simps3D

        self.losses = torch.zeros(epochs).to(dev)
        self.eval_losses = []
        prev_loss = torch.tensor([0.0]).to(dev)
        start_time = time.time()

        # nn = self.model.linears[0].out_features
        # nl = len(self.model.linears) - 1
        # j = 0
        # while Path(models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}').exists():
        #     j += 1
        
        for i in range(epochs):
            def closure():
                # internal loss
                u_pred = self(self.model, self.x)
                u_pred.double()

                IntEnergy = self.energy(u_pred, self.x)
                internal_loss = integral3D(IntEnergy, dx=dxdydz_dom[0], dy=dxdydz_dom[1], dz=dxdydz_dom[2], shape=shape)

                # dirichlet loss
                dir_pred = self(self.model, dirBC_coords)
                bc_dir = loss_squared_sum(dir_pred, dirBC_values)
                dir_loss = torch.sum(bc_dir)

                # neumann loss
                neu_pred = self(self.model, neuBC_coords)
                bc_neu = torch.bmm((neu_pred + neuBC_coords).unsqueeze(1), neuBC_values.unsqueeze(2))

                neu_loss = integral2D(bc_neu, dx=dxdydz_neu[0], dy=dxdydz_neu[1], shape=[shape[neu_axis[1]], shape[neu_axis[0]]])

                # external loss
                body_f = torch.matmul(u_pred.unsqueeze(1), fb.unsqueeze(2))
                
                body_loss = integral3D(body_f, dx=dxdydz_dom[0], dy=dxdydz_dom[1], dz=dxdydz_dom[2], shape=shape)
                external_loss = body_loss + neu_loss

                # total loss
                loss = internal_loss + dir_loss - external_loss
                optimizer.zero_grad()
                loss.backward()

                # save losses as attributes for printing and plotting
                self.internal_loss = internal_loss
                self.external_loss = external_loss
                self.energy_loss = loss

                self.current_loss = loss
                return loss

            optimizer.step(closure)

            loss_change = torch.abs(self.current_loss - prev_loss)
            prev_loss = self.current_loss

            # if i == 50:
            #     original_change = loss_change
            #     lowest_change = original_change
            #     best_epoch = i
            #     torch.save(self.model.state_dict(), 
            #                 models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}')
            # elif i > 50:
            #     # store model if loss change decreases by a factor of 10
            #     if loss_change <= 0.1*lowest_change:
            #         lowest_change = loss_change
            #         best_epoch = i
            #         torch.save(self.model.state_dict(), 
            #                    models_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{shape[-1]}_{j}')

            # if eval_data:
            #     eval_shape = [len(eval_data[0]), len(eval_data[1]), len(eval_data[2])]
            #     _, u_eval, xyz_eval = self.evaluate_model(eval_data[0], eval_data[1], eval_data[2], True)
            #     eval_internal = self.energy(u_eval, xyz_eval)
            #     eval_loss1 = simps3D(eval_internal, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)

            #     eval_BF = torch.matmul(u_eval.unsqueeze(1), fb.unsqueeze(2))
            #     eval_loss2 = simps3D(eval_BF, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=eval_shape)
            #     self.eval_loss = eval_loss1 - eval_loss2

            #     print(f'Iter: {i:3d}, Energy: {self.energy_loss.item():10.5f}, Int: {self.internal_loss:10.5f}, Ext: {self.external_loss:10.5f}, Eval loss: {self.eval_loss:10.5f}, Loss_change: {loss_change.item():8.5f}')
            #     self.losses.append([self.current_loss.detach().cpu(), self.eval_loss.detach().cpu()])
            # else:
            # print(f'Iter: {i+1:3d}, ' +
            #       f'Energy: {self.energy_loss.item():10.5f}, ' +
            #       f'Int: {self.internal_loss:10.5f}, ' +
            #       f'Ext: {self.external_loss:10.5f}, ' +
            #       f'Loss_change: {loss_change.item():10.5f}')
            self.losses[i] = self.current_loss

        # print(f'Model at epoch {best_epoch:3d} stored with energy change: {lowest_change:8.5f}, ')

    def __call__(self, model, x):
        u = model(x).to(dev)
        Ux, Uy, Uz = x[:, 0] * u.T.unsqueeze(1)
        u_pred = torch.cat((Ux.T, Uy.T, Uz.T), dim=-1)
        return u_pred

    def evaluate_model(self, x, y, z, return_pred_tensor=False):
        raise NotImplementedError("You have to implement an 'evaluate_model' method in the subclass!")

def loss_squared_sum(input, target):
    return torch.sum((input - target)**2, dim=1) / input.shape[1]*input.data.nelement()

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

def simps2D(U, dx=None, dy=None, shape=None):
    Nx, Ny = shape
    U = U.flatten().reshape(Nx, Ny)
    return simpson(simpson(U, dx=dy), dx=dx)

def simps3D(U, dx=None, dy=None, dz=None, shape=None):
    Nx, Ny, Nz = shape
    U3D = U.flatten().reshape(Nx, Ny, Nz)
    return simpson(simpson(simpson(U3D, dx=dz), dx=dy), dx=dx)

def simps2D_LV(U, dx, dy, shape):
    Nx, Ny = shape
    U = U.flatten().reshape(Nx, Ny)
    return simpson(simpson(U, x=dx, axis=0), x=dy)

def simps3D_LV(U, dx, dy, dz, shape):
    Nx, Ny, Nz = shape
    U3D = U.flatten().reshape(Nx, Ny, Nz)
    return simpson(simpson(simpson(U3D, x=dx, axis=0), x=dy, axis=0), x=dz)

def write_vtk_v2(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    if isinstance(U, dict):
        gridToVTK(filename, xx, yy, zz, pointData=U)
    else:
        gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

def write_vtk_LV(filename, x_space, y_space, z_space, U):
    # xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    xx, yy, zz = x_space, y_space, z_space
    if isinstance(U, dict):
        gridToVTK(filename, xx, yy, zz, pointData=U)
    else:
        gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})