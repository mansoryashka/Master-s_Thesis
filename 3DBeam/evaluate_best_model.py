import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

import sys
sys.path.insert(0, "..")
from DEM import DeepEnergyMethod, MultiLayerNet, dev, L2norm3D, write_vtk_v2
from DemBeam3D import energy

current_path = Path().cwd()
models_path = current_path / 'trained_models'
arrays_path = current_path / 'stored_arrays'

N_test = 30
L = 4
H = 1
D = 1
dx = L/N_test
dy = dz = H/N_test

x = np.linspace(0, L, 4*N_test + 2)[1:-1]
y = np.linspace(0, H, N_test + 2)[1:-1]
z = np.linspace(0, D, N_test + 2)[1:-1]

u_fem30 = np.load(arrays_path / 'u_fem_N30.npy')

""" Run 1 """
N = 30
lr = .1
num_layers = [2, 3, 4, 5]
num_neurons = [30, 40, 50, 60]

cum_norms = np.zeros((4, 4))
run_path = models_path / 'run1'
for i, nl in enumerate(num_layers):
    print(i)
    for k, nn in enumerate(num_neurons):
        best_norm = np.inf
        for j in range(20):
            current_model_path = run_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{N}_{j}'
            assert current_model_path.exists()
            model = MultiLayerNet(3, *[nn]*nl, 3)
            model.load_state_dict(torch.load(current_model_path))

            DemBeam = DeepEnergyMethod(model, energy)
            current_pred = DemBeam.evaluate_model(x, y, z)
            norm = L2norm3D(current_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
            cum_norms[i, k] += norm
            if norm < best_norm:
                best_norm = norm
                best_model = current_model_path
            # print(f'N: {N}, lr: {lr:6f}, # layers: {nl:5d}, # neurons: {nn:5d}, norm: {norm:10.7f}')
        # print(best_norm, best_model.relative_to(run_path))

print(cum_norms/20, '\n')
# print(cum_norms/40)

print(
""" Run 2 """
)
N = 30
lrs = [0.001, 0.01, 0.1, 0.5]
num_layers = [2, 3, 4, 5]
nn = 50
cum_norms = np.zeros((4, 4))
run_path = models_path / 'run2'
for i, lr in enumerate(lrs):
    for k, nl in enumerate(num_layers):
        best_norm = np.inf
        for j in range(20):
            current_model_path = run_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{N}_{j}'
            assert current_model_path.exists()
            model = MultiLayerNet(3, *[nn]*nl, 3)
            model.load_state_dict(torch.load(current_model_path))

            DemBeam = DeepEnergyMethod(model, energy)
            current_pred = DemBeam.evaluate_model(x, y, z)
            norm = L2norm3D(current_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
            cum_norms[i, k] += norm
            if norm < best_norm:
                best_norm = norm
                best_model = current_model_path
            # print(f'N: {N}, lr: {lr:6f}, # layers: {nl:5d}, # neurons: {nn:5d}, norm: {norm:10.7f}')
        # print(best_norm, best_model.relative_to(run_path))
print(cum_norms/20, '\n')
# print(cum_norms/40)

print(
""" Run 3 """
)
N = 30
lrs = [.01, .05, .1, .5]
num_neurons = [20, 30, 40, 50]
nl = 2

cum_norms = np.zeros((4, 4))
run_path = models_path / 'run3'
for i, lr in enumerate(lrs):
    for k, nn in enumerate(num_neurons):
        best_norm = np.inf
        for j in range(20):
            current_model_path = run_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{N}_{j}'
            # assert current_model_path.exists()
            if not current_model_path.exists():
                print(current_model_path.relative_to(run_path), ' does not exist!!')
                continue
            model = MultiLayerNet(3, *[nn]*nl, 3)
            model.load_state_dict(torch.load(current_model_path))

            DemBeam = DeepEnergyMethod(model, energy)
            current_pred = DemBeam.evaluate_model(x, y, z)
            norm = L2norm3D(current_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)
            cum_norms[i, k] += norm
            if norm < best_norm:
                best_norm = norm
                best_model = current_model_path
            # print(f'N: {N}, lr: {lr:6f}, # layers: {nl:5d}, # neurons: {nn:5d}, norm: {norm:10.7f}')
#         print(best_norm, best_model.relative_to(run_path))
print(cum_norms/20, '\n')
# print(cum_norms/40)

print(
""" Run 4 """
)
Ns = [20, 30, 40, 50, 60]
lrs = [.01, .05, .1, .5]
# Ns = [40]
# lrs = [.01]
nl = 2
nn = 50
cum_norms = np.zeros((4, 5))
run_path = models_path / 'run4'
# for lr in lrs:
#     for N in Ns:
for i, lr in enumerate(lrs):
    for k, N in enumerate(Ns):
        best_norm = np.inf
        for j in range(20):
            current_model_path = run_path / f'model_lr{lr}_nn{nn}_nl{nl}_N{N}_{j}'
            # assert current_model_path.exists()
            if not current_model_path.exists():
                print(current_model_path.relative_to(run_path), ' does not exist!!')
                continue
            model = MultiLayerNet(3, *[nn]*nl, 3)
            model.load_state_dict(torch.load(current_model_path))
            model.eval()

            DemBeam = DeepEnergyMethod(model, energy)
            current_pred = DemBeam.evaluate_model(x, y, z)
            norm = L2norm3D(current_pred - u_fem30, 4*N_test, N_test, N_test, dx, dy, dz)

            cum_norms[i, k] += norm
            if norm < best_norm:
                best_norm = norm
                best_model = current_model_path
            # print(f'N: {N}, lr: {lr:6f}, # layers: {nl:5d}, # neurons: {nn:5d}, norm: {norm:10.7f}')
        # print(best_norm, best_model.relative_to(run_path))

print(cum_norms/20, '\n')
# print(cum_norms/40)

"""
[[0.00673402 0.00435929 0.00329564 0.00292316 0.0027569 ]
 [0.00678354 0.00369826 0.00302654 0.00276949 0.00284289]
 [0.00654576 0.00376218 0.00302105 0.00298395 0.00290571]
 [0.00646983 0.00359294 0.0031326  0.00319676 0.00352643]]




"""