import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from torch.autograd import grad

import matplotlib
matplotlib.rcParams['figure.dpi'] = 350

import sys
sys.path.insert(0, "../../3DBeam")
sys.path.insert(0, "../..")
from DemBeam3D import DeepEnergyMethodBeam, train_and_evaluate, MultiLayerNet, write_vtk_v2, dev

def define_domain(N=5):
    u = np.linspace(0, np.pi, N)
    v = np.linspace(0, 2*np.pi, N)
    r1 = 2; r2 = 4
    r = np.linspace(r1, r2, 3)
    print(r); exit()
    R, U, V = np.meshgrid(r, u, v)
    x = R.ravel()*np.outer(np.sin(U), np.cos(V))
    y = R.ravel()*np.outer(np.sin(U), np.sin(V))
    z = R.ravel()*np.outer(np.ones(np.size(V)), np.cos(U))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(x, y, z, s=0.1, alpha=0.05)
    # plt.savefig('domain.pdf')


    x_endo = r1*np.outer(np.cos(v), np.sin(u))
    y_endo = r1*np.outer(np.sin(v), np.sin(u))
    z_endo = r1*np.outer(np.ones(np.size(v)), np.cos(u))

    x_epi = r2*np.outer(np.cos(v), np.sin(u))
    y_epi = r2*np.outer(np.sin(v), np.sin(u))
    z_epi = r2*np.outer(np.ones(np.size(v)), np.cos(u))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('equal')

    ax.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.5)
    ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=0.1)
    plt.savefig('domain.pdf')
define_domain()