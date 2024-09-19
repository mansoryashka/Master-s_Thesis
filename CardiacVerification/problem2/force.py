import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../..")
from problem2 import define_domain
# import seaborn as sns 
# sns.set()

import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

if __name__ == '__main__':
    N = 24; M = 3
    half_domain = int(N/2-1)
    # half_domain = N
    domain, _, neumann = define_domain(N, M, n_cond=1)
    neu_coords = neumann['coords'].reshape((N, N, 3))
    x_perp = neu_coords[..., 0] / 7
    y_perp = neu_coords[..., 1] / 7
    z_perp = neu_coords[..., 2] / 17
    neu_coords = neumann['values'].reshape((N, N, 3))
    x_perp_scaled = neu_coords[..., 0]
    y_perp_scaled = neu_coords[..., 1]
    z_perp_scaled = neu_coords[..., 2]
    endocardium = domain.reshape((N, M, N, 3))[:, 0]
    x_endo = endocardium[..., 0]
    y_endo = endocardium[..., 1]
    z_endo = endocardium[..., 2]

    fig1 = plt.figure(figsize=(18, 6))
    # plt.style.use('default')
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.view_init(elev=-90)
    ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.view_init(elev=0, azim=75)
    ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
    ax3.axis('off')
    ax3.set_aspect('equal')
    ax3.view_init(elev=0, azim=75)

    # plot endocardial surface
    ax1.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
    ax2.quiver(x_endo[:half_domain, :],
               y_endo[:half_domain, :],
               z_endo[:half_domain, :],
               x_perp[:half_domain, :],
               y_perp[:half_domain, :],
               z_perp[:half_domain, :], alpha=1)

    ax2.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    ax1.quiver(x_endo, 
               y_endo, 
               z_endo, 
               x_perp, 
               y_perp, 
               z_perp, alpha=1)
    ax3.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    ax3.quiver(x_endo[:half_domain, :], 
               y_endo[:half_domain, :], 
               z_endo[:half_domain, :], 
        x_perp_scaled[:half_domain, :], 
        y_perp_scaled[:half_domain, :], 
        z_perp_scaled[:half_domain, :], alpha=1)
    
    fig1.tight_layout()


    fig2 = plt.figure(figsize=(14, 8))
    # plt.style.use('default')
    ax3 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3.set_aspect('equal')
    ax3.view_init(elev=0, azim=75)
    ax3.axis('off')
    ax4 = fig2.add_subplot(1, 2, 2, projection='3d')
    ax4.axis('off')
    ax4.set_aspect('equal')
    ax4.view_init(elev=0, azim=75)

    # plot endocardial surface
    ax3.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
    ax3.quiver(x_endo[:half_domain, :], 
               y_endo[:half_domain, :], 
               z_endo[:half_domain, :], 
               x_perp[:half_domain, :],
               y_perp[:half_domain, :], 
               z_perp[:half_domain, :], alpha=1)
    ax4.plot_surface(x_endo, y_endo, z_endo, cmap='autumn', alpha=.1)
    # ax.plot_surface(x_epi, y_epi, z_epi, cmap='autumn', alpha=.1)
    ax4.quiver(x_endo[:half_domain, :], 
               y_endo[:half_domain, :], 
               z_endo[:half_domain, :], 
        x_perp_scaled[:half_domain, :], 
        y_perp_scaled[:half_domain, :], 
        z_perp_scaled[:half_domain, :], alpha=1)

    fig2.tight_layout()
    plt.show()
    fig1.savefig('figures/force1.pdf')
    fig2.savefig('figures/force2.pdf')
    plt.close()