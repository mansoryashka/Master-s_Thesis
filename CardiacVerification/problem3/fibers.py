import numpy as np
import matplotlib.pyplot as plt
import torch



import sys
from problem3 import generate_fibers, define_domain
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150

if __name__ == '__main__':
    N = 41; M = 9
    shape = [N, M, N]

    domain, dirichlet, neumann = define_domain(N, M, n_cond=15, plot=False)
    f0, s0, n0 = generate_fibers(N, M, alpha_endo=60, alpha_epi=-60)
    plot_domain = np.copy(domain).reshape((N, M, N, 3))
    plot_f0 = np.copy(2*f0).reshape((3, N, M, N))
    plot_s0 = np.copy(2*s0).reshape((3, N, M, N))
    plot_n0 = np.copy(2*n0).reshape((3, N, M, N))
    n1 = int(N/2)

    n2 = 6
    n3 = -1
    dn1 = 1
    dn2 = 2
    # m1 = 
    endocardium = plot_domain[:, 0]
    epicardium = plot_domain[:, -1]
    print(epicardium.shape)
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.view_init(elev=0, azim=85)

    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.view_init(elev=0, azim=85)
    fig2 = plt.figure(figsize=(8, 8))
    ax3 = fig2.add_subplot(projection='3d')

    ax3.set_box_aspect((2, 2, 1))
    ax3.axis('off')

    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_zlabel('$z$')
    ax3.view_init(elev=0, azim=-10)
    ax3.set_zlim((4.5, 5.5))


    # ax4.quiver(plot_domain[0::dn1, :, 2::dn2, 0], 
    #            plot_domain[0::dn1, :, 2::dn2, 1], 
    #            plot_domain[0::dn1, :, 2::dn2, 2], 
    #             plot_f0[0, 0::dn1, :, 2::dn2], 
    #             plot_f0[1, 0::dn1, :, 2::dn2], 
    #             plot_f0[2, 0::dn1, :, 2::dn2], alpha=0.5, color='tab:blue', label='s0')
    # plt.show()
    # exit()

    ax1.plot_surface(epicardium[..., 0], epicardium[..., 1],  epicardium[..., 2], cmap='autumn', alpha=0.3)
    ax1.quiver(plot_domain[0:n1:dn1, -1, 2::dn2, 0],
               plot_domain[0:n1:dn1, -1, 2::dn2, 1],
               plot_domain[0:n1:dn1, -1, 2::dn2, 2],
                plot_f0[0, 0:n1:dn1, -1, 2::dn2],
                plot_f0[1, 0:n1:dn1, -1, 2::dn2], 
                plot_f0[2, 0:n1:dn1, -1, 2::dn2], alpha=0.8)
    

    ax2.plot_surface(endocardium[..., 0], endocardium[..., 1],  endocardium[..., 2], cmap='autumn', alpha=0.3)
    ax2.plot_surface(epicardium[..., 0], epicardium[..., 1],  epicardium[..., 2], cmap='autumn', alpha=0.05)
    ax2.quiver(plot_domain[0:n1:dn1, 0, 2:-1:dn2, 0], 
               plot_domain[0:n1:dn1, 0, 2:-1:dn2, 1], 
               plot_domain[0:n1:dn1, 0, 2:-1:dn2, 2], 
                plot_f0[0, 0:n1:dn1, 0, 2:-1:dn2], 
                plot_f0[1, 0:n1:dn1, 0, 2:-1:dn2], 
                plot_f0[2, 0:n1:dn1, 0, 2:-1:dn2], alpha=0.8)

    
    plot_f0 = plot_f0/6
    plot_s0 = plot_s0/6
    plot_n0 = plot_n0/6
    ax3.plot_surface(endocardium[n2-2:n2+1:dn1, n3-1:, 0], 
                     endocardium[n2-2:n2+1:dn1, n3-1:, 1],  
                     endocardium[n2-2:n2+1:dn1, n3-1:, 2] + 0.4, cmap='autumn', alpha=0.10)
    ax3.plot_surface( epicardium[n2-2:n2+1:dn1, n3-1:, 0], 
                      epicardium[n2-2:n2+1:dn1, n3-1:, 1],  
                      epicardium[n2-2:n2+1:dn1, n3-1:, 2] + 0.4, cmap='autumn', alpha=0.10)
    ax3.text(endocardium[5, n3-1, 0], 
             endocardium[5, n3-1, 1] - 1.2,  
             endocardium[5, n3-1, 2] + 0.2, 'Endocardium', fontdict={'size': 22})
    ax3.text(epicardium[4, n3-1, 0], 
             epicardium[4, n3-1, 1],  
             epicardium[4, n3-1, 2] + 0.2, 'Epicardium', fontdict={'size': 22})
    
    ax3.quiver(plot_domain[n2-1:n2:dn1, :, n3:, 0], 
               plot_domain[n2-1:n2:dn1, :, n3:, 1], 
               plot_domain[n2-1:n2:dn1, :, n3:, 2], 
                plot_f0[0, n2-1:n2:dn1, :, n3:], 
                plot_f0[1, n2-1:n2:dn1, :, n3:], 
                plot_f0[2, n2-1:n2:dn1, :, n3:], alpha=1, color='tab:blue', label='f0')
    # ax3.quiver(plot_domain[n2-1:n2:dn1, :, n3:, 0], 
    #            plot_domain[n2-1:n2:dn1, :, n3:, 1], 
    #            plot_domain[n2-1:n2:dn1, :, n3:, 2], 
    #             plot_s0[0, n2-1:n2:dn1, :, n3:], 
    #             plot_s0[1, n2-1:n2:dn1, :, n3:], 
    #             plot_s0[2, n2-1:n2:dn1, :, n3:], alpha=1, color='tab:green', label='s0')
    # ax3.quiver(plot_domain[n2-1:n2:dn1, :, n3:, 0], 
    #            plot_domain[n2-1:n2:dn1, :, n3:, 1], 
    #            plot_domain[n2-1:n2:dn1, :, n3:, 2], 
    #             plot_n0[0, n2-1:n2:dn1, :, n3:], 
    #             plot_n0[1, n2-1:n2:dn1, :, n3:], 
    #             plot_n0[2, n2-1:n2:dn1, :, n3:], alpha=1, color='tab:red', label='n0')
    # ax1.legend()
    # ax2.legend()
    fig.tight_layout()
    fig2.tight_layout()
    # ax3.legend()
    fig.savefig('figures/fibers.pdf')
    fig2.savefig('figures/fibers2.pdf')
    plt.show()