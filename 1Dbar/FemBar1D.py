import dolfin
import numpy as np
import matplotlib.pyplot as plt

dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

def FEM_1D_Beam(N, x0 = -1, L = 1):
    mesh = dolfin.IntervalMesh(N, x0, L)

    V = dolfin.VectorFunctionSpace(mesh, "P", 1)
    u = dolfin.Function(V)
    v = dolfin.TestFunction(V)

    eps = dolfin.grad(u)

    def Psi(e):
        return pow(1 + e, 3 / 2) - (3 / 2) * e - 1

    psi = Psi(eps[0, 0])
    X = dolfin.SpatialCoordinate(mesh)
    X = dolfin.Expression(("x[0]",), degree=1)
    f = X
    t = 0.0

    ffun = dolfin.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    ffun.set_all(0)

    right = dolfin.CompiledSubDomain("near(x[0], L)", L=L)
    right_marker = 1
    right.mark(ffun, right_marker)

    left = dolfin.CompiledSubDomain("near(x[0], -1)")
    left_marker = 2
    left.mark(ffun, left_marker)

    ds = dolfin.ds(domain=mesh, subdomain_data=ffun)
    dx = dolfin.dx(domain=mesh)

    energy = psi * dx - dolfin.inner(f, u) * dx

    virtual_work = dolfin.derivative(energy, u, v) \
                + t * dolfin.inner(u, v) * ds(right_marker)

    bc = dolfin.DirichletBC(V, dolfin.Constant([0.0]), ffun, left_marker)

    dolfin.solve(virtual_work == 0, u, bcs=bc,)

    return u

if __name__ == '__main__':
    from DemBar1D import exact, dexact, L2norm
    L = 1.0
    x0 = -1.0
    Ntest = 200
    x = np.linspace(x0, L, Ntest+2, endpoint=True)[1:-1]
    dx = x[1] - x[0]

    u_ex = exact(x)
    du_ex = dexact(x)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    norms = {}
    #### Skal jeg kjøre FEM for de samme verdiene som DEM? ###
    for N in [10, 100, 1000]:
        u = FEM_1D_Beam(N)

        us = np.array([u(xi) for xi in x])
        abs_err = np.abs((us - u_ex))
        ax.semilogy(x, abs_err, label=f'N = {N}')

        ### Trolig bedre å ha i tabell
        du = np.gradient(us, dx)
        abs_err2 = np.abs((du - du_ex))
        ax2.semilogy(x, abs_err2, label=f'N = {N}')
        # print(f'N: {N}, L2norm = {L2norm(us, u_ex)}')
        norms[N] = L2norm(us, u_ex, dx)
    
    # np.save('u_fem', us)

    ax.set_xlabel('$X$')
    ax.set_ylabel('Absolute error')
    ax.legend(loc='lower right')
    # fig.savefig('figures/error_u_fem.pdf')

    ax2.set_xlabel('$X$')
    ax2.set_ylabel('Absolute error')
    ax2.axhline(y=3e-5)
    ax2.legend(loc='lower right')
    # fig2.savefig('figures/error_du_fem.pdf')

    fig, ax = plt.subplots()
    ax.plot(x, u_ex, label='Exact')
    ax.plot(x, us, label='FEM', linestyle='--')
    # plt.show()
    # fig.savefig("output/u.pdf")

    Ns = np.asarray([10, 50, 100, 500, 1000, 5000, 10000])
    e = np.zeros(Ns.shape)
    for i, N in enumerate(Ns):
        u = FEM_1D_Beam(N)
        us = np.array([u(xi) for xi in x])

        e[i] = L2norm(us, u_ex, dx)
        print(L2norm(us, u_ex, dx))
        print(e[i])

    print(np.log(e[1:]/e[:-1])/np.log((1/Ns[1:])/(1/Ns[:-1])))

    fig3, ax3 = plt.subplots()
    ax3.loglog(Ns, e, '--o')
    ax3.set_ylabel('Relative error of derformation')
    ax3.set_xlabel('N')
    # fig3.savefig('figures/convergence.pdf')
    plt.show()
    ### et forsøk  på å regne ut konvergensrate? ###
    # prev_key=1
    # prev_val=1
    # for i, (key, val) in enumerate(norms.items()):
    #     # print(key, prev_key)
    #     print(np.log(prev_val/val)/np.log((1/prev_key)/(1/key)))
    #     prev_key, prev_val = key, val
    #     # print(f'{key:5d} & {val:8.3g}')
    # plt.show()
