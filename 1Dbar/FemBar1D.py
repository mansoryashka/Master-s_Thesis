import dolfin
import numpy as np
import matplotlib.pyplot as plt

def FEM_1D_Beam(N, x0 = -1, L = 1):
    mesh = dolfin.IntervalMesh(N, x0, L)

    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
    u = dolfin.Function(V)
    v = dolfin.TestFunction(V)

    eps = dolfin.grad(u)

    def Psi(e):
        return pow(1 + e, 3 / 2) - (3 / 2) * e - 1

    # fig, ax = plt.subplots()
    # e = np.linspace(-1, 2.0, 50)
    # ax.plot(e, Psi(e))
    # fig.savefig("output/psi.pdf")

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
                # solver_parameters={"newton_solver": {
                #                     # "absolute_tolerance": 1e-9, 
                #                     # "relative_tolerance": 1e-9,
                #                     'linear_solver': 'mumps'
                #                     }})

    return u

if __name__ == '__main__':
    from DemBar1D import exact, du_exact, L2norm
    L = 1.0
    x0 = -1.0
    Ntest = 200
    x = np.linspace(x0, L, Ntest, endpoint=True)
    dx = x[1] - x[0]

    u_ex = exact(x)
    du_ex = du_exact(x)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    norms = {}
    #### Skal jeg kjøre FEM for de samme verdiene som DEM? ###
    for N in [10, 100, 1000, 10000]:
    # for N in [5, 10, 20, 100]:
        u = FEM_1D_Beam(N)

        us = np.array([u(xi) for xi in x])
        abs_err = np.abs(us - u_ex)
        ax.semilogy(x, abs_err, label=f'N = {N}')

        ### Trolig bedre å ha i tabell
        du = np.gradient(us, dx)
        abs_err2 = np.abs(du - du_ex)
        ax2.semilogy(x, abs_err2, label=f'N = {N}')
        # print(f'N: {N}, L2norm = {L2norm(us, u_ex)}')
        norms[N] = L2norm(us, u_ex)


    ax.legend()
    ax2.legend()
    # plt.show()


    fig, ax = plt.subplots()
    ax.plot(x, u_ex, label='Exact')
    ax.plot(x, us, label='FEM', linestyle='--')
    plt.show()
    # fig.savefig("output/u.pdf")

    ### et forsøk  på å regne ut konvergensrate? ###
    plt.figure()
    prev_key=1
    prev_val=1
    for i, (key, val) in enumerate(norms.items()):
        print(key, prev_key)
        print(np.log10(prev_val/val)/np.log10((1/prev_key)/(1/key)))
        prev_key, prev_val = key, val
    plt.loglog(norms.keys(), norms.values(), '--o')
    
# dx = x[1] - x[0]
# epsilon = np.zeros(len(us))
# epsilon[:-1] = (us[:-1] - us[1:])/dx
# energy = Psi(epsilon)
# print(np.sum(energy)*dx) # - x*us)*dx)
# print(dolfin.assemble(Psi(eps[0,0])*dolfin.dx))
