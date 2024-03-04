import dolfin
import ufl
import numpy as np

l = 4
h = 1 
d = 1
N_test = 30

E = 1000
nu = 0.3

lmbd = E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))

# dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

def FEM_3D(N):
    # define mesh and domain w/trial and test functions
    mesh = dolfin.BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(l, h, d), 4*N, N, N)

    V = dolfin.VectorFunctionSpace(mesh, 'P', 1)
    u = dolfin.Function(V)  
    v = dolfin.TestFunction(V)

    neumann_domain = dolfin.MeshFunction("size_t", mesh, 2)
    neumann_domain.set_all(0)
    dolfin.CompiledSubDomain("near(x[0], side) && on_boundary", side=4.0, tol=10e-10).mark(neumann_domain, 1)
    ds = dolfin.Measure("ds", subdomain_data=neumann_domain)

    # boundary condtions
    def boundary(x, on_boundary):
        return on_boundary and x[0] < 1e-10

    bc = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), boundary)

    # F = dolfin.grad(u) + dolfin.Identity(3)
    # J = dolfin.det(F)
    # B = F * F.T
    # C = F.T * F
    # I1 = dolfin.tr(C)

    F = dolfin.grad(u) + dolfin.Identity(3)
    J = dolfin.det(F)
    C = F.T * F
    I1 = dolfin.tr(C)
    # E = 0.5 * (C - dolfin.Identity(3))

    f = dolfin.Constant((0.0, -5.0, 0.0))
    # Fb = J * dolfin.inv(F).T * f        # body force in reference configuration
    # f2 = J*dolfin.inv(F).T*f

    # # define epsilon and sigma
    # def epsilon(u):
    #     return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    # def sigma(u):
    #     return lmbd*dolfin.tr(epsilon(u))*dolfin.Identity(3) + 2*mu*epsilon(u)

    # def sigma(u):
    #     F = dolfin.nabla_grad(u) + dolfin.Identity(3)
    #     J = dolfin.det(F)
    #     P =  mu * F + (lmbd * dolfin.ln(J) - mu) * dolfin.inv(F.T)
    #     sigma = 1/J * P * F.T
    #     return sigma

    # #a = dolfin.inner(sigma(u), dolfin.grad(v))*dolfin.dx(domain=mesh)
    # F = dolfin.variable(dolfin.grad(u) + dolfin.Identity(3))
    # J = dolfin.det(F)
    # C = F.T * F
    # I1 = dolfin.tr(C)

    # psi = 0.5*lmbd*dolfin.ln(J)**2 - mu*dolfin.ln(J) + 0.5*mu*(I1 - 3)
    # P = dolfin.diff(psi, F)
    # L = dolfin.inner(f, v)*ds(1)

    # J2 = dolfin.derivative(dolfin.inner(P, dolfin.grad(v)) * dolfin.dx(domain=mesh), u)
    # dolfin.solve(J2 == L, u, bcs=bc,
    #             solver_parameters={"linear_solver": "mumps"})

    psi = 0.5*lmbd*dolfin.ln(J)**2 - mu*dolfin.ln(J) + 0.5*mu*(I1 - 3)
    energy = psi*dolfin.dx(domain=mesh) - dolfin.dot(f, u)*dolfin.dx(domain=mesh)
    total_internal_work = dolfin.derivative(energy, u, v)
    total_virtual_work = total_internal_work #- dolfin.inner(f, v)*ds(1)

    dolfin.solve(total_virtual_work == 0, u, bc,
                solver_parameters={'newton_solver': {
                                    # 'absolute_tolerance': 1e-6,
                                    'linear_solver': 'mumps'}})

    dolfin.File('output/FEMBeam3D_noJ.pvd') << u

    x = np.linspace(0, l, 4*N_test+2)[1:-1]
    y = np.linspace(0, h, N_test+2)[1:-1]
    z = np.linspace(0, d, N_test+2)[1:-1]
    u_fem = np.zeros((3, 4*N_test, N_test, N_test))

    for i in range(4*N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, i, j, k] = u(x[i], y[j], z[k])

    np.save(f'stored_arrays/u_fem_N{N}', u_fem)

    print(dolfin.assemble(psi*dolfin.dx))               # 6.924290983627352
    print(dolfin.assemble(dolfin.dot(f, u)*dolfin.dx))  # 14.651664345327262

if __name__ == '__main__':
    # for N in [5, 10, 15, 20, 25, 30]:
    for N in [20]:
        print('N = ', N)
        FEM_3D(N)