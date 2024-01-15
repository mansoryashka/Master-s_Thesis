import dolfin
import ufl
import numpy as np

l = 4
h = 1 
d = 1
N = 25

E = 1000
nu = 0.3

lmbd = E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))


# define mesh and domain w/trial and test functions
mesh = dolfin.BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(l, h, d), 4*N, N, N)

V = dolfin.VectorFunctionSpace(mesh, 'P', 1)
uh = dolfin.TrialFunction(V)
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

# define epsilon and sigma
def epsilon(u):
    return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u):
    return lmbd*dolfin.tr(epsilon(u))*dolfin.Identity(3) + 2*mu*epsilon(u)
# def sigma(F):
#     return dolfin.det(F)*mu * F * F.T + dolfin.det(F)*(lmbd*dolfin.ln(dolfin.det(F)) - mu)*dolfin.Identity(3)

# equations for the PDE
f = dolfin.Constant((0.0, -5.0, 0.0))

F = dolfin.grad(u) + dolfin.Identity(3)
# a = dolfin.inner(sigma(F), epsilon(v))*dolfin.dx
a = dolfin.inner(ufl.nabla_div(sigma(uh)), v)*dolfin.dx
L = dolfin.dot(f, v)*ds(1) # + dolfin.dot(dolfin.Constant((0, 0, 0)), v)*dolfin.ds


#J = dolfin.det(F)
#C =F.T * F
#I1 = dolfin.tr(C)
#E = 0.5 * (C - dolfin.Identity(3))

# psi = 1/2*lmbd*dolfin.ln(J)**2 - mu*dolfin.ln(J) + 1/2*mu*(I1 - 3)
#psi = 1/2*lmbd*dolfin.tr(E)**2 + mu*dolfin.tr(E*E) # + 1/2*mu*(I1 - 3)

# dolfin.info(dolfin.LinearVariationalSolver.default_parameters(), True)

#energy = psi*dolfin.dx(domain=mesh) - dolfin.dot(f, v)*ds(1)
#total_virtual_work = dolfin.derivative(energy, u, v)

# dolfin.solve(total_virtual_work == 0, u, bc,
#              solver_parameters={'newton_solver':
#                                 {'absolute_tolerance': 1e-6,
#                                 'linear_solver': 'mumps'}})
dolfin.solve(a == L, u, bcs=bc,
            solver_parameters={"linear_solver": "mumps"})

dolfin.File('output/3dbeam_lin.pvd') << u


x = np.linspace(0, l, 4*N)
y = np.linspace(0, h, N)
z = np.linspace(0, d, N)
u_fem = np.zeros((3, 4*N, N, N))

for i in range(4*N):
    for j in range(N):
        for k in range(N):
            u_fem[:, i, j, k] = u(x[i], y[j], z[k])

np.save('u_fem', u_fem)