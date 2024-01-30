import dolfin
import ufl
import numpy as np

l = 4
h = 1 
d = 1
N = 15

E = 1000
nu = 0.3

lmbd = E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))


# define mesh and domain w/trial and test functions
mesh = dolfin.BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(l, h, d), 4*N, N, N)

V = dolfin.VectorFunctionSpace(mesh, 'P', 1)
uh = dolfin.TrialFunction(V)
u = dolfin.Function(V)  
u2 = dolfin.Function(V)  
v = dolfin.TestFunction(V)
v2 = dolfin.TestFunction(V)

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
# E = 0.5 * (C - dolfin.Identity(3))

f = dolfin.Constant((0.0, -5.0, 0.0))
# f2 = J*dolfin.inv(F).T*f

# # define epsilon and sigma
# def epsilon(u):
#     return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# def sigma(u):
#     return lmbd*dolfin.tr(epsilon(u))*dolfin.Identity(3) + 2*mu*epsilon(u)

def sigma(u):
    F = dolfin.nabla_grad(u) + dolfin.Identity(3)
    J = dolfin.det(F)
    P =  mu * F + (lmbd * dolfin.ln(J) - mu) * dolfin.inv(F.T)
    sigma = 1/J * P * F.T
    return sigma

a = dolfin.inner(sigma(u), dolfin.grad(v))*dolfin.dx(domain=mesh)
L = dolfin.inner(f, v)*ds(1)

J2 = dolfin.derivative(a, u)
dolfin.solve(J2 == L, u, bcs=bc,
            solver_parameters={"linear_solver": "mumps"})

F = dolfin.grad(u2) + dolfin.Identity(3)
J = dolfin.det(F)
C = F.T * F
I1 = dolfin.tr(C)

psi = 0.5*lmbd*dolfin.ln(J)**2 - mu*dolfin.ln(J) + 0.5*mu*(I1 - 3)
energy = psi*dolfin.dx(domain=mesh) #- dolfin.dot(f, u)*ds(1)
total_internal_work = dolfin.derivative(energy, u2, v2)
total_virtual_work = total_internal_work - dolfin.inner(f, v2)*ds(1)

dolfin.solve(total_virtual_work == 0, u2, bc,
             solver_parameters={'newton_solver': {
                                # 'absolute_tolerance': 1e-6,
                                'linear_solver': 'mumps'}})

dolfin.File('output/3dbeam_sig.pvd') << u
dolfin.File('output/3dbeam_sig2.pvd') << u2


# breakpoint()

# x = np.linspace(0, l, 4*N)
# y = np.linspace(0, h, N)
# z = np.linspace(0, d, N)
# u_fem = np.zeros((3, 4*N, N, N))

# for i in range(4*N):
#     for j in range(N):
#         for k in range(N):
#             u_fem[:, i, j, k] = u(x[i], y[j], z[k])

# np.save('u_fem', u_fem)