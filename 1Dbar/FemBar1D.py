import dolfin

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 1000

mesh = dolfin.IntervalMesh(N, -1, L)

V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
u = dolfin.Function(V)
v = dolfin.TestFunction(V)

eps = dolfin.grad(u)

def Psi(e):
    return pow(1 + e, 3 / 2) - (3 / 2) * e - 1

# fig, ax = plt.subplots()
# e = np.linspace(-1, 2.0, 50)
# ax.plot(e, Psi(e))
# fig.savefig("psi.png")

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

dolfin.solve(virtual_work == 0, u, bcs=bc, 
             solver_parameters={"newton_solver": 
                                {"absolute_tolerance": 1e-9, 
                                 "relative_tolerance": 1e-9,
                                 'linear_solver': 'mumps'
                                 }})

x = np.linspace(-1, L, 20)
us = np.array([u(xi) for xi in x])
fig, ax = plt.subplots()
ax.plot(x, us)
fig.savefig("u.png")


dx = x[1] - x[0]
epsilon = np.zeros(len(us))
epsilon[:-1] = (us[:-1] - us[1:])/dx
energy = Psi(epsilon)
# print(np.sum(energy)*dx) # - x*us)*dx)
print(dolfin.assemble(Psi(eps[0,0])*dolfin.dx))
