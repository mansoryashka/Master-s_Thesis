import dolfin

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 20

mesh = dolfin.IntervalMesh(N, -1, L)

V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
u = dolfin.Function(V)
v = dolfin.TestFunction(V)

# F = dolfin.grad(u) + dolfin.Identity(1)
# J = dolfin.det(F)
# C = F.T * F
# I1 = dolfin.tr(C)

# E = 1.0
# nu = 1.0
# lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
# mu = E / (2 * (1 + nu))
# psi = 0.5 * lmbda * dolfin.ln(J) ** 2 - mu * dolfin.ln(J) + 0.5 * mu * (I1 - 3)

eps = dolfin.grad(u)


def Psi(e):
    return pow(1 + e, 3 / 2) - (3 / 2) * e - 1


fig, ax = plt.subplots()
e = np.linspace(-1, 2.0, 50)
ax.plot(e, Psi(e))
fig.savefig("psi.png")

psi = Psi(eps[0, 0])
X = dolfin.SpatialCoordinate(mesh)
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


bc = dolfin.DirichletBC(V, dolfin.Constant([0.0]), left)

energy = psi * dx - dolfin.inner(f, u) * dx

virtual_work = dolfin.derivative(energy, u, v) + t * dolfin.inner(u, v) * ds(
    right_marker
)

dolfin.solve(virtual_work == 0, u, bcs=bc)


x = np.linspace(-1, L, 20)
us = [u(xi) for xi in x]
fig, ax = plt.subplots()
ax.plot(x, us)
fig.savefig("u.png")
