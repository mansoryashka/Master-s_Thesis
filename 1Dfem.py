import fenics
import dolfin


L = 1.0
N = 20

mesh = dolfin.IntervalMesh(N, -1, L)

V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
u = dolfin.Function(V)

# F = dolfin.grad(u) + dolfin.Identity(1)
# J = dolfin.det(F)
# C = F.T * F
# I1 = dolfin.tr(C)

# E = 1.0
# nu = 1.0
# lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
# mu = E / (2 * (1 + nu))
# psi = 0.5 * lmbda * dolfin.ln(J) ** 2 - mu * dolfin.ln(J) + 0.5 * mu * (I1 - 3)

eps = dolfin.grad(u)[0, 0]


psi = pow(1 + eps, 2 / 3) - (2 / 3) * eps - 1
X = dolfin.SpatialCoordinate(mesh)
f = X   
t = dolfin.as_vector([0.0])

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

def boundary(x, on_boundary):
    return on_boundary and left
bc = dolfin.DirichletBC(V, dolfin.as_vector([0.0]), boundary)

energy = psi * dx - dolfin.inner(f, u) * dx + dolfin.inner(t, u) * ds(right_marker)

dolfin.solve(energy == 0, u, bcs=[bc])
# vtkfile = dolfin.File("1Dsolution.pvd")
# vtkfile << u
