import dolfin
import ufl
import numpy as np

l = 4
h = 1 
d = 1
N_test = 30
N_strain = 21

E = 1000
nu = 0.3

lmbd = E * nu / ((1 + nu)*(1 - 2*nu))
mu = E / (2*(1 + nu))

print(lmbd, mu)
#exit()

dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

def FEM_3D(N):
    # define mesh and domain w/trial and test functions
    mesh = dolfin.BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(l, h, d), 4*N, N, N)
    # breakpoint()

    V = dolfin.VectorFunctionSpace(mesh, 'P', 1)
    u = dolfin.Function(V)  
    v = dolfin.TestFunction(V)

    # boundary condtions
    def boundary(x, on_boundary):
        return on_boundary and x[0] < 1e-10

    bc = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), boundary)

    left = dolfin.CompiledSubDomain("near(x[0], 0)")

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)
    left.mark(ffun, 1)
    with dolfin.XDMFFile("output/ffun.xdmf") as ffun_file:
        ffun_file.write(ffun)

    exit()
    F = dolfin.grad(u) + dolfin.Identity(3)
    J = dolfin.det(F)
    C = F.T * F
    I1 = dolfin.tr(C)
    # E = 0.5 * (C - dolfin.Identity(3))

    f = dolfin.Constant((0.0, -5.0, 0.0))


    psi = 0.5*lmbd*dolfin.ln(J)**2 - mu*dolfin.ln(J) + 0.5*mu*(I1 - 3)
    # psi = 100*(J-1)**2 + 0.5*mu*(I1 - 3)
    energy = psi*dolfin.dx - dolfin.dot(f, u)*dolfin.dx
    total_internal_work = dolfin.derivative(energy, u, v)
    total_virtual_work = total_internal_work

    dolfin.solve(total_virtual_work == 0, u, bc,
                solver_parameters={'newton_solver': {
                                    # 'absolute_tolerance': 1e-6,
                                    'linear_solver': 'mumps'}})


    P = mu * F + (lmbd * dolfin.ln(dolfin.det(F)) - mu) * dolfin.inv(F).T
    secondPiola = dolfin.inv(F) * P
    Sdev = secondPiola - (1./3)*dolfin.tr(secondPiola)*dolfin.Identity(3) # deviatoric stress
    von_Mises = dolfin.sqrt(3./2*dolfin.inner(Sdev, Sdev))
    # u = dolfin.project(u, V)
    V = dolfin.FunctionSpace(mesh, "Lagrange", 1)
    # W = dolfin.TensorFunctionSpace(mesh, "Lagrange", 1)
    VonMises = dolfin.project(von_Mises, V)

    u.rename('Displacement', '')
    VonMises.rename('VonMises stress', '')

    # outfile = dolfin.XDMFFile('output/FemBeam3D.xdmf')
    # outfile.parameters['flush_output'] = True
    # outfile.parameters['functions_share_mesh'] = True
    # outfile.write(u, 0.0)
    # outfile.write(VonMises, 0.0)

    x = np.linspace(0, l, 4*N_test+2)[1:-1]
    y = np.linspace(0, h, N_test+2)[1:-1]
    z = np.linspace(0, d, N_test+2)[1:-1]
    u_fem = np.zeros((3, N_test, 4*N_test, N_test))

    for i in range(4*N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, j, i, k] = u(x[i], y[j], z[k])

    x_strain = np.linspace(0, l, 4*N_strain+2)[1:-1]
    y_strain = np.linspace(0, h, N_strain)
    z_strain = np.linspace(0, d, N_strain)
    u_strain = np.zeros((3, N_strain, 4*N_strain, N_strain))

    for i in range(4*N_strain):
        for j in range(N_strain):
            for k in range(N_strain):
                u_strain[:, j, i, k] = u(x_strain[i], y_strain[j], z_strain[k])



    np.save(f'stored_arrays/u_strain', u_strain)
    # np.save(f'stored_arrays/u_fem_N{N}', u_fem)
    return u

    print(dolfin.assemble(psi*dolfin.dx))               # 6.924290983627352
    print(dolfin.assemble(dolfin.dot(f, u)*dolfin.dx))      # 14.651664345327262

    # print(dolfin.assemble(psi*dolfin.dx))               # 2.873653069727217
    # print(dolfin.assemble(dolfin.dot(f, u)*ds(1)))      # 5.996554979767974

if __name__ == '__main__':
    # for N in [5, 10, 15, 20, 25, 30]:
    for N in [5]:
        print('N = ', N)
        FEM_3D(N)