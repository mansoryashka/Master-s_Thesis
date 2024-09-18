import dolfin
import ufl
import numpy as np

def neo_hookean(F: ufl.Coefficient, mu: float = 15.0) -> ufl.Coefficient:
    r"""Neo Hookean model

    .. math::
        \Psi(F) = \frac{\mu}{2}(I_1 - 3)

    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    mu : float, optional
        Material parameter, by default 15.0

    Returns
    -------
    ufl.Coefficient
        Strain energy density
    """

    C = F.T * F
    I1 = dolfin.tr(C)
    return 0.5 * mu * (I1 - 3)


def active_stress_energy(
    F: ufl.Coefficient, f0: dolfin.Function, Ta: dolfin.Constant
) -> ufl.Coefficient:
    """Active stress energy

    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    f0 : dolfin.Function
        Fiber direction
    Ta : dolfin.Constant
        Active tension

    Returns
    -------
    ufl.Coefficient
        Active stress energy
    """

    J = dolfin.det(F)
    I4f = dolfin.inner(F * f0, F * f0)
    return 0.5 * Ta / J * (I4f - 1)


def compressibility(F: ufl.Coefficient, kappa: float = 1e3) -> ufl.Coefficient:
    r"""Penalty for compressibility

    .. math::
        \kappa (J \mathrm{ln}J - J + 1)

    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    kappa : float, optional
        Parameter for compressibility, by default 1e3

    Returns
    -------
    ufl.Coefficient
        Energy for compressibility
    """
    J = dolfin.det(F)
    # return kappa * (J * dolfin.ln(J) - J + 1)
    return kappa / 2 * (J - 1)**2

def FEM_Cube(N):
    # Create a Unit Cube Mesh
    mesh = dolfin.UnitCubeMesh(N, N, N)

    # Function space for the displacement
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    # The displacement
    u = dolfin.Function(V)
    # Test function for the displacement
    u_test = dolfin.TestFunction(V)

    # Compute the deformation gradient
    F = dolfin.grad(u) + dolfin.Identity(3)

    # Active tension
    Ta = dolfin.Constant(1.0)
    # Set fiber direction to be constant in the x-direction
    f0 = dolfin.Constant([1.0, 0.0, 0.0])

    E = 1000
    nu = 0.9
    mu = E / (2 * (1 + nu))

    # Collect the contributions to the total energy (here using the Holzapfel Ogden model)
    elastic_energy = (
        # transverse_holzapfel_ogden(F, f0=f0)
        neo_hookean(F, mu=mu)
        + active_stress_energy(F, f0, Ta)
        + compressibility(F)
    )
    # Here we can also use the Neo Hookean model instead
    # elastic_energy = neo_hookean(F) + active_stress_energy(F, f0, Ta) + compressibility(F)

    # Define some subdomain. Here we mark the x = 0 plane with the marker 1
    left = dolfin.CompiledSubDomain("near(x[0], 0)")
    left_marker = 1
    # And we define the Dirichlet boundary condition on this side
    # We specify that the displacement should be zero in all directions
    bcs = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), left)

    # We also define a subdomain on the opposite wall
    right = dolfin.CompiledSubDomain("near(x[0], 1)")
    # and we give a marker of two
    right_marker = 2


    # We create a facet function for marking the facets
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    # We set all values to zero
    ffun.set_all(0)
    # Then then mark the left and right subdomains
    left.mark(ffun, left_marker)
    right.mark(ffun, right_marker)

    # We can also save this file to xdmf and visualize it in Paraview
    # with dolfin.XDMFFile("output/ffun.xdmf") as ffun_file:
    #     ffun_file.write(ffun)


    # Now we can form to total internal virtual work which is the
    # derivative of the energy in the system
    quad_degree = 2
    internal_virtual_work = dolfin.derivative(
        elastic_energy * dolfin.dx(metadata={"quadrature_degree": quad_degree}), u, u_test
    )

    # We can also apply a force on the right boundary using a Neumann boundary condition
    # traction = dolfin.Constant(1.0)
    traction = dolfin.Constant(-0.5)
    Norm = dolfin.FacetNormal(mesh)
    n = traction * Norm
    ds = dolfin.ds(domain=mesh, subdomain_data=ffun)
    external_virtual_work = dolfin.inner(u_test, n) * ds(right_marker)

    # The total virtual work is the sum of the internal and external virtual work
    total_virtual_work = internal_virtual_work + external_virtual_work

    # The we solve for the displacement u
    dolfin.solve(total_virtual_work == 0, u, bcs=[bcs],
                solver_parameters={'newton_solver': {
                    # 'absolute_tolerance': 1e-6,
                    'linear_solver': 'mumps'}})


    # We can visualize the solution in Paraview
    # with dolfin.XDMFFile("output/u.xdmf") as u_file:
    #     u_file.write_checkpoint(
    #         u,
    #         function_name="u",
    #         time_step=0.0,
    #         encoding=dolfin.XDMFFile.Encoding.HDF5,
    #         append=False,
    #     )

    N_test = 20
    x = y = z = np.linspace(0, 1, N_test+2)[1:-1]
    u_fem = np.zeros((3, N_test, N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, j, i, k] = u(x[i], y[j], z[k])
    np.save(f'stored_arrays/u_fem{N}', u_fem)
    
    N_test = 21
    x = y = z = np.linspace(0, 1, N_test+2)[1:-1]
    u_fem = np.zeros((3, N_test, N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, j, i, k] = u(x[i], y[j], z[k])
    np.save(f'stored_arrays/u_strain_diag', u_fem)

    N_test = 21
    x = np.linspace(0, 1, N_test+2)[1:-1]
    y = z = np.linspace(0, 1, N_test)
    u_fem = np.zeros((3, N_test, N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, j, i, k] = u(x[i], y[j], z[k])
    np.save(f'stored_arrays/u_strain_x', u_fem)

    N_test = 21
    z = np.linspace(0, 1, N_test+2)[1:-1]
    y = x = np.linspace(0, 1, N_test)
    u_fem = np.zeros((3, N_test, N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, j, i, k] = u(x[i], y[j], z[k])
    np.save(f'stored_arrays/u_strain_z', u_fem)

    print(dolfin.assemble(elastic_energy*dolfin.dx))            # -37.3210
    print(dolfin.assemble(dolfin.inner(u, n)*ds(right_marker))) # 0.048729


if __name__ == '__main__':
    # Ns = [5, 10, 15, 20]
    Ns = [20]
    for N in Ns:
        print('N = ', N)
        FEM_Cube(N)