import dolfin
import ufl


def subplus(x):
    r"""
    Ramp function
    .. math::
       \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    r"""
    Heaviside function
    .. math::
       \mathcal{H}(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


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


def transverse_holzapfel_ogden(
    F: ufl.Coefficient,
    f0: dolfin.Function,
    a: float = 2.280,
    b: float = 9.726,
    a_f: float = 1.685,
    b_f: float = 15.779,
) -> ufl.Coefficient:
    r"""Transverse isotropic version of the model from Holzapfel and Ogden [1]_.

    The strain energy density function is given by
    .. math::
        \Psi(I_1, I_{4\mathbf{f}_0}, I_{4\mathbf{s}_0}, I_{8\mathbf{f}_0\mathbf{s}_0})
        = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
        + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
        \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
       
    where
    .. math::
        (x)_+ = \max\{x,0\}
    and
    .. math::
        \mathcal{H}(x) = \begin{cases}
            1, & \text{if $x > 0$} \\
            0, & \text{if $x \leq 0$}
        \end{cases}
    is the Heaviside function.

    .. [1] Holzapfel, Gerhard A., and Ray W. Ogden.
        "Constitutive modelling of passive myocardium:
        a structurally based framework for material characterization.
        "Philosophical Transactions of the Royal Society of London A:
        Mathematical, Physical and Engineering Sciences 367.1902 (2009):
        3445-3475.

    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    f0 : dolfin.Function
        Fiber direction
    a : float, optional
        Material parameter, by default 2.280
    b : float, optional
        Material parameter, by default 9.726
    a_f : float, optional
        Material parameter, by default 1.685
    b_f : float, optional
        Material parameter, by default 15.779

    Returns
    -------
    ufl.Coefficient
        Strain energy density
    """

    C = F.T * F
    I1 = dolfin.tr(C)
    I4f = dolfin.inner(C * f0, f0)

    return (a / (2.0 * b)) * (dolfin.exp(b * (I1 - 3)) - 1.0) + (
        a_f / (2.0 * b_f)
    ) * heaviside(I4f - 1) * (dolfin.exp(b_f * subplus(I4f - 1) ** 2) - 1.0)


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
    return kappa * (J - 1)**2

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
    nu = 0.3
    mu = E / (2*(1 + nu))

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
    with dolfin.XDMFFile("output/u.xdmf") as u_file:
        u_file.write_checkpoint(
            u,
            function_name="u",
            time_step=0.0,
            encoding=dolfin.XDMFFile.Encoding.HDF5,
            append=False,
        )

    N_test = 20
    x = y = z = np.linspace(0, 1, N_test+2)[1:-1]
    u_fem = np.zeros((3, N_test, N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            for k in range(N_test):
                u_fem[:, i, j, k] = u(x[i], y[j], z[k])
    np.save(f'stored_arrays/u_fem{N}', u_fem)

    print(dolfin.assemble(elastic_energy*dolfin.dx))
    print(dolfin.assemble(dolfin.inner(u, n)*ds(right_marker)))
    return u


if __name__ == '__main__':
    import numpy as np
    Ns = [5, 10, 15, 20]
    for N in Ns:
        # print('N = ', N)
        u = FEM_Cube(N)