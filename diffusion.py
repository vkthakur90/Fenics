from mpi4py import MPI
import numpy as np
import dolfinx as dfx
import ufl
from petsc4py import PETSc

# 1. Create a quadrilateral mesh on unit square with 32x32 cells
mesh = dfx.mesh.create_unit_square(
    MPI.COMM_WORLD, 32, 32, cell_type=dfx.mesh.CellType.quadrilateral
)

# 2. Define function space with CG2 basis (quadratic continuous Lagrange)
V = dfx.fem.FunctionSpace(mesh, ("CG", 2))

# 3. Define Dirichlet boundary condition: u = 0 on the boundary
u_bc = dfx.fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0.0)

def boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1)

bdofs = dfx.fem.locate_dofs_geometrical(V, boundary)
bc = dfx.fem.dirichletbc(u_bc, bdofs)

# 4. Define variational problem: -Δu = 1 (Poisson with unit source)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dfx.fem.Constant(mesh, PETSc.ScalarType(1.0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 5. Solve linear problem with Dirichlet BC
problem = dfx.fem.petsc.LinearProblem(a, L, bcs=[bc],
                                      petsc_options={"ksp_type": "cg"})
uh = problem.solve()

# 6. Save solution and mesh to XDMF for visualization
with dfx.io.XDMFFile(mesh.comm, "solution_quad_cg2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)
