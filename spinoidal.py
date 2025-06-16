import numpy as np
import ufl
from dolfinx import mesh, fem, nls
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
import time

# Parameters
nx, ny = 64, 64
dt = 1e-3
T = 0.1
M = 1.0
kappa = 1e-2
theta = 1.0  # implicit Euler

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

# Function spaces: use mixed formulation (c, mu)
P1 = fem.FunctionSpace(domain, ("CG", 1))
ME = fem.FunctionSpace(domain, ufl.MixedElement([P1.ufl_element(), P1.ufl_element()]))

# Trial/test functions
u = fem.Function(ME)     # solution at t_{n+1}
u0 = fem.Function(ME)    # solution at t_n
v = ufl.TestFunction(ME)

c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)
vc, vmu = ufl.split(v)

# Double-well free energy derivative
def dfdc(c):
    return 4 * c * (c - 1) * (2 * c - 1)  # from f(c) = c^2 (1-c)^2

# Weak form
F0 = (c - c0) / dt * vc * ufl.dx + M * ufl.dot(ufl.grad(mu), ufl.grad(vc)) * ufl.dx
F1 = mu * vmu * ufl.dx - dfdc(c) * vmu * ufl.dx + kappa * ufl.dot(ufl.grad(c), ufl.grad(vmu)) * ufl.dx
F = F0 + F1

# Time stepping
u_sol = fem.Function(ME)
problem = fem.petsc.NonlinearProblem(F, u, bcs=[])
solver = nls.petsc.NewtonSolver(domain.comm, problem)
solver.rtol = 1e-6

# Initial condition
x = domain.geometry.x
c_init = 0.5 + 0.01 * np.random.randn(len(x))
with u0.sub(0).vector.localForm() as loc:
    loc.set(c_init)

u.x.array[:] = u0.x.array  # initialize u

# Output file
with XDMFFile(domain.comm, "spinodal.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u.sub(0), 0.0)

    t = 0.0
    step = 0
    while t < T:
        t += dt
        step += 1
        u0.x.array[:] = u.x.array  # update old solution
        solver.solve(u)
        u_sol.x.array[:] = u.x.array
        file.write_function(u.sub(0), t)
        print(f"Step {step}, Time {t:.4f}")

print("Done!")
