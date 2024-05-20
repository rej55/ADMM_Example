using LinearAlgebra
using Plots

include("ADMM/admm_solver.jl")
include("linear_solver/pcg.jl")
include("linear_solver/preconditioner/diagonal_scaling.jl")

# Define objective functions : J(x) := 1/2 * x'Px + q'x
P = [1.0 0 0; 0 0.01 0;0 0 0.1]
q = [-1; -1; -1]

# Define constraints : l <= Ax <= u
A = Matrix(1.0I, 3, 3)
l = [-1; -1; -1]
u = [200000; 200000; 200000]

# Set parameters
rho = 0.1
sigma = 0.1
alpha = 0.5
iter_max = 1000
param = [rho, sigma, alpha, iter_max]

# Solve
output_normal = admm_solver(P, q, A, l, u, param, (A, b, x0) -> pcg(A, b, x0, (A, b) -> b))
output_diag = admm_solver(P, q, A, l, u, param, (A, b, x0) -> pcg(A, b, x0, (A, b) -> diagonal_scaling(A, b)))
output_inv = admm_solver(P, q, A, l, u, param, (A, b, x0) -> A\b)

println(output_diag.solution)

# Plot results
plt = plot(output_normal.objective_values, markershape=:circle, label="normal")
plot!(plt, output_diag.objective_values, markershape=:circle, label="diag")
plot!(plt, output_inv.objective_values, markershape=:circle, label="inv")
display(plt)
