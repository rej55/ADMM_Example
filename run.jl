using LinearAlgebra

include("ADMM/admm_solver.jl")

# Define objective functions : J(x) := 1/2 * x'Px + q'x
P = [1.0 0 0; 0 0.5 0;0 0 0.5]
q = [-1; -1; -1]

# Define constraints : l <= Ax <= u
A = Matrix(1.0I, 3, 3)
l = [-1; -1; -1]
u = [200000; 200000; 200000]

# Set parameters
rho = 0.1
sigma = 0.1
alpha = 0.5
iter_max = 100
param = [rho, sigma, alpha, iter_max]

output = admm_solver(P, q, A, l, u, param)

println(output.solution)
