using LinearAlgebra

include("ADMM/admm_solver.jl")

# Define objective functions : J(x) := x'Px + q'x
P = [1.0 0; 0 0.5]
q = [-1; -1]

# Define constraints : l <= Ax <= u
A = Matrix(1.0I, 2, 2)
l = [-1; -1]
u = [2; 2]

# Set parameters
rho = 0.1
sigma = 0.1
alpha = 0.5
iter_max = 100
param = [rho, sigma, alpha, iter_max]

x_opt = admm_solver(P, q, A, l, u, param)

println(x_opt)
