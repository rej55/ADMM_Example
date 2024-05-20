using LinearAlgebra
include("linear_solver/linear_solver.jl")

function admm_solver(P, q, A, l, u, param)
  # Deploy parameters
  rho, sigma, alpha, iter_max = param

  # Dimensions
  ny, nx = size(A)

  # Define initial guess
  x = zeros(nx)
  x_tilde = zeros(nx)
  z = zeros(ny)
  y = zeros(ny)
  v = zeros(ny)

  for i = 1:iter_max
    # Solve linear system
    sol = linear_solver([P + sigma * Matrix(1.0I, nx, nx) transpose(A); A -1.0 / rho *  Matrix(1.0I, ny, ny)], [sigma * x - q; z - 1.0 / rho * y], [x_tilde;v])
    x_tilde = sol[1:nx]
    v = sol[(nx + 1):(nx + ny)]

    # Calculate z_tilde and x
    z_tilde = z + 1.0 / rho * (v - y)
    x = alpha * x_tilde + (1 - alpha) * x

    # Calculate z_next and project onto the feasible region
    z_next = alpha * z_tilde + (1 - alpha) * z + 1.0 / rho * y
    for i = 1:ny
      if z_next[i] < l[i]
        z_next[i] = l[i]
      elseif z[i] > u[i]
        z_next[i] = u[i]
      end
    end

    # Update y and z
    y = y + rho * (alpha * z_tilde + (1 - alpha) * z - z_next)
    z = z_next

    # Print
    println("i = ", i, " : x = ", x, ", J = ", 0.5 * transpose(x) * A * x + transpose(q) * x)
  end
  return x
end
