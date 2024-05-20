using LinearAlgebra

include("precond.jl")

# Preconditioned conjugate gradient
function pcg(A, b, x0)
  # Set convergence criteria
  epsilon = 0.000001

  # Set initial guess
  x = x0
  r = b - A * x0
  p = precond(A, r)
  z = p

  for i = 1:100
    alpha = dot(r, z) / dot(A * p, p)
    x = x + alpha * p
    rn = r - alpha * A * p
    zn = precond(A, rn)
    beta = dot(rn, zn) / dot(r, z)
    p = zn + beta * p
    r = rn
    z = zn

    # Check convergence
    if dot(r, r) < epsilon
      println("[pcg] converged! ||r|| = ", dot(r, r), ", iter = ", i)
      break
    end
  end

  return x
end