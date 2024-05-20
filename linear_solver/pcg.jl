using LinearAlgebra

# Preconditioned conjugate gradient
function pcg(A, b, x0, precond)
  # Set convergence criteria
  epsilon = 0.0001

  # Set initial guess
  x = x0
  r = b - A * x0
  p = precond(A, r)
  z = p

  for i = 1:1000
    alpha = dot(r, z) / dot(A * p, p)
    x = x + alpha * p
    rn = r - alpha * A * p
    zn = precond(A, rn)
    beta = dot(rn, zn) / dot(r, z)
    p = zn + beta * p
    r = rn
    z = zn

    # Check convergence
    if norm(r) < epsilon
      # println("--- [pcg] Converged! ||r|| = ", dot(r, r), ", iter = ", i)
      break
    end
  end

  return x
end
