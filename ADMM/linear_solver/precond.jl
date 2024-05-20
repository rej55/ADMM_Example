using LinearAlgebra

function precond(A, b)
  # Diagonal scaling
  x = b ./ diag(A)

  return x
end
