using LinearAlgebra

function diagonal_scaling(A, b)
  # Diagonal scaling
  x = b ./ diag(A)

  return x
end
