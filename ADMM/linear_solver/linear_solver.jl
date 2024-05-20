include("pcg.jl")

function linear_solver(A, b, x0)
  return pcg(A, b, x0)
end
