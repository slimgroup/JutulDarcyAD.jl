using DrWatson
@quickactivate "JutulDarcyRules"

using JutulDarcyRules
using Flux
using LinearAlgebra
using Optim

nx = 30
ny = 1
nz = 15

dims = (nx, ny, nz)
g = CartesianMesh(dims, (30.0, 30.0, 30.0) .* dims)
nc = prod(dims)
Kx = 20 * md * ones(nx, nz)
Kxtrue = deepcopy(Kx)
Kxtrue[:, 6:8] .*= 6

Trans_true = KtoTrans(g, K1to3(Kxtrue))

Kx = 10 * md * ones(prod(g.dims))
@time grad = gradient(() -> .5 * norm(KtoTrans(g, K1to3(Kx))-Trans_true)^2f0, Flux.params(Kx))[Kx]

# ## Set up L-BFGS
function fg(F, G, x)
       grads = gradient(Flux.params(x)) do
              F = .5 * norm(KtoTrans(g, K1to3(x))-Trans_true)^2 / norm(Trans_true)^2
       end
       G .= grads[x]
       return F
end

method = LBFGS()
optimopt = Optim.Options(iterations = 200, store_trace = true, show_trace = true, show_every = 1)
result = optimize(Optim.only_fg!(fg), Kx, method, optimopt)
Kxinv = result.minimizer

println(norm(Kxinv-vec(Kxtrue))/norm(Kxtrue))

@time Kxinv1 = TransToK(g, Trans_true)

println(norm(Kxinv1-K1to3(Kxtrue))/norm(K1to3(Kxtrue)))
println(norm(KtoTrans(g, Kxinv1)-Trans_true)/norm(Trans_true))
