## A simple 2D example for fluid-flow simulation

using DrWatson
@quickactivate "JutulDarcyRules"

using JutulDarcyRules
using LinearAlgebra
using PythonPlot

## grid size
n = (30, 1, 15)
d = (30.0, 30.0, 30.0)

## permeability
K = 20 * md * ones(n)
ϕ = 0.25 * ones(n)
model = jutulModel(n, d, vec(ϕ), K1to3(K))

## simulation time steppings
tstep = 40 * ones(50)
tot_time = sum(tstep)

## injection & production
inj_loc = (3, 1, 9) .* d
prod_loc = (28, 1, 9) .* d
irate = 5e-3
q = jutulSource(irate, [inj_loc, prod_loc])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
Trans = KtoTrans(CartesianMesh(model), K1to3(K))
@time result = S(log.(Trans), q; info_level=1)
@time result = S(log.(Trans), q)

## plotting
fig=figure(figsize=(20,12));
subplot(1,2,1);
imshow(reshape(Saturations(result.states[end]), n[1], n[end])', vmin=0, vmax=1); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(Pressure(result.states[end]), n[1], n[end])', vmin=minimum(Pressure(result.states[end])), vmax=maximum(Pressure(result.states[end]))); colorbar(); title("pressure")
