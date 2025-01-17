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
K0 = 200 * md * ones(n)
K = deepcopy(K0)
K[:,:,1:2:end] .*= 100

ϕ = 0.25
model = jutulModel(n, d, ϕ, K1to3(K))

## simulation time steppings
tstep = 50 * ones(10)
tot_time = sum(tstep)

## injection & production
inj_loc = (15, 1, 10) .* d
irate = 5e-3
q = jutulVWell(irate, (450., 30.); startz = 270., endz = 330.)

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
Trans = KtoTrans(CartesianMesh(model), K1to3(K))
Trans0 = KtoTrans(CartesianMesh(model), K1to3(K0))
@time states = S(log.(Trans), q)
@time states0 = S(log.(Trans0), q)

## plotting
fig=figure(figsize=(20,12));
subplot(1,2,1);
imshow(reshape(Saturations(states.states[end]), n[1], n[end])'); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(Pressure(states.states[end]), n[1], n[end])'); colorbar(); title("pressure")

## plotting
fig=figure(figsize=(20,12));
subplot(1,2,1);
imshow(reshape(Saturations(states0.states[end]), n[1], n[end])'); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(Pressure(states0.states[end]), n[1], n[end])'); colorbar(); title("pressure")

exist_co2 = sum(Saturations(states.states[end]) .* states.states[end].state[:Reservoir][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
inj_co2 = JutulDarcyRules.ρCO2 * q.irate * JutulDarcyRules.day * sum(tstep)

norm(exist_co2-inj_co2)/norm(exist_co2+inj_co2)
