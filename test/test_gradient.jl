model, model0, q, q1, q2, state0, state1, tstep = test_config();

## set up modeling operator
S0 = jutulModeling(model0, tstep)
S = jutulModeling(model, tstep)

## simulation
x = log.(KtoTrans(CartesianMesh(model), model.K))
x0 = log.(KtoTrans(CartesianMesh(model0), model0.K))
using Flux: withgradient

ϕ = S.model.ϕ
ϕ0 = S.model.ϕ

function misfit(x0, ϕ0, q, states_ref)
    states = S0(x0, ϕ0, q)
    sat_misfit = 0.5 * sum(sum((s[:Reservoir][:Saturations][1, :] .- sr[:Reservoir][:Saturations][1, :]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    pres_misfit = 0.5 * sum(sum((s[:Reservoir][:Pressure] .- sr[:Reservoir][:Pressure]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    # sat_misfit
    sat_misfit + pres_misfit * 1e-14
end

function misfit_simple(x0, ϕ0, q, states_ref)
    states = S0(x0, ϕ0, q)
    sat_misfit = 0.5 * sum(sum((s[:Saturations][1, :] .- sr[:Saturations][1, :]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    pres_misfit = 0.5 * sum(sum((s[:Pressure] .- sr[:Pressure]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    # sat_misfit
    sat_misfit + pres_misfit * 1e-14
end

dx = randn(MersenneTwister(2023), length(x0))
dx = dx/norm(dx) * norm(x0)/5.0

dϕ = randn(MersenneTwister(2023), length(ϕ))
ϕmask = ϕ .< 1
dϕ[.!ϕmask] .= 0
dϕ[ϕmask] = dϕ[ϕmask]/norm(dϕ[ϕmask]) * norm(ϕ[ϕmask])
dϕ = vec(dϕ)

states_ref, jmodel, state0_, jforces, parameters0, x0_0 = S(x, ϕ, q; return_extra=true)
# JutulDarcy.plot_reservoir(jmodel, vcat([state0_[:Reservoir]], JutulDarcy.ReservoirSimResult(jmodel, states_ref, jforces).states))

v_initial = misfit(x0, ϕ0, q, states_ref)
@show v_initial

misfit_dx = x0->misfit(x0, ϕ, q, states_ref)
misfit_dϕ = ϕ0->misfit(x, ϕ0, q, states_ref)
misfit_dboth = (x0,ϕ0)->misfit(x0, ϕ0, q, states_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)
@show vx vϕ v norm(gx) norm(gϕ) norm(g)

@testset "Taylor-series gradient test of jutulModeling with wells" begin
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dϕ, ϕ0, dϕ, gϕ[1]; h0=5e-1)
end

states1_ref, jmodel1, state1_, jforces1, parameters1, x0_1 = S(x, ϕ, q1; return_extra=true)
# Jutul.plot_interactive(jmodel1, vcat([state1_], states1_ref.states))

misfit_dx = x0->misfit_simple(x0, ϕ0, q1, states1_ref)
misfit_dϕ = ϕ0->misfit_simple(x0, ϕ0, q1, states1_ref)
misfit_dboth = (x0,ϕ0)->misfit_simple(x0, ϕ0, q1, states1_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)
@show vx vϕ v norm(gx) norm(gϕ) norm(g)

@testset "Taylor-series gradient test of simple jutulModeling" begin
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dx, x0, dx, gx[1])
end

states2_ref = S(x, q2)

misfit_dx = x0->misfit(x0, ϕ, q2, states2_ref)
misfit_dϕ = ϕ0->misfit(x, ϕ0, q2, states2_ref)
misfit_dboth = (x0,ϕ0)->misfit(x0, ϕ0, q2, states2_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)
@show vx vϕ v norm(gx) norm(gϕ) norm(g)

@testset "Taylor-series gradient test of jutulModeling with vertical wells" begin
    # This test is very brittle. There may be an issue here.
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dϕ, ϕ0, dϕ, gϕ[1]; h0=5e-1)
end
