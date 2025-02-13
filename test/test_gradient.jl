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
    states = S0(x0, ϕ0, q; config)
    sat_misfit = 0.5 * sum(sum((s[:Reservoir][:Saturations][1, :] .- sr[:Reservoir][:Saturations][1, :]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    pres_misfit = 0.5 * sum(sum((s[:Reservoir][:Pressure] .- sr[:Reservoir][:Pressure]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    # sat_misfit
    sat_misfit + pres_misfit * 1e-14
end

function misfit_simple(x0, ϕ0, q, states_ref)
    states = S0(x0, ϕ0, q; config=config1)
    sat_misfit = 0.5 * sum(sum((s[:Saturations][1, :] .- sr[:Saturations][1, :]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    pres_misfit = 0.5 * sum(sum((s[:Pressure] .- sr[:Pressure]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    # sat_misfit
    sat_misfit + pres_misfit * 1e-14
end

rng = MersenneTwister(2023)

function sample_dx()
    dx = randn(rng, length(x0))
    dx = dx/norm(dx) * norm(x0)/5.0
end

dx = sample_dx()

ϕmask = ϕ .< 1
function sample_dϕ()
    ϕfactor = randn(rng, model.n[1:2:3])
    kernel = [0.25, 0.5, 0.25]
    kernel = kernel * kernel'
    kernel_idx = CartesianIndices(kernel) .- CartesianIndex((2, 2))
    dϕ = zeros(size(ϕfactor))
    for c in CartesianIndices(ϕfactor)
        if c.I[1] == 1 || c.I[1] == model.n[1] || c.I[2] == 1 || c.I[2] == model.n[3]
            continue
        end
        c_kernel = kernel_idx .+ c
        dϕ[c_kernel] .= kernel * ϕfactor[c_kernel]
    end
    dϕ = vec(dϕ)
    dϕ[.!ϕmask] .= 0
    # dϕ[ϕmask] = ϕ[ϕmask] .* exp.(ϕfactor[ϕmask])
    dϕ[ϕmask] = ϕfactor[ϕmask]/norm(ϕfactor[ϕmask]) * norm(ϕ[ϕmask])
end
dϕ = sample_dϕ()

@time states_ref, case, sim, x0_0 = S(x, ϕ, q; return_extra=true)

config = simulator_config(sim)
for m in Jutul.submodels_symbols(case.model)
    config[:tolerances][m][:default] = 1e-10
end
config[:linear_solver].config.relative_tolerance = 1e-10
config[:info_level] = -1

@time states_ref = S(x, ϕ, q; config)

v_initial = misfit(x0, ϕ0, q, states_ref)
@show v_initial

misfit_dx = x0->misfit(x0, ϕ, q, states_ref)
misfit_dϕ = ϕ0->misfit(x, ϕ0, q, states_ref)
misfit_dboth = (x0,ϕ0)->misfit(x0, ϕ0, q, states_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)

@testset "Taylor-series gradient test of jutulModeling with wells" begin
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dϕ, ϕ0, dϕ, gϕ[1])
end

@time states1_ref, case1, sim1, x0_1 = S(x, ϕ, q1; return_extra=true)

config1 = simulator_config(sim1)
config1[:tolerances][:default] = 1e-10

misfit_dx = x0->misfit_simple(x0, ϕ, q1, states1_ref)
misfit_dϕ = ϕ0->misfit_simple(x, ϕ0, q1, states1_ref)
misfit_dboth = (x0,ϕ0)->misfit_simple(x0, ϕ0, q1, states1_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)
@show vx vϕ v norm(gx) norm(gϕ) norm(g)

@testset "Taylor-series gradient test of simple jutulModeling" begin
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dϕ, ϕ0, dϕ, gϕ[1])
end

states2_ref = S(x, q2; config)

misfit_dx = x0->misfit(x0, ϕ, q2, states2_ref)
misfit_dϕ = ϕ0->misfit(x, ϕ0, q2, states2_ref)
misfit_dboth = (x0,ϕ0)->misfit(x0, ϕ0, q2, states2_ref)

vx, gx = withgradient(misfit_dx, x0)
vϕ, gϕ = withgradient(misfit_dϕ, ϕ0)
v, g = withgradient(misfit_dboth, x0, ϕ0)
@show vx vϕ v norm(gx) norm(gϕ) norm(g)

@testset "Taylor-series gradient test of jutulModeling with vertical wells" begin
    grad_test(misfit_dx, x0, dx, gx[1])
    grad_test(misfit_dϕ, ϕ0, dϕ, gϕ[1])
end
