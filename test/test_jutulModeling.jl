model, model0, q, q1, q2, init_state, init_state1, tstep = test_config();

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
x = log.(KtoTrans(CartesianMesh(model), model.K))
x0 = log.(KtoTrans(CartesianMesh(model0), model0.K))

@testset "Test mass conservation for well modeling" begin
    states = S(x, q)
    for i = 1:length(states.states)
        exist_co2 = sum(Saturations(states.states[i]) .* states.states[i][:Reservoir][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
        inj_co2 = JutulDarcyRules.ρCO2 * q.irate * JutulDarcyRules.day * sum(tstep[1:i])
        @test isapprox(exist_co2, inj_co2) rtol=1e-3
    end
end

@testset "Test mass conservation for simple modeling" begin
    states = S(x, q1)
    for i = 1:length(states.states)
        exist_co2 = sum(Saturations(states.states[i]) .* states.states[i][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
        inj_co2 = JutulDarcyRules.ρCO2 * q.irate * JutulDarcyRules.day * sum(S.tstep[1:i])
        @test isapprox(exist_co2, inj_co2) rtol=1e-3
    end
end

@testset "Test mass conservation for well modeling, different injection rates" begin
    states = S(x, q)
    pre_co2 = sum(Saturations(states.states[end]) .* states.states[end][:Reservoir][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
    q2 = jutulForce(q.irate * 0.5, q.loc)
    S.tstep ./= 2
    states_end = S(x, q2; state0=Dict{Symbol, Any}(:Reservoir=>states.states[end][:Reservoir]))
    for i = 1:length(states_end.states)
        exist_co2 = sum(Saturations(states_end.states[i]) .* states_end.states[i][:Reservoir][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
        inj_co2 = JutulDarcyRules.ρCO2 * q2.irate * JutulDarcyRules.day * sum(S.tstep[1:i])
        @test isapprox(exist_co2-pre_co2, inj_co2) rtol=1e-3
    end
end

@testset "Test mass conservation for vertical well modeling" begin
    states = S(x, q2)
    for i = 1:length(states.states)
        exist_co2 = sum(Saturations(states.states[i]) .* states.states[i][:Reservoir][:PhaseMassDensities][1,:] .* model.ϕ) * prod(model.d)
        inj_co2 = JutulDarcyRules.ρCO2 * q.irate * JutulDarcyRules.day * sum(tstep[1:i])
        @test isapprox(exist_co2, inj_co2) rtol=1e-3
    end
end
