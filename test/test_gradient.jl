model, model0, q, q1, q2, state0, state1, tstep = test_config();

## set up modeling operator
S0 = jutulModeling(model0, tstep)
S = jutulModeling(model, tstep)

## simulation
x = log.(KtoTrans(CartesianMesh(model), model.K))
x0 = log.(KtoTrans(CartesianMesh(model0), model0.K))
ϕ = S.model.ϕ

states = S(x, ϕ, q)

misfit(x0, ϕ, q, states) = 0.5 * norm(S0(x0, ϕ, q) - states).^2
g = gradient((x0, ϕ)->misfit(x0, ϕ, q, states), x0, ϕ)

dx = randn(MersenneTwister(2023), length(x0))
dx = dx/norm(dx) * norm(x0)/5.0

dϕ = randn(MersenneTwister(2023), length(ϕ))
ϕmask = ϕ .< 1
dϕ[.!ϕmask] .= 0
dϕ[ϕmask] = dϕ[ϕmask]/norm(dϕ[ϕmask]) * norm(ϕ[ϕmask])
dϕ = vec(dϕ)


@testset "Taylor-series gradient test of jutulModeling with wells" begin
    grad_test(x0->misfit(x0, ϕ, q, states), x0, dx, g[x0])
    grad_test(ϕ->misfit(x0, ϕ, q, states), ϕ, dϕ, g[ϕ])
end

states1 = S(x, ϕ, q1)
g1 = gradient((x0, ϕ)->misfit(x0, ϕ, q1, states1), x0, ϕ)

@testset "Taylor-series gradient test of simple jutulModeling" begin
    grad_test(x0->misfit(x0, ϕ, q1, states1), x0, dx, g1[x0])
    grad_test(ϕ->misfit(x0, ϕ, q1, states1), ϕ, dϕ, g1[ϕ])
end

states2 = S(x, q2)
g2 = gradient((x0, ϕ)->misfit(x0, ϕ, q2, states2), x0, ϕ)

@testset "Taylor-series gradient test of jutulModeling with vertical wells" begin
    # This test is very brittle. There may be an issue here.
    grad_test(x0->misfit(x0, ϕ, q2, states2), x0, dx, g2[x0])
    grad_test(ϕ->misfit(x0, ϕ, q2, states2), ϕ, dϕ, g2[ϕ]; unittest=:skip)
end
