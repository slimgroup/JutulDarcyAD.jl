function test_config()
    n = (30, 1, 15)
    d = (30.0, 30.0, 30.0)

    ## permeability
    K0 = 40 * md * ones(n)
    ϕ = 0.25
    K = deepcopy(K0)
    K[:,:,1:2:end] .*= 40

    model0 = jutulModel(n, d, ϕ, K1to3(K0))
    model = jutulModel(n, d, ϕ, K1to3(K))

    ## simulation time steppings
    tstep = 50 * ones(10)

    ## injection & production
    inj_loc = (15, 1, 10) .* d
    prod_loc = (30, 1, 10) .* d
    irate = 5e-3
    q = jutulForce(irate, inj_loc)
    q1 = jutulSource(irate, [inj_loc])
    q2 = jutulVWell(irate, inj_loc[1:2]; startz = 9 * d[3], endz = 11 * d[3])
    state0 = jutulState(JutulDarcyRules.setup_well_model(model, q, tstep)[3])
    state1 = JutulDarcyRules.setup_simple_model(model, q1, tstep)[3]
    return model, model0, q, q1, q2, state0, state1, tstep
end

include("grad_test.jl")
