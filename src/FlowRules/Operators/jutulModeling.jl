export jutulModeling

struct jutulModeling{D, T}
    model::jutulModel{D, T}
    tstep::Vector{T}
end

display(M::jutulModeling{D, T}) where {D, T} =
    println("$(D)D jutulModeling structure with $(sum(M.tstep)) days in $(length(M.tstep)) time steps")

function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}

    Transmissibilities = exp.(LogTransmissibilities)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = ignore_derivatives() do
        setup_well_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
    end
    model.models.Reservoir.data_domain[:porosity] = ϕ
    parameters[:Reservoir][:Transmissibilities] = Transmissibilities
    parameters[:Reservoir][:FluidVolume] = prod(S.model.d) .* ϕ
    isnothing(state0) || (state0_[:Reservoir] = get_Reservoir_state(state0))

    ### simulation
    states, report = simulate_ad(state0_, model, tstep, parameters, forces; opt_config=nothing, max_timestep_cuts = 1000, info_level=info_level)
    # sim, config = setup_reservoir_simulator(model, state0_, parameters);
    # states, report = simulate!(sim, tstep, forces = forces, config = config);
    output = jutulStates(states)
    return output
end

function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::jutulSource{D, N};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}

    Transmissibilities = exp.(LogTransmissibilities)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = ignore_derivatives() do
        setup_simple_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
    end
    model.data_domain[:porosity] = ϕ
    parameters[:Transmissibilities] = Transmissibilities
    parameters[:FluidVolume] = prod(S.model.d) .* ϕ
    isnothing(state0) || (state0_ = state0)

    ### simulation
    states, report = simulate_ad(state0_, model, tstep, parameters, forces; opt_config=nothing, max_timestep_cuts = 1000, info_level=info_level)
    return jutulSimpleStates(states)
end

function (S::jutulModeling{D, T})(f::Union{jutulForce{D, N}, jutulVWell{D, N}, jutulSource{D, N}};
    LogTransmissibilities::AbstractVector{T}=KtoTrans(CartesianMesh(S.model), S.model.K), ϕ::AbstractVector{T}=S.model.ϕ,
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}

    return S(LogTransmissibilities, ϕ, f; state0=state0, visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O, info_level=info_level)
end

function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}, jutulSource{D, N}};
    ϕ::AbstractVector{T}=S.model.ϕ,
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}

    return S(LogTransmissibilities, ϕ, f; state0=state0, visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O, info_level=info_level)
end
