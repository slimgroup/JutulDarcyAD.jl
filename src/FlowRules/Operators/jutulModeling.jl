export jutulModeling

using ChainRulesCore: @ignore_derivatives
using Flux

struct jutulModeling{D, T}
    model::jutulModel{D, T}
    tstep::Vector{T}
end

display(M::jutulModeling{D, T}) where {D, T} =
    println("$(D)D jutulModeling structure with $(sum(M.tstep)) days in $(length(M.tstep)) time steps")

function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, return_extra=false, kwargs...) where {D, T, N}

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

    # @show keys(state0)
    # isnothing(state0) || (state0_[:Reservoir] = get_Reservoir_state(state0))
    # @show keys(state0_)

    # Set up config for gradient.
    opt_config_params, mapper = @ignore_derivatives begin
        active = Dict{Symbol, Any}(:Reservoir => [:Transmissibilities, :FluidVolume])
        opt_config_params = optimization_config(model, parameters, active)
        for (ki, vi) in opt_config_params
            if ki != :Reservoir
                empty!(vi)
            end
        end
        opt_config_params[:Reservoir][:FluidVolume][:active] = true
        opt_config_params[:Reservoir][:Transmissibilities][:active] = true
        # opt_config_params[:Reservoir][:Transmissibilities][:scaler] = :log

        targets = Jutul.optimization_targets(opt_config_params, model)
        mapper, = Jutul.variable_mapper(model, :parameters, targets = targets, config = opt_config_params)
        lims = Jutul.optimization_limits(opt_config_params, mapper, parameters, model) # Secretly changes config in place.
        opt_config_params, mapper
    end

    x0 = vectorize_variables(model, parameters, mapper, config = opt_config_params)

    ### simulation
    # output = simulate_ad(state0_, model, tstep, parameters, forces; opt_config_params, parameters_ref = parameters, max_timestep_cuts = 1000, info_level=info_level)
    output, case, sim = simulate_ad(state0_, model, tstep, x0, forces; opt_config_params, parameters_ref = parameters, max_timestep_cuts = 1000, info_level=info_level, return_extra=true, kwargs...)

    states, report = output
    # sim, config = setup_reservoir_simulator(model, state0_, parameters);
    # states, report = simulate!(sim, tstep, forces = forces, config = config);
    # output = jutulStates(states)
    if return_extra
        return output, case, sim, x0
    end
    return output
end

function ChainRulesCore.rrule(
    S::jutulModeling{D, T},
    LogTransmissibilities::AbstractVector{T},
    ϕ::AbstractVector{T},
    f::Union{jutulForce{D, N}, jutulVWell{D, N}};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, kwargs...
) where {D, T, N}

    Transmissibilities, LogTransmissibilities_pullback = Flux.pullback((x) -> exp.(x), LogTransmissibilities)
    FluidVolume, ϕ_pullback = Flux.pullback((x) -> prod(S.model.d) .* x, ϕ)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = ignore_derivatives() do
        setup_well_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
    end
    model.models.Reservoir.data_domain[:porosity] = ϕ
    parameters[:Reservoir][:Transmissibilities] = Transmissibilities
    parameters[:Reservoir][:FluidVolume] = FluidVolume
    # isnothing(state0) || (state0_[:Reservoir] = get_Reservoir_state(state0))

    # Set up config for gradient.
    opt_config_params, mapper = @ignore_derivatives begin
        active = Dict{Symbol, Any}(:Reservoir => [:Transmissibilities, :FluidVolume])
        opt_config_params = optimization_config(model, parameters, active)
        for (ki, vi) in opt_config_params
            if ki != :Reservoir
                empty!(vi)
            end
        end
        opt_config_params[:Reservoir][:FluidVolume][:active] = true
        opt_config_params[:Reservoir][:Transmissibilities][:active] = true
        # opt_config_params[:Reservoir][:Transmissibilities][:scaler] = :log

        targets = Jutul.optimization_targets(opt_config_params, model)
        mapper, = Jutul.variable_mapper(model, :parameters, targets = targets, config = opt_config_params)
        lims = Jutul.optimization_limits(opt_config_params, mapper, parameters, model) # Secretly changes config in place.
        opt_config_params, mapper
    end

    x0 = vectorize_variables(model, parameters, mapper; config = opt_config_params, T)

    ### simulation
    function simulate_ad_wrapper(model, x)
        output = simulate_ad(state0_, model, tstep, x, forces; opt_config_params, parameters_ref = parameters, max_timestep_cuts = 1000, info_level=info_level, kwargs...)
    end
    output, simulate_pullback = Flux.pullback(simulate_ad_wrapper, model, x0)

    states, report = output

    function S_pullback(doutput)
        dmodel, dx0 = simulate_pullback(doutput)
        if isa(dx0, AbstractVector)
            dparams = devectorize_variables(model, dx0, mapper; config = opt_config_params)
        else
            dparams = dx0
        end
        dTransmissibilities = dparams[:Reservoir][:Transmissibilities]
        dLogTransmissibilities = Transmissibilities .* dTransmissibilities
        # dLogTransmissibilities = TLogTransmissibilities_pullback()[1]
        dϕ = dmodel.data_domain.data[:porosity][1]
        df = @not_implemented("I don't know how to do this.")
        return NoTangent(), dLogTransmissibilities, dϕ, df
    end
    return output, S_pullback
end


function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::jutulSource{D, N};
    state0=nothing, visCO2::T=T(visCO2 * 1e1), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, return_extra=false, kwargs...) where {D, T, N}

    Transmissibilities, LogTransmissibilities_pullback = Flux.pullback((x) -> exp.(x), LogTransmissibilities)
    FluidVolume, ϕ_pullback = Flux.pullback((x) -> prod(S.model.d) .* x, ϕ)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = ignore_derivatives() do
        setup_simple_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
    end
    model.data_domain[:porosity] = ϕ
    parameters[:Transmissibilities] = Transmissibilities
    parameters[:FluidVolume] = FluidVolume
    isnothing(state0) || (state0_ = state0)

    # Set up config for gradient.
    opt_config_params, mapper = @ignore_derivatives begin
        active = [:Transmissibilities, :FluidVolume]
        opt_config_params = optimization_config(model, parameters, active)
        opt_config_params[:FluidVolume][:active] = true
        opt_config_params[:Transmissibilities][:active] = true

        targets = Jutul.optimization_targets(opt_config_params, model)
        mapper, = Jutul.variable_mapper(model, :parameters, targets = targets, config = opt_config_params)
        lims = Jutul.optimization_limits(opt_config_params, mapper, parameters, model) # Secretly changes config in place.
        opt_config_params, mapper
    end

    x0 = vectorize_variables(model, parameters, mapper; config = opt_config_params, T)

    ### simulation
    output, case, sim = simulate_ad(state0_, model, tstep, x0, forces; opt_config_params, parameters_ref = parameters, max_timestep_cuts = 1000, info_level=info_level, return_extra=true, kwargs...)
    if return_extra
        return output, case, sim, x0
    end
    return output
end


function ChainRulesCore.rrule(
    S::jutulModeling{D, T},
    LogTransmissibilities::AbstractVector{T},
    ϕ::AbstractVector{T},
    f::jutulSource{D,N};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, kwargs...
) where {D, T, N}
    Transmissibilities, LogTransmissibilities_pullback = Flux.pullback((x) -> exp.(x), LogTransmissibilities)
    FluidVolume, ϕ_pullback = Flux.pullback((x) -> prod(S.model.d) .* x, ϕ)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = ignore_derivatives() do
        setup_simple_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
    end
    model.data_domain[:porosity] = ϕ
    parameters[:Transmissibilities] = Transmissibilities
    parameters[:FluidVolume] = FluidVolume
    isnothing(state0) || (state0_ = state0)

    # Set up config for gradient.
    opt_config_params, mapper = @ignore_derivatives begin
        active = [:Transmissibilities, :FluidVolume]
        opt_config_params = optimization_config(model, parameters, active)
        opt_config_params[:FluidVolume][:active] = true
        opt_config_params[:Transmissibilities][:active] = true

        targets = Jutul.optimization_targets(opt_config_params, model)
        mapper, = Jutul.variable_mapper(model, :parameters, targets = targets, config = opt_config_params)
        lims = Jutul.optimization_limits(opt_config_params, mapper, parameters, model) # Secretly changes config in place.
        opt_config_params, mapper
    end

    x0 = vectorize_variables(model, parameters, mapper; config = opt_config_params, T)

    ### simulation
    function simulate_ad_wrapper(model, x)
        output = simulate_ad(state0_, model, tstep, x, forces; opt_config_params, parameters_ref = parameters, max_timestep_cuts = 1000, info_level=info_level, kwargs...)
    end
    output, simulate_pullback = Flux.pullback(simulate_ad_wrapper, model, x0)

    function S_pullback(doutput)
        dmodel, dx0 = simulate_pullback(doutput)
        if isa(dx0, AbstractVector)
            dparams = devectorize_variables(model, dx0, mapper; config = opt_config_params)
        else
            dparams = dx0
        end
        dTransmissibilities = dparams[:Transmissibilities]
        dLogTransmissibilities = Transmissibilities .* dTransmissibilities
        # dLogTransmissibilities = LogTransmissibilities_pullback(dTransmissibilities)[1]
        dϕ = dmodel.data_domain.data[:porosity][1]
        df = @not_implemented("I don't know how to do this.")
        return NoTangent(), dLogTransmissibilities, dϕ, df
    end
    return output, S_pullback
end

function (S::jutulModeling{D, T})(f::Union{jutulForce{D, N}, jutulVWell{D, N}, jutulSource{D, N}};
    LogTransmissibilities::AbstractVector{T}=KtoTrans(CartesianMesh(S.model), S.model.K), ϕ::AbstractVector{T}=S.model.ϕ,
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, kwargs...) where {D, T, N}

    return S(LogTransmissibilities, ϕ, f; state0=state0, visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O, info_level=info_level, kwargs...)
end

function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}, jutulSource{D, N}};
    ϕ::AbstractVector{T}=S.model.ϕ,
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1, kwargs...) where {D, T, N}

    return S(LogTransmissibilities, ϕ, f; state0=state0, visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O, info_level=info_level, kwargs...)
end
