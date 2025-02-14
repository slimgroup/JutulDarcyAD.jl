using ChainRulesCore: rrule, unthunk, NoTangent, @not_implemented
using Jutul: Jutul, simulate, setup_parameter_optimization, optimization_config, vectorize_variables, SimulationModel, MultiModel, submodels, get_primary_variables
using JutulDarcy

function simulate_ad(state0, model, tstep, parameters, forces;
    opt_config_params=nothing, parameters_ref=nothing,
    opt_config_state0=nothing, state0_ref=nothing,
    return_extra=false,
    kwargs...)
    # We'll have to devectorize everything that we want the gradient. The `opt_config`
    # defines which variables are in the vector and how to vectorize/devectorize them.

    # Devectorize state0.
    if isa(state0, AbstractVector)
        if isnothing(opt_config_state0)
            error("Expected opt_config_state0 to define the vectorization of state0.")
        end
        if isnothing(state0_ref)
            error("Expected state0_ref to define the default state0 values.")
        end
        state0_t =  deepcopy(state0_ref)
        targets = Jutul.optimization_targets(opt_config_state0, model)
        mapper, = Jutul.variable_mapper(model, :primary; targets, config = opt_config_state0)
        # lims = Jutul.optimization_limits(opt_config_state0, mapper, state0_t, model) # Secretly changes config in place.
        devectorize_variables!(state0_t, model, state0, mapper, config = opt_config_state0)
    else
        state0_t = state0
    end

    # Devectorize parameters.
    if isa(parameters, AbstractVector)
        if isnothing(opt_config_params)
            error("Expected opt_config_params to define the vectorization of params.")
        end
        if isnothing(parameters_ref)
            error("Expected parameters_ref to define the default parameters values.")
        end
        parameters_t = deepcopy(parameters_ref)
        # @show opt_config_params
        targets = Jutul.optimization_targets(opt_config_params, model)
        mapper, = Jutul.variable_mapper(model, :parameters; targets, config = opt_config_params)
        devectorize_variables!(parameters_t, model, parameters, mapper, config = opt_config_params)
    else
        parameters_t = parameters
    end

    # TODO: should help user by erroring if kwargs also specifies parameters and forces.
    case = JutulCase(model, tstep, forces; parameters = parameters_t, state0 = state0_t)
    # output = simulate(case; kwargs...);
    sim = Simulator(case)
    output = simulate!(sim, case.dt; forces = case.forces, kwargs...)
    if return_extra
        return output, case, sim
    end
    return output
end

get_state_keys(model::SimulationModel, state::Dict{Any, Any}) = keys(state)
function get_state_keys(model::MultiModel, state::Dict{Any, Any})
    state_keys = Dict{Any, Any}()
    for (k, m) in pairs(submodels(model))
        if haskey(state, k)
            state_keys[k] = keys(state[k])
        else
            state_keys[k] = ()
        end
    end
    return state_keys
end

set_state_keys!(model::SimulationModel, state::Dict{Any, Any}) = nothing
function set_state_keys!(model::MultiModel, state::Dict{Any, Any})
    for (k, m) in pairs(submodels(model))
        if !haskey(state, k)
            state[k] = nothing
        end
    end
end

get_eltype(model::SimulationModel, state) = eltype(state.Saturations)
get_eltype(model::MultiModel, state) = get_eltype(model[:Reservoir], state.Reservoir)

get_eltype(model::SimulationModel, state::Dict{Any, Any}) = eltype(state[:Saturations])
get_eltype(model::MultiModel, state::Dict{Any, Any}) = get_eltype(model[:Reservoir], state[:Reservoir])

function ChainRulesCore.rrule(::typeof(simulate_ad), state0, model, tstep, parameters, forces;
    opt_config_params, parameters_ref=nothing,
    kwargs...)
    output, case, sim = simulate_ad(state0, model, tstep, parameters, forces; opt_config_params, parameters_ref, kwargs..., return_extra=true);
    state0_t = case.state0
    parameters_t = case.parameters
    states, ref = output
    function simulate_ad_pullback(doutput)
        # For reverse-AD on a scalar L, we take an input dstates = dL/dy and
        #  apply the adjoint Jacobian dy/dxᵀ to get the parameter gradient dL/dx.
        # Jutul provides a way to get the gradient of an arbitrary scalar F, so we
        #  need to choose F such that its gradient is the proper Jacobian action.
        #   - Let F = ||M(x) + c||^2/2.
        #   - Then dF/dx = Jᵀ(M(x) + c).
        #   - Choose c = dy - M(x).
        #   - Then dF/dx = Jᵀdy as desired.

        dstates = unthunk(unthunk(doutput).states)

        # 1. First, we define F, which needs to subtract two Jutul states.
        #   This is easier to do if the states are vectorized, so we'll set
        #   up a vectorizer first. It should be restricted to the targets
        #   that are nonzero in dstates.

        targets_states = get_state_keys(model, first(dstates))
        mapper_states = first(Jutul.variable_mapper(model, :primary; targets=targets_states))
        for s in dstates
            set_state_keys!(model, s)
        end

        function F(model::SimulationModel, state_ad, dt, step_no, forces)
            obj = 0.0
            for (k, v) in mapper_states
                state_ad_k = state_ad[k]
                dstate_k = dstates[step_no][k]
                state_k = states[step_no][k]
                c = dstate_k .- state_k
                obj += sum((state_ad_k .+ c) .^ 2)
            end
            return obj / 2
        end
        function F(model::MultiModel, state_ad, dt, step_no, forces)
            obj = 0.0
            for (model, targets) in mapper_states
                state_ad_model = state_ad[model]
                dstate_model = dstates[step_no][model]
                state_model = states[step_no][model]
                for (k, v) in targets
                    state_ad_k = state_ad_model[k]
                    dstate_k = dstate_model[k]
                    state_k = state_model[k]
                    c = dstate_k .- state_k
                    obj += sum((state_ad_k .+ c) .^ 2)
                end
            end
            return obj / 2
        end
        sens = JutulDarcy.reservoir_sensitivities(case, output, F;
            include_parameters = true,
            include_state0 = false,
            adjoint_arg=(; use_sparsity=false),
        )

        dparameters = Dict{Symbol, Any}()
        if isa(model, MultiModel)
            for (k, m) in pairs(submodels(model))
                @info "Extracting gradient from data domain for model $k"
                dparameters[k] = Dict{Symbol, Any}()
                if k == :Reservoir
                    for pk in keys(parameters_t[k])
                        dparameters[k][pk] = sens.data[pk][1]
                    end
                else
                    for pk in keys(parameters_t[k])
                        dparameters[k][pk] = @not_implemented("I don't know how to do this.")
                    end
                end
            end
        else
            @info "Extracting gradient from data domain"
            for pk in keys(parameters_t)
                dparameters[pk] = sens.data[pk][1]
            end
        end

        if isa(parameters, AbstractVector)
            # Convert dparameters to a vector.
            targets_params = Jutul.optimization_targets(opt_config_params, model)
            mapper_params, = Jutul.variable_mapper(model, :parameters; targets=targets_params, config = opt_config_params)
            dparameters = vectorize_variables(model, dparameters, mapper_params)
        end

        dsimulate = NoTangent()
        dstate0 = @not_implemented("I don't know how to do this.")
        dmodel = (most_of_it = @not_implemented("This is too difficult."), data_domain=sens)
        dtstep = NoTangent()
        dforces = @not_implemented("I don't know how to do this.")

        @info "Returning gradients"
        return dsimulate, dstate0, dmodel, dtstep, dparameters, dforces
    end
    return output, simulate_ad_pullback
end

export devectorize_variables

function devectorize_variables(model, V, type_or_map = :primary; config = nothing, T = Float64)
    mapper = Jutul.get_mapper_internal(model, type_or_map)
    state = Dict{Symbol, Any}()
    for (k, v) in mapper
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        F = Jutul.opt_scaler_function(config, k, inv = true)
        (; n, offset) = v
        state[k] = zeros(T, n)
        Jutul.devectorize_variable!(state, V, k, v, F, config = c)
    end
    return state
end


function devectorize_variables(model::MultiModel, V, type_or_map = :primary; config = nothing)
    mapper = Jutul.get_mapper_internal(model, type_or_map)
    state = Dict{Symbol, Any}()
    for (k, submodel) in pairs(model.models)
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        state[k] = devectorize_variables(submodel, V, mapper[k], config = c)
    end
    return state
end

function ChainRulesCore.rrule(::typeof(vectorize_variables), model, state_or_prm, type_or_map = :primary;
    config = nothing, T = Float64)
    v = vectorize_variables(model, state_or_prm, type_or_map; config, T)
    function vectorize_pullback(dv)
        dstate_or_prm = devectorize_variables(model, dv, type_or_map; config)
        dmodel = NoTangent()
        dtype_or_map = NoTangent()
        return NoTangent(), dmodel, dstate_or_prm, dtype_or_map
    end
    return v, vectorize_pullback
end

function ChainRulesCore.rrule(::typeof(devectorize_variables), model, V, type_or_map = :primary; config = nothing)
    state_or_prm = devectorize_variables(model, V, type_or_map; config)
    function devectorize_pullback(dstate_or_prm)
        dv = vectorize_variables(model, dstate_or_prm, type_or_map; config, T = eltype(V))
        dmodel = NoTangent()
        dtype_or_map = NoTangent()
        return NoTangent(), dmodel, dv, dtype_or_map
    end
    return state_or_prm, devectorize_pullback
end
