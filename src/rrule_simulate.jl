using ChainRulesCore: rrule, unthunk, NoTangent, @not_implemented
using Jutul: simulate, setup_parameter_optimization, optimization_config, vectorize_variables, SimulationModel, MultiModel, submodels, get_primary_variables
using JutulDarcy

function simulate_ad(state0, model, tstep, parameters, forces; opt_config=nothing, kwargs...)
    # TODO: should help user by erroring if kwargs also specifies parameters and forces.
    return simulate(state0, model, tstep; kwargs..., parameters, forces);
end

get_state_keys(model::SimulationModel, state::Dict{Any, Any}) = keys(state)
function get_state_keys(model::MultiModel, state::Dict{Any, Any})
    state_keys = Dict{Any, Any}()
    for (k, m) in pairs(submodels(model))
        state_keys[k] = keys(state[k])
    end
    return state_keys
end

get_eltype(model::SimulationModel, state) = eltype(state.Saturations)
get_eltype(model::MultiModel, state) = get_eltype(model[:Reservoir], state.Reservoir)

get_eltype(model::SimulationModel, state::Dict{Any, Any}) = eltype(state[:Saturations])
get_eltype(model::MultiModel, state::Dict{Any, Any}) = get_eltype(model[:Reservoir], state[:Reservoir])

function ChainRulesCore.rrule(::typeof(simulate_ad), state0, model, tstep, parameters, forces; opt_config=nothing, kwargs...)
    output = simulate_ad(state0, model, tstep, parameters, forces; kwargs...);
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

        @show typeof(doutput)
        dstates = unthunk(unthunk(doutput).states)
        @show typeof(dstates)
        @show typeof(dstates[1])
        @show Jutul.variable_mapper(model, :primary)

        # 1. First, we define F, which needs to subtract two Jutul states.
        #   This is easier to do if the states are vectorized, so we'll set
        #   up a vectorizer first. It should be restricted to the targets
        #   that are nonzero in dstates.

        targets = get_state_keys(model, first(dstates))
        @show targets
        mapper = first(Jutul.variable_mapper(model, :primary; targets))
        function F(model, state_ad, dt, step_no, forces)
            # Vectorize everything (with the right type).
            # @show get_eltype(model, dstates[step_no])
            # T = get_eltype(model, dstates[step_no])
            T = Real
            d_state_vec = vectorize_variables(model, dstates[step_no], mapper; T)

            # @show get_eltype(model, states[step_no])
            # T = get_eltype(model, states[step_no])
            T = Real
            state_vec = vectorize_variables(model, states[step_no], mapper; T)

            # @show get_eltype(model, state_ad)
            # T = get_eltype(model, state_ad)
            T = Real
            state_ad_vec = vectorize_variables(model, state_ad, mapper; T)

            c = d_state_vec - state_vec
            return sum((state_ad_vec + c) .^ 2) / 2
        end

        # 2. We need an optimization config. This defines the parameters that we need
        #    the gradient of.
        if isnothing(opt_config)
            opt_config = optimization_config(model, parameters, use_scaling = true, rel_min = 0.1, rel_max = 10)
            for (ki, vi) in opt_config
                if ki in [:TwoPointGravityDifference, :PhaseViscosities]
                    vi[:active] = false
                end
                if ki == :Transmissibilities
                    vi[:scaler] = :log
                end
            end
        end
        opt = setup_parameter_optimization(model, state0, parameters, tstep, forces, F, opt_config, print = 100, param_obj = true, use_sparsity=true);

        # We put the states here so that it doesn't need to recompute the forward problem.
        opt.data[:states] = output.states
        opt.data[:reports] = output.reports

        # 3. We have Jutul compute the gradient.
        dparameters = opt.dF!(similar(opt.x0), opt.x0)

        # dparameters_t = deepcopy(parameters)
        dparameters_t = Dict{Symbol, Any}(
            k => zeros(v.n)
            for (k, v) in pairs(opt.data[:mapper])
        )
        devectorize_variables!(dparameters_t, model, dparameters, opt.data[:mapper], config = opt.data[:config])

        dsimulate = NoTangent()
        dstate0 = NoTangent()
        dmodel = @not_implemented("This is too difficult.")
        dtstep = NoTangent()
        dforces = @not_implemented("I don't know how to do this.")

        return dsimulate, dstate0, dmodel, dtstep, dparameters, dforces
    end
    return output, simulate_ad_pullback
end
