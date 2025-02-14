# function rrule(S::jutulModeling{D, T}, LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}};
#     state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
#     ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}
   
#     Transmissibilities = exp.(LogTransmissibilities)

#     ### set up simulation time
#     tstep = day * S.tstep

#     ### set up simulation configurations
#     model, parameters, state0_, forces = setup_well_model(S.model, f, tstep; visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O)

#     model.models.Reservoir.data_domain[:porosity] = ϕ
#     parameters[:Reservoir][:Transmissibilities] = Transmissibilities
#     parameters[:Reservoir][:FluidVolume] .= prod(S.model.d) .* ϕ

#     isnothing(state0) || (state0_[:Reservoir] = get_Reservoir_state(state0))

#     ### simulation
#     sim, config = setup_reservoir_simulator(model, state0_, parameters);
#     states, reports = simulate!(sim, tstep, forces = forces, config = config, max_timestep_cuts = 1000, info_level=info_level);
#     output = jutulStates(states)
    
#     ### optimization framework
#     cfg = optimization_config(model, parameters, Dict(:Reservoir => [:FluidVolume, :Transmissibilities], :Injector => [:FluidVolume]))
#     cfg[:Reservoir][:Transmissibilities][:scaler] = :log

#     function pullback(dy)
#         states_ref_ = output(vec(output)-dy)
#         check_valid_state(states_ref_)
#         states_ref = dict(states_ref_)
#         mass_mismatch = (m, state, dt, step_no, forces) -> loss_per_step(m, state, dt, step_no, forces, states_ref)
#         F_o, dF_o, F_and_dF, x0, lims, data = setup_parameter_optimization(model, state0_, parameters, tstep, forces, mass_mismatch, cfg, param_obj = true, print = info_level, config = config, use_sparsity = false);
#         data[:states] = states
#         data[:reports] = reports
#         g = dF_o(similar(x0), x0);
#         n_faces = length(LogTransmissibilities)
#         n_cells = prod(S.model.n)
#         dLogTransmissibilities = g[1:n_faces]
#         dϕ = g[n_faces + 1 : n_faces + n_cells] * prod(S.model.d)
#         return NoTangent(), dLogTransmissibilities, dϕ, NoTangent()
#     end
#     return output, pullback
# end

# function rrule(S::jutulModeling{D, T}, LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::jutulSource{D, N};
#     state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
#     ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}
    
#     Transmissibilities = exp.(LogTransmissibilities)

#     ### set up simulation time
#     tstep = day * S.tstep

#     ### set up simulation configurations
#     model, parameters, state0_, forces = setup_simple_model(S.model, f, tstep, state0; visCO2, visH2O, ρCO2, ρH2O)
#     model.data_domain[:porosity] = ϕ

#     isnothing(state0) || (state0_ = state0)

#     states, reports = simulate(dict(state0_), model, tstep, parameters = parameters, forces = forces, info_level = info_level, max_timestep_cuts = 1000)
#     output = jutulSimpleStates(states)
#     cfg = optimization_config(model, parameters, use_scaling = false, rel_min = 0., rel_max = Inf)
#     for (ki, vi) in cfg
#         if ki in [:TwoPointGravityDifference, :PhaseViscosities]
#             vi[:active] = false
#         end
#         if ki == :Transmissibilities
#             vi[:scaler] = :log
#         end
#     end

#     function pullback(dy)
#         states_dy = output(dy)
#         states_ref = dict(output-states_dy)
#         function mass_mismatch(m, state, dt, step_no, forces)
#             state_ref = states_ref[step_no]
#             fld = :Saturations
#             fld2 = :Pressure
#             val = state[fld]
#             val2 = state[fld2]
#             ref = state_ref[fld]
#             ref2 = state_ref[fld2]
#             return 0.5 * sum((val[1,:] - ref[1,:]).^2) + 0.5 * sum((val2-ref2).^2)
#         end
#         mass_mismatch = (m, state, dt, step_no, forces) -> loss_per_step_simple(m, state, dt, step_no, forces, states_ref)
#         Jutul.evaluate_objective(mass_mismatch, model, states_ref, tstep, forces)
#         F_o, dF_o, F_and_dF, x0, lims, data = setup_parameter_optimization(model,
#         dict(state0_), parameters, tstep, forces, mass_mismatch, cfg, print = -1, param_obj = true);
#         data[:states] = states
#         data[:reports] = reports
#         g = dF_o(similar(x0), x0);
#         n_faces = length(LogTransmissibilities)
#         n_cells = prod(S.model.n)
#         dLogTransmissibilities = g[1:n_faces]
#         dϕ = g[n_faces + 1 : n_faces + n_cells] * prod(S.model.d)
#         return NoTangent(), dLogTransmissibilities, dϕ, NoTangent()
#     end
#     return output, pullback
# end

# function loss_per_step(m, state, dt, step_no, forces, states_ref)
#     state_ref = states_ref[step_no]
#     fld = :Saturations
#     fld2 = :Pressure
#     val = state[:Reservoir][fld]
#     val2 = state[:Reservoir][fld2]
#     ref = state_ref[:Reservoir][fld]
#     ref2 = state_ref[:Reservoir][fld2]
#     return inner_mismatch(val, ref, val2, ref2)
# end

# function loss_per_step_simple(m, state, dt, step_no, forces, states_ref)
#     state_ref = states_ref[step_no]
#     fld = :Saturations
#     fld2 = :Pressure
#     val = state[fld]
#     val2 = state[fld2]
#     ref = state_ref[fld]
#     ref2 = state_ref[fld2]
#     return inner_mismatch(val, ref, val2, ref2)
# end

# function inner_mismatch(val, ref, val2, ref2)
#     mismatch_s = zero(eltype(val))
#     for i in axes(val, 2)
#         mismatch_s += (val[1,i] - ref[1,i])^2
#     end
#     mismatch_p = zero(eltype(val2))
#     for i in eachindex(val2)
#         mismatch_p += (val2[i] - ref2[i])^2
#     end
#     return eltype(val)(0.5) * mismatch_s + eltype(val2)(0.5) * mismatch_p
# end
