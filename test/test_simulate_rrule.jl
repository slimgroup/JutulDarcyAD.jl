# # Example demonstrating optimzation of parameters against observations
# We create a simple test problem: A 1D nonlinear displacement. The observations
# are generated by solving the same problem with the true parameters. We then
# match the parameters against the observations using a different starting guess
# for the parameters, but otherwise the same physical description of the system.
using Jutul
using JutulDarcy
using JutulDarcyRules: simulate_ad
using LinearAlgebra
using Flux
using ChainRulesCore
using JutulDarcy: Pressure

function setup_bl(;nc = 100, time = 1.0, nstep = 100, poro = 0.1, perm = 9.8692e-14)
    T = time
    tstep = repeat([T/nstep], nstep)
    G = get_1d_reservoir(nc, poro = poro, perm = perm)
    nc = number_of_cells(G)

    bar = 1e5
    p0 = 1000*bar
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()))
    model = SimulationModel(G, sys)
    model.primary_variables[:Pressure] = Pressure(minimum = -Inf, max_rel = nothing)
    kr = BrooksCoreyRelativePermeabilities(sys, [2.0, 2.0])
    replace_variables!(model, RelativePermeabilities = kr)
    tot_time = sum(tstep)

    parameters = setup_parameters(model, PhaseViscosities = [1e-3, 5e-3]) # 1 and 5 cP
    state0 = setup_state(model, Pressure = p0, Saturations = [0.0, 1.0])

    irate = 100*sum(parameters[:FluidVolume])/tot_time
    src  = [SourceTerm(1, irate, fractional_flow = [1.0-1e-3, 1e-3]), 
            SourceTerm(nc, -irate, fractional_flow = [1.0, 0.0])]
    forces = setup_forces(model, sources = src)

    return (model, state0, parameters, forces, tstep)
end
# Number of cells and time-steps
N = 100
Nt = 100
poro_ref = 0.1
perm_ref = 9.8692e-14
# ## Set up and simulate reference
model_ref, state0_ref, parameters_ref, forces, tstep = setup_bl(nc = N, nstep = Nt, poro = poro_ref, perm = perm_ref)
states_ref, = simulate(state0_ref, model_ref, tstep, parameters = parameters_ref, forces = forces, info_level = -1);

# ## Set up another case where the porosity is different
model, state0, parameters, = setup_bl(nc = N, nstep = Nt, poro = 2*poro_ref, perm = 1.0*perm_ref)
output = simulate(state0, model, tstep, parameters = parameters, forces = forces, info_level = -1);
states, ref = output

# ## Define objective function
# Define objective as mismatch between water saturation in current state and
# reference state. The objective function is currently a sum over all time
# steps. We implement a function for one term of this sum.
function mass_mismatch(m, state, dt, step_no, forces)
    state_ref = states_ref[step_no]
    fld = :Saturations
    val = state[fld]
    ref = state_ref[fld]
    err = 0
    for i in axes(val, 2)
        err += (val[1, i] - ref[1, i])^2
    end

    fld = :Pressure
    val = state[fld]
    ref = state_ref[fld]
    for i in axes(val, 2)
        err += (val[i] - ref[i])^2
    end
    return dt*err
end
function objective(output)
    states, rep = output
    misfit = Jutul.evaluate_objective(mass_mismatch, model, states, tstep, forces)
    return misfit
end
@assert objective((states_ref, nothing)) == 0.0
@assert objective(output) > 0.0

# ## Set up a configuration for the optimization
#
# The optimization code enables all parameters for optimization by default, with
# relative box limits 0.1 and 10 specified here. If use_scaling is enabled the
# variables in the optimization are scaled so that their actual limits are
# approximately box limits.
#
# We are not interested in matching gravity effects or viscosity here.
# Transmissibilities are derived from permeability and varies significantly. We
# can set log scaling to get a better conditioned optimization system, without
# changing the limits or the result.

cfg = optimization_config(model, parameters)
for (ki, vi) in cfg
    if ki in [:TwoPointGravityDifference, :PhaseViscosities]
        vi[:active] = false
    end
    if ki == :Transmissibilities
        vi[:scaler] = :default
    end
end
print_obj = 100
#-
# ## Set up parameter optimization
#
# This gives us a set of function handles together with initial guess and limits.
# Generally calling either of the functions will mutate the data Dict. The options are:
# F_o(x) -> evaluate objective
# dF_o(dFdx, x) -> evaluate gradient of objective, mutating dFdx (may trigger evaluation of F_o)
# F_and_dF(F, dFdx, x) -> evaluate F and/or dF. Value of nothing will mean that the corresponding entry is skipped.
opt_info = setup_parameter_optimization(model, state0, parameters, tstep, forces, mass_mismatch, cfg, print = print_obj, param_obj = true);
F_o, dF_o, F_and_dF, x0, lims, data = opt_info
F_initial = F_o(x0)
dF_initial = dF_o(similar(x0), x0)
@info "Initial objective: $F_initial, gradient norm $(norm(dF_initial))"

output1 = simulate_ad(state0, model, tstep, parameters, forces; opt_config_params=cfg, info_level=-1);
output2 = simulate_ad(state0, model, tstep, x0, forces; parameters_ref=parameters, opt_config_params=cfg, info_level=-1);
@show norm(output1.states[1][:Saturations] .- output2.states[1][:Saturations])
@show norm(output1.states[1][:Pressure] .- output2.states[1][:Pressure])
@show norm(output1.states[99][:Saturations] .- output2.states[99][:Saturations])
@show norm(output1.states[99][:Pressure] .- output2.states[99][:Pressure])
@show objective(output1)
@show objective(output2)
# error("hi")

# mapper = opt_info.data[:mapper]
# parameters_t = deepcopy(parameters)
# x0 = vectorize_variables(model, parameters, mapper, config = cfg)
# devectorize_variables!(parameters_t, model, x0, mapper, config = cfg)

# parameters_t = deepcopy(parameters)
# targets = Jutul.optimization_targets(cfg, model)
# mapper, = Jutul.variable_mapper(model, :parameters; targets, config = cfg)
# # lims = Jutul.optimization_limits(cfg, mapper, parameters_t, model) # Secretly changes config in place.
# devectorize_variables!(parameters_t, model, x0, mapper, config = cfg)


# model, state0, parameters, = setup_bl(nc = N, nstep = Nt, poro = 2*poro_ref, perm = 1.0*perm_ref)

# Now we'll use Flux's interface for both doutput and dparameters.
@info "Getting two pullbacks explicitly."
function simulate_wrapper(state0, model, tstep, x, forces, opt_config_params)
    simulate_ad(state0, model, tstep, x, forces; parameters_ref=parameters, opt_config_params, info_level = -1)
end
output3, pullback_output = Flux.pullback(simulate_wrapper, state0, model, tstep, x0, forces, cfg)
@show norm(output3.states[1][:Saturations] .- output2.states[1][:Saturations])
@show norm(output3.states[1][:Pressure] .- output2.states[1][:Pressure])
@show norm(output3.states[99][:Saturations] .- output2.states[99][:Saturations])
@show norm(output3.states[99][:Pressure] .- output2.states[99][:Pressure])
@show objective(output3)
@show objective(output2)
misfit_val, pullback_misfit = Flux.pullback(objective, output3)
@info "Initial objective: $misfit_val, gradient norm TBD"
@show misfit_val - F_initial
@assert norm(F_initial - misfit_val) < 1e-10

@info "Running first pullback."
dmisfit = pullback_misfit(1.0)
doutput = dmisfit[1]
@info "Running second pullback."
dstate0, dmodel, dtstep, dparameters, dforces = pullback_output(doutput)

@info "Initial objective: $misfit_val, gradient norm $(norm(dparameters))"

@test norm(dF_initial) ≈ norm(dparameters)

# Now we'll use Flux's interface to directly get dparameters.
function full_objective(x; opt_config_params)
    output = simulate_ad(state0, model, tstep, x, forces; parameters_ref=parameters, opt_config_params, info_level=-1);
    return objective(output)
end

@info "Getting gradient directly."

misfit_val, dparameters = Flux.withgradient(x -> full_objective(x; opt_config_params=cfg), x0)

@info "Initial objective: $misfit_val, gradient norm $(norm(dparameters))"

# Gradient test.
dx = x0 .* (1 .- exp.(1e-1 * randn(size(x0))))

@info "Running gradient test on Jutul's gradient"
grad_test(F_o, x0, dx, dF_initial)

@info "Running gradient test on my gradient"
grad_test(F_o, x0, dx, dparameters[1])

# Now I'll include pressure in the objective.
function mass_mismatch(m, state, dt, step_no, forces)
    state_ref = states_ref[step_no]
    fld = :Saturations
    val = state[fld]
    ref = state_ref[fld]
    err = 0
    for i in axes(val, 2)
        err += (val[1, i] - ref[1, i])^2
    end

    fld = :Pressure
    val = state[fld]
    ref = state_ref[fld]
    for i in axes(val, 2)
        err += (val[i] - ref[i])^2
    end
    return dt*err
end

misfit_val, dparameters = Flux.withgradient(x -> full_objective(x; opt_config_params=cfg), x0)
@info "Initial objective: $misfit_val, gradient norm $(norm(dparameters))"
@info "Running gradient test on my gradient (with pressure)"
grad_test(F_o, x0, dx, dparameters[1])

# Now I'll do it with different parameters.
config = optimization_config(model, parameters)
for (ki, vi) in config
    if ki in [:PhaseViscosities]
        vi[:active] = false
    end
    if ki == :Transmissibilities
        vi[:scaler] = :default
    end
end
targets = Jutul.optimization_targets(config, model)
mapper, = Jutul.variable_mapper(model, :parameters, targets = targets, config = config)
x0 = vectorize_variables(model, parameters, mapper, config = config)
dx = x0 .* (1 .- exp.(1e-1 * randn(size(x0))))

misfit_val, dparameters = Flux.withgradient(x -> full_objective(x; opt_config_params=config), x0)

@info "Running gradient test on gradient of different parameters (with pressure)"
grad_test(x -> full_objective(x; opt_config_params=config), x0, dx, dparameters[1])

function full_objective2(tstep; opt_config_params = nothing)
    output = simulate_ad(state0, model, tstep, parameters, forces; opt_config_params);
    return objective(output)
end
misfit_val, dmodel = Flux.withgradient(x -> full_objective2(x; opt_config_params=config), tstep)

# Now I won't include pressure in the objective.
function mass_mismatch(m, state, dt, step_no, forces)
    state_ref = states_ref[step_no]
    fld = :Saturations
    val = state[fld]
    ref = state_ref[fld]
    err = 0
    for i in axes(val, 2)
        err += (val[1, i] - ref[1, i])^2
    end
    return dt*err
end

misfit_val, dparameters = Flux.withgradient(x -> full_objective(x; opt_config_params=config), x0)

@info "Running gradient test on gradient of different parameters (without pressure)"
grad_test(x -> full_objective(x; opt_config_params=config), x0, dx, dparameters[1])
