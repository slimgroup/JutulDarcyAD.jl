function force(M::jutulModel{D, T}, w::jutulForce{D, T}, tstep::Vector{T};
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), g::T=T(10.0)) where {D, T}

    ## set up well information
    cell_loc = [Int.(round.(w.loc[i] ./ M.d[1:length(w.loc[1])])) for i = 1:length(w.loc)]
    Is = [setup_well(CartesianMesh(M), M.K, [cell_loc[i]], name = w.name[i]) for i = 1:length(w.loc)]
    ctrls = [w.name[i]==:Injector ? InjectorControl(TotalRateTarget(w.irate), [1.0, 0.0], density = ρCO2) : ProducerControl(BottomHolePressureTarget(50*bar)) for i = 1:length(w.loc)]
    controls = Dict()
    for i = 1:length(w.loc)
        controls[w.name[i]] = ctrls[i]
    end
    return Is, controls
end

function force(M::jutulModel{D, T}, w::jutulVWell{D, T}, tstep::Vector{T};
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), g::T=T(10.0)) where {D, T}

    ## set up well information
    cell_loc = [Int.(round.(w.loc[i] ./ M.d[1:length(w.loc[1])])) for i = 1:length(w.loc)]
    heel = [isnothing(w.startz) ? 1 : Int(div(w.startz[i], M.d[end])) for i = 1:length(w.loc)]
    toe = [isnothing(w.endz) ? M.n[end] : Int(div(w.endz[i], M.d[end])) for i = 1:length(w.loc)]
    Is = [setup_vertical_well(CartesianMesh(M), M.K, cell_loc[i]..., name = w.name[i]; heel = heel[i], toe = toe[i]) for i = 1:length(w.loc)]
    ctrls = [w.name[i]==:Injector ? InjectorControl(TotalRateTarget(w.irate), [1.0, 0.0], density = ρCO2) : ProducerControl(BottomHolePressureTarget(50*bar)) for i = 1:length(w.loc)]
    controls = Dict()
    for i = 1:length(w.loc)
        controls[w.name[i]] = ctrls[i]
    end
    return Is, controls
end

function setup_well_model(M::jutulModel{D, T}, f::Union{jutulForce{D, T}, jutulVWell{D, T}}, tstep::Vector{T}, state0=nothing;
    visCO2::T=T(visCO2), visH2O::T=T(visH2O), ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), g::T=T(10.0)) where {D, T}

    ### set up well controls
    Is, controls = force(M, f, tstep; ρCO2=ρCO2, ρH2O=ρH2O, g=g)    

    ### set up model, parameters
    sys = ImmiscibleSystem((VaporPhase(), AqueousPhase()), reference_densities = [ρCO2, ρH2O])
    mesh = CartesianMesh(M)
    domain_spec = reservoir_domain(mesh, porosity = M.ϕ, permeability = M.K)
    PhaseViscosities = [visCO2, visH2O]
    model_parameters = Dict(:Reservoir => Dict(:PhaseViscosities=> PhaseViscosities))
    model, parameters = setup_reservoir_model(domain_spec, sys, wells = Is, parameters=model_parameters)
    select_output_variables!(model.models.Reservoir, :all)
    ρ = ConstantCompressibilityDensities(p_ref = 150*bar, density_ref = [ρCO2, ρH2O], compressibility = [1e-4/bar, 1e-6/bar])
    for (k, m) in pairs(model.models)
        if k == :Reservoir || JutulDarcy.model_or_domain_is_well(m)
            set_secondary_variables!(m; PhaseMassDensities=ρ)
        end
    end
    replace_variables!(model, PhaseMassDensities = ρ)
    replace_variables!(model, RelativePermeabilities = BrooksCoreyRelativePermeabilities(sys, [2.0, 2.0], [0.1, 0.1], 1.0))
    for x ∈ keys(model.models)
        Jutul.select_output_variables!(model.models[x], :all)
    end

    ### initial state
    if !isnothing(state0) && haskey(state0, :Reservoir) && haskey(state0[:Reservoir], :Pressure)
        p0 = state0[:Reservoir][:Pressure]
    else
        Z = repeat((1:M.n[end])*M.d[end], inner = prod(M.n[1:2]))
        p0 = ρH2O * g * (Z .+ M.h) .+ JutulDarcy.DEFAULT_MINIMUM_PRESSURE
    end
    if !isnothing(state0) && haskey(state0, :Reservoir) && haskey(state0[:Reservoir], :Saturations)
        S0 = state0[:Reservoir][:Saturations]
    else
        S0 = [0.0, 1.0]
    end
    state0 = setup_reservoir_state(model, Pressure = p0, Saturations = S0)

    ### forces
    bc = if M.pad
        boundary = Int[]
        for cell in 1:number_of_cells(mesh)
            I, J, K = cell_ijk(mesh, cell)
            if I == 1 || I == M.n[1]
                push!(boundary, cell)
            end
        end
        flow_boundary_condition(boundary, domain_spec, p0[boundary]; fractional_flow=[0.0, 1.0])
    else
        nothing
    end

    forces = setup_reservoir_forces(model, control = controls; bc)
    return model, parameters, state0, forces
end

function source(M::jutulModel{D, T}, model, f::jutulSource{D, T}, p0; ρCO2::T=T(ρCO2)) where {D, T}
    cell_loc = [Int.(round.(f.loc[i] ./ M.d)) for i = 1:length(f.loc)]
    cell = [sum([(cell_loc[i][d]-1) * prod(M.n[1:d-1]) for d = length(cell_loc[i]):-1:1]) + 1 for i = 1:length(cell_loc)]
    src  = [SourceTerm(cell[i], f.irate[i]; type = JutulDarcy.VolumeSource, fractional_flow = [T(f.irate[i] > 0), T(1)-T(f.irate[i] > 0)]) for i = 1:length(f.loc)]

    bc = if M.pad
        mesh = CartesianMesh(M)
        boundary = Int[]
        for cell in 1:number_of_cells(mesh)
            I, J, K = cell_ijk(mesh, cell)
            if I == 1 || I == M.n[1]
                push!(boundary, cell)
            end
        end
        flow_boundary_condition(boundary, model.data_domain, p0[boundary]; fractional_flow=[0.0, 1.0])
    else
        nothing
    end
    return setup_forces(model, sources = src; bc)
end

function simple_model(M::jutulModel{D, T}; ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O)) where {D, T}
    sys = ImmiscibleSystem((VaporPhase(), AqueousPhase()), reference_densities = [ρCO2, ρH2O])
    g = CartesianMesh(M.n, M.d .* M.n)
    domain_spec = reservoir_domain(g, porosity = M.ϕ, permeability = M.K)
    model = SimulationModel(domain_spec, sys, output_level = :all)
    model.primary_variables[:Pressure] = JutulDarcy.Pressure(minimum = -Inf, max_rel = nothing)
    ρ = ConstantCompressibilityDensities(p_ref = 150*bar, density_ref = [ρCO2, ρH2O], compressibility = [1e-4/bar, 1e-6/bar])
    set_secondary_variables!(model; PhaseMassDensities=ρ)
    replace_variables!(model, PhaseMassDensities = ρ)
    replace_variables!(model, RelativePermeabilities = BrooksCoreyRelativePermeabilities(sys, [2.0, 2.0], [0.1, 0.1], 1.0))
    return model
end

function setup_simple_model(M::jutulModel{D, T}, f::jutulSource{D, T}, tstep::Vector{T}, state0=nothing;
    visCO2::T=T(visCO2), visH2O::T=T(visH2O), ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), g::T=T(10.0)) where {D, T}
    model = simple_model(M; ρCO2, ρH2O)

    parameters = setup_parameters(model, PhaseViscosities = [visCO2, visH2O]);

    if isnothing(state0)
        Z = repeat((1:M.n[end])*M.d[end], inner = prod(M.n[1:2]))
        p0 = ρH2O * g * (Z .+ M.h) .+ JutulDarcy.DEFAULT_MINIMUM_PRESSURE
        state0 = setup_state(model, Pressure = p0, Saturations = [0.0, 1.0])
    else
        p0 = state0[:Pressure]
    end
    forces = source(M, model, f, p0; ρCO2)
    return model, parameters, state0, forces
end
