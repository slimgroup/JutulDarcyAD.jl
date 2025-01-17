__precompile__()
module JutulDarcyRules

    export JutulDarcyRulesPATH, Darcy, md

    using LinearAlgebra
    using Statistics
    using JutulDarcy
    using Jutul
    using Optim
    using Flux
    using ChainRulesCore
    import Jutul: JutulGeometry, get_facepos, compute_face_trans, compute_half_face_trans, expand_perm
    import Jutul: SimulationModel, select_output_variables!
    import Jutul: optimization_targets, variable_mapper, optimization_limits, print_parameter_optimization_config
    import Jutul: objective_opt!, gradient_opt!, objective_and_gradient_opt!
    import Base: +, -, *, /, ==
    import Base: display, length, size, getindex, setindex!, IndexStyle, vec, firstindex, lastindex
    import LinearAlgebra: norm, dot
    import ChainRulesCore: rrule

    JutulDarcyRulesPATH = dirname(pathof(JutulDarcyRules))

    visCO2 = 1e-4
    visH2O = 1e-3
    ρCO2 = 7e2
    ρH2O = 1e3
    bar = 1e5

    const Darcy = 9.869232667160130e-13
    const md = Darcy * 1e-3
    
    const day = 24*3600.0

    include("PropertyConversion/PropertyConversion.jl")
    include("FlowRules/FlowRules.jl")
    include("rrule_simulate.jl")

end # module JutulDarcyRules
