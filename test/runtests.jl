using Test
using Jutul
using JutulDarcyRules
using Flux
using Printf
using Random
using LinearAlgebra

Random.seed!(2023)

@testset "JutulDarcyRules tests" begin
    include("test_utils.jl")

    include("test_model_parameter.jl")

    include("test_gradient.jl")

    include("test_conversion.jl")

    include("test_jutulState.jl")
    include("test_jutulForce.jl")
    include("test_jutulSource.jl")
    include("test_jutulModel.jl")
    include("test_jutulModeling.jl")

    @testset "Compare to JutulDarcy example" begin
        include("test_simulate_rrule.jl")
    end
end