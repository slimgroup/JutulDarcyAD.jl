export jutulState, jutulStates, jutulSimpleState, jutulSimpleStates, dict, Saturations, Pressure

abstract type jutulAllState{T} <: DenseVector{T} end
abstract type jutulSimpleOrMultiModelStates{T} <: jutulAllState{T} end
abstract type jutulSimpleOrMultiModelState{T} <: jutulAllState{T} end

struct jutulState{T} <: jutulSimpleOrMultiModelState{T}
    state::Dict
end

struct jutulStates{T} <: jutulSimpleOrMultiModelStates{T}
    states::Vector{jutulState{T}}
end

struct jutulSimpleState{T} <: jutulSimpleOrMultiModelState{T}
    state::Dict
end

struct jutulSimpleStates{T} <: jutulSimpleOrMultiModelStates{T}
    states::Vector{jutulSimpleState{T}}
end

state_T(T) = Dict{Symbol, T}
complex_state_T(T) = Union{Dict{Symbol, T}, AbstractVector{Dict{Symbol, T}}}
display(state::jutulAllState{T}) where T = println("$(typeof(state))")


jutulState(state::Dict) = jutulState{eltype(state[:Reservoir][:Saturations])}(state)
function jutulStates(states::Vector{S}) where {T, S<:complex_state_T(T)}
    error("HI")
    jutulStates{eltype(states[1][:Reservoir][:Saturations])}([jutulState(states[i]::state_T(T)) for i = 1:length(states)])
end
jutulSimpleState(state::state_T(T)) where T = jutulSimpleState{eltype(state[:Saturations])}(state)
jutulSimpleStates(states::Vector{S}) where {T, S<:complex_state_T(T)} = jutulSimpleStates{eltype(states[1][:Saturations])}([jutulSimpleState(states[i]::state_T(T)) for i = 1:length(states)])

using ChainRulesCore: ChainRulesCore
function ChainRulesCore.rrule(::Type{jutulStates}, states::Vector{S}) where {T, S<:complex_state_T(T)}
    @info "I'm here"
    @show typeof(states)
    jstates = jutulStates(states)
    jutulStates_pullback(ΔjutulStates) = NoTangent(), dict(jstates(ΔjutulStates))
    return jstates, jutulStates_pullback
end


function ChainRulesCore.rrule(::Type{jutulSimpleStates}, states::Vector{S}) where {T, S<:complex_state_T(T)}
    @info "I'm here 2"
    @show typeof(states)
    jstates = jutulSimpleStates(states)
    jutulSimpleStates_pullback(ΔjutulSimpleStates) = NoTangent(), dict(jstates(ΔjutulSimpleStates))
    return jstates, jutulSimpleStates_pullback
end

# function ChainRulesCore.rrule(::Type{jutulSimpleStates}, states::Vector{S}) where {T, S<:jutulSimpleState{T}}
#     @info "I'm here 3"
#     @show typeof(states)
#     jstates = jutulSimpleStates(states)
#     jutulSimpleStates_pullback(ΔjutulSimpleStates) = NoTangent(), dict(jstates(ΔjutulSimpleStates))
#     return jstates, jutulSimpleStates_pullback
# end

# function ChainRulesCore.rrule(::Type{jutulStates{T}}, states::Vector{jutulState{T}}) where {T}
#     @info "But I'm here"
#     jutulStates_pullback(ΔjutulStates) = NoTangent(), ΔjutulStates.
#     return jutulStates(states), jutulStates_pullback
# end

Saturations(state::jutulState) = state.state[:Reservoir][:Saturations][1,:]
Pressure(state::jutulState) = state.state[:Reservoir][:Pressure]
Saturations(state::jutulSimpleState) = state.state[:Saturations][1,:]
Pressure(state::jutulSimpleState) = state.state[:Pressure]

Saturations(state::jutulSimpleOrMultiModelStates) = vcat([Saturations(state.states[i]) for i = 1:get_nt(state)]...)
Pressure(state::jutulSimpleOrMultiModelStates) = vcat([Pressure(state.states[i]) for i = 1:get_nt(state)]...)

get_Reservoir_state(state::jutulState) = state.state[:Reservoir]
get_Reservoir_state(state::jutulSimpleState) = state.state
get_Reservoir_state(state::jutulSimpleOrMultiModelStates) = get_Reservoir_state(state.states[end])

get_nt(state::jutulSimpleOrMultiModelStates) = length(state.states)
get_nn(state::jutulSimpleOrMultiModelState) = length(Saturations(state))
get_nn(state::jutulSimpleOrMultiModelStates) = get_nn(state.states[1])

###### turn jutulStates to state dictionary

dict(state::jutulSimpleOrMultiModelState) = state.state
dict(state::jutulSimpleOrMultiModelStates) = [dict(state.states[i]) for i = 1:get_nt(state)]
dict(state::Dict) = state

###### AbstractVector

length(state::jutulSimpleOrMultiModelState) = length(Saturations(state)) + length(Pressure(state))
length(state::jutulSimpleOrMultiModelStates) = sum([length(state.states[i]) for i = 1:get_nt(state)])
size(state::jutulAllState) = (length(state),)

vec(state::jutulAllState) = vcat(Saturations(state), Pressure(state))

IndexStyle(::jutulAllState) = IndexLinear()
function getindex(state::jutulState, i::Int)
    if i <= get_nn(state)
        ## getindex for saturation
        return state.state[:Reservoir][:Saturations][1,i]
    else
        ## getindex for pressure
        return state.state[:Reservoir][:Pressure][i-get_nn(state)]
    end
end

function getindex(state::jutulSimpleState, i::Int)
    if i <= get_nn(state)
        ## getindex for saturation
        return state.state[:Saturations][1,i]
    else
        ## getindex for pressure
        return state.state[:Pressure][i-get_nn(state)]
    end
end

function setindex!(state::jutulState{T}, v, i::Int) where T
    if i <= get_nn(state)
        ## setindex for saturation
        state.state[:Reservoir][:Saturations][1,i] = T(v)
        state.state[:Reservoir][:Saturations][2,i] = T(1) - T(v)
    else
        ## setindex for pressure
        state.state[:Reservoir][:Pressure][i-get_nn(state)] = T(v)
    end
end

function setindex!(state::jutulSimpleState{T}, v, i::Int) where T
    if i <= get_nn(state)
        ## setindex for saturation
        state.state[:Saturations][1,i] = T(v)
        state.state[:Saturations][2,i] = T(1) - T(v)
    else
        ## setindex for pressure
        state.state[:Pressure][i-get_nn(state)] = T(v)
    end
end

function getindex(states::jutulStates, i::Int)
    idx_t = div(i-1, get_nn(states)) + 1
    idx_n = mod(i-1, get_nn(states)) + 1
    if idx_t <= get_nt(states)
        ## getindex for saturation
        return states.states[idx_t].state[:Reservoir][:Saturations][1,idx_n]
    else
        ## getindex for pressure
        return states.states[idx_t-get_nt(states)].state[:Reservoir][:Pressure][idx_n]
    end
end

function getindex(states::jutulSimpleStates, i::Int)
    idx_t = div(i-1, get_nn(states)) + 1
    idx_n = mod(i-1, get_nn(states)) + 1
    if idx_t <= get_nt(states)
        ## getindex for saturation
        return states.states[idx_t].state[:Saturations][1,idx_n]
    else
        ## getindex for pressure
        return states.states[idx_t-get_nt(states)].state[:Pressure][idx_n]
    end
end

function setindex!(states::jutulStates{T}, v, i::Int) where T
    idx_t = div(i-1, get_nn(states)) + 1
    idx_n = mod(i-1, get_nn(states)) + 1
    if idx_t <= get_nt(states)
        ## setindex for saturation
        states.states[idx_t].state[:Reservoir][:Saturations][1,idx_n] = T(v)
    else
        ## setindex for pressure
        states.states[idx_t-get_nt(states)].state[:Reservoir][:Pressure][idx_n] = T(v)
    end
end

function setindex!(states::jutulSimpleStates{T}, v, i::Int) where T
    idx_t = div(i-1, get_nn(states)) + 1
    idx_n = mod(i-1, get_nn(states)) + 1
    if idx_t <= get_nt(states)
        ## setindex for saturation
        states.states[idx_t].state[:Saturations][1,idx_n] = T(v)
    else
        ## setindex for pressure
        states.states[idx_t-get_nt(states)].state[:Pressure][idx_n] = T(v)
    end
end

getindex(state::jutulAllState, i...) = getindex(vec(state), i...)

firstindex(A::jutulAllState) = 1
lastindex(A::jutulAllState) = length(A)

norm(A::jutulAllState, order::Real=2) = norm(vec(A), order)

dot(A::jutulAllState, B::jutulAllState) = dot(vec(A), vec(B))
dot(A::jutulAllState, B::AbstractArray) = dot(vec(A), vec(B))
dot(A::AbstractArray, B::jutulAllState) = dot(vec(A), vec(B))

function (states::jutulState{T})(x::AbstractArray) where T
    @assert length(states) == length(x)
    states_ = deepcopy(states)
    states_ .= T.(x)
    states_.state[:Reservoir][:Saturations][2,:] = T(1) .- states_.state[:Reservoir][:Saturations][1,:]
    return states_
end

function (states::jutulSimpleState{T})(x::AbstractArray) where T
    @assert length(states) == length(x)
    states_ = deepcopy(states)
    states_ .= T.(x)
    states_.state[:Saturations][2,:] = T(1) .- states_.state[:Saturations][1,:]
    return states_
end

function (states::jutulStates{T})(x::AbstractArray) where T
    @assert length(states) == length(x)
    states_ = deepcopy(states)
    states_ .= T.(x)
    for i = 1:get_nt(states)
        states_.states[i].state[:Reservoir][:Saturations][2,:] = T(1) .- states_.states[i].state[:Reservoir][:Saturations][1,:]
    end
    return states_
end

function (states::jutulSimpleStates{T})(x::AbstractArray) where T
    @assert length(states) == length(x)
    states_ = deepcopy(states)
    states_ .= T.(x)
    for i = 1:get_nt(states)
        states_.states[i].state[:Saturations][2,:] = T(1) .- states_.states[i].state[:Saturations][1,:]
    end
    return states_
end

function jutulSimpleState(M::jutulModel{D, T}; ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), g::T=T(10.0)) where {D, T}
    ## default state at time 0 with all water
    error("deprecated. Please just do it yourself.")
    Z = repeat((1:M.n[end])*M.d[end], inner = prod(M.n[1:2]))
    p0 = ρH2O * g * (Z .+ M.h) # rho * g * h
    state0 = setup_state(simple_model(M; ρCO2=ρCO2, ρH2O=ρH2O), Pressure = p0, Saturations = [0.0, 1.0])
    return jutulSimpleState(state0)
end

for op in [:+, :-, :*, :/]
    @eval function $(op)(A::jutulAllState{T}, B::jutulAllState{T}) where T
        return A(broadcast($(op), vec(A), vec(B)))
    end
end

==(A::jutulAllState{T}, B::jutulAllState{T}) where {T} = vec(A) == vec(B)

function check_valid_state(states::jutulState{T}) where T
    @assert all(isapprox.(sum(states.state[:Reservoir][:Saturations], dims=1),1; rtol=sqrt(eps(T))))
end

function check_valid_state(states::jutulSimpleState{T}) where T
    @assert all(isapprox.(sum(states.state[:Saturations], dims=1),1; rtol=sqrt(eps(T))))
end

function check_valid_state(states::jutulSimpleOrMultiModelStates{T}) where T
    for i = 1:get_nt(states)
        check_valid_state(states.states[i])
    end
end