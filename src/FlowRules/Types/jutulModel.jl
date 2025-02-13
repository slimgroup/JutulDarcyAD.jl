export jutulModel, CartesianMesh

struct jutulModel{D, T}
    n::NTuple{D, Int64}
    d::NTuple{D, T}
    ϕ::Vector{T}
    K::Union{Matrix{T}, T}
    h::T    # depth of the top grid
    pad::Bool
end

function jutulModel(n::NTuple{D, Int64}, d::NTuple{D, T}, ϕ::T, K::Union{Matrix{T}, T}; h::T=T(0), pad::Bool=true) where {D, T}
    ϕ_full = ϕ .* ones(T, n)
    ϕ = vec(ϕ_full)
    return jutulModel(n, d, ϕ, K, h, pad)
end

jutulModel(n::NTuple{D, Int64}, d::NTuple{D, T}, ϕ::Vector{T}, K::Union{Matrix{T}, T}) where {D, T} = jutulModel(n, d, ϕ, K, T(0))

display(M::jutulModel{D, T}) where {D, T} =
    println("$(D)D jutulModel with size $(M.n) and grid spacing $(M.d) at depth $(M.h) m")

CartesianMesh(M::jutulModel{D, T}) where {D, T} = CartesianMesh(M.n, M.d .* M.n)

==(A::jutulModel{D, T}, B::jutulModel{D, T}) where {D,T} = (A.n == B.n && A.d == B.d && A.ϕ == B.ϕ && A.K == B.K && A.h == B.h && A.pad == B.pad)