export jutulSource

struct jutulSource{D, T}
    irate::Vector{T}
    name::Vector{Symbol}
    loc::Vector{NTuple{D, T}}
end

jutulSource(irate::T, loc::NTuple{D, T}) where {D, T} = jutulSource(irate, [loc])
jutulSource(irate::T, loc::Vector{NTuple{D, T}}) where {D, T} = jutulSource([(-T(1))^(i+1)*irate for i = 1:length(loc)], vcat(:Injector, [:Producer for i = 2:length(loc)]), loc)

display(f::jutulSource{D, T}) where {D, T} =
    println("$(D)D jutulSource structure with $(length(f.loc)) injection/production wells and rate $(f.irate[1]) m^3/s")

==(A::jutulSource{D, T}, B::jutulSource{D, T}) where {D,T} = (A.irate == B.irate && A.name == B.name && A.loc == B.loc)