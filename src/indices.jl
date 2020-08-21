import Base: ndims, size, getindex

###### Array Domains ######

struct ArrayDomain{N}
    indexes::NTuple{N, Any}
end

ArrayDomain(xs...) = ArrayDomain(xs)
ArrayDomain(xs::Array) = ArrayDomain((xs...,))

indexes(a::ArrayDomain) = a.indexes
chunks(a::ArrayDomain{N}) where {N} = IndexBlocks(
    ntuple(i->first(indexes(a)[i]), Val(N)), map(x->[length(x)], indexes(a)))

(==)(a::ArrayDomain, b::ArrayDomain) = indexes(a) == indexes(b)
Base.getindex(arr::AbstractArray, d::ArrayDomain) = arr[indexes(d)...]

function intersect(a::ArrayDomain, b::ArrayDomain)
    if a === b
        return a
    end
    ArrayDomain(map((x, y) -> _intersect(x, y), indexes(a), indexes(b)))
end

function project(a::ArrayDomain, b::ArrayDomain)
    map(indexes(a), indexes(b)) do p, q
        q .- (first(p) - 1)
    end |> ArrayDomain
end

function getindex(a::ArrayDomain, b::ArrayDomain)
    ArrayDomain(map(getindex, indexes(a), indexes(b)))
end

"""
    alignfirst(a) -> ArrayDomain

Make a subdomain a standalone domain.

# Example
```julia-repl
julia> alignfirst(ArrayDomain(11:25, 21:100))
ArrayDomain((1:15), (1:80))
```
"""
alignfirst(a::ArrayDomain) =
    ArrayDomain(map(r->1:length(r), indexes(a)))

function size(a::ArrayDomain, dim)
    idxs = indexes(a)
    length(idxs) < dim ? 1 : length(idxs[dim])
end
size(a::ArrayDomain) = map(length, indexes(a))
length(a::ArrayDomain) = prod(size(a))
ndims(a::ArrayDomain) = length(size(a))
isempty(a::ArrayDomain) = length(a) == 0


"""
    domain(x::AbstractArray) -> ArrayDomain

The domain of an array is an ArrayDomain.
"""
domain(x::AbstractArray) = ArrayDomain([1:l for l in size(x)])

struct IndexBlocks{N} <: AbstractArray{ArrayDomain{N}, N}
    start::NTuple{N, Int}
    cumlength::Tuple
end
Base.@deprecate_binding BlockedDomains IndexBlocks

ndims(x::IndexBlocks{N}) where {N} = N
size(x::IndexBlocks) = map(length, x.cumlength)
function _getindex(x::IndexBlocks{N}, idx::Tuple) where N
    starts = map((vec, i) -> i == 0 ? 0 : getindex(vec,i), x.cumlength, map(x->x-1, idx))
    ends = map(getindex, x.cumlength, idx)
    ArrayDomain(map(UnitRange, map(+, starts, x.start), map((x,y)->x+y-1, ends, x.start)))
end

function getindex(x::IndexBlocks{N}, idx::Int) where N
    if N == 1
        _getindex(x, (idx,))
    else
        _getindex(x, ind2sub(x, idx))
    end
end

getindex(x::IndexBlocks, idx::Int...) = _getindex(x,idx)

Base.IndexStyle(::Type{<:IndexBlocks}) = IndexCartesian()

function transpose(x::IndexBlocks{2})
    IndexBlocks(reverse(x.start), reverse(x.cumlength))
end
function transpose(x::IndexBlocks{1})
    IndexBlocks((1, x.start[1]), ([1], x.cumlength[1]))
end

function Base.adjoint(x::IndexBlocks{2})
    IndexBlocks(reverse(x.start), reverse(x.cumlength))
end
function Base.adjoint(x::IndexBlocks{1})
    IndexBlocks((1, x.start[1]), ([1], x.cumlength[1]))
end

function (*)(x::IndexBlocks{2}, y::IndexBlocks{2})
    if x.cumlength[2] != y.cumlength[1]
        throw(DimensionMismatch("Block distributions being multiplied are not compatible"))
    end
    IndexBlocks((x.start[1],y.start[2]), (x.cumlength[1], y.cumlength[2]))
end

function (*)(x::IndexBlocks{2}, y::IndexBlocks{1})
    if x.cumlength[2] != y.cumlength[1]
        throw(DimensionMismatch("Block distributions being multiplied are not compatible"))
    end
    IndexBlocks((x.start[1],), (x.cumlength[1],))
end

merge_cumsums(x,y) = vcat(x, y .+ x[end])

function Base.cat(x::IndexBlocks, y::IndexBlocks; dims::Int)
    N = max(ndims(x), ndims(y))
    get_i(x,y, i) = length(x) <= i ? x[i] : length(y) <= i ? y[i] : Int[]
    for i=1:N
        i == dims && continue
        if get_i(x,y,i) != get_i(y,x,i)
            throw(DimensionMismatch("Blocked domains being concatenated have different distributions along dimension $i"))
        end
    end
    output = Any[x.cumlength...]
    output[dims] = merge_cumsums(x.cumlength[dims], y.cumlength[dims])
    IndexBlocks(x.start, (output...,))
end

Base.hcat(xs::IndexBlocks...) = cat(xs..., dims=2)
Base.vcat(xs::IndexBlocks...) = cat(xs..., dims=1)

function reduce(xs::IndexBlocks; dims)
    if dims isa Int
        IndexBlocks(xs.start,
                     setindex(xs.cumlength, dims, [1]))
    else
        reduce((a,d)->reduce(a,dims=d), dims, init=xs)
    end
end

cumulative_domains(x::IndexBlocks) = x
