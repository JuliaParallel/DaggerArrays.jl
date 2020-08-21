import Base: ndims, size, getindex, length, isempty

###### Array Domains ######

struct ArrayDomain{N}
    indexes::NTuple{N, Any}
end

ArrayDomain(xs...) = ArrayDomain(xs)
ArrayDomain(xs::Array) = ArrayDomain((xs...,))

indexes(a::ArrayDomain) = a.indexes
chunks(a::ArrayDomain{N}) where {N} = IndexBlocks(
    ntuple(i->first(indexes(a)[i]), Val(N)), map(x->[length(x)], indexes(a)))

Base.:(==)(a::ArrayDomain, b::ArrayDomain) = indexes(a) == indexes(b)
Base.getindex(arr::AbstractArray, d::ArrayDomain) = arr[indexes(d)...]

function getindex(a::ArrayDomain, b::ArrayDomain)
    ArrayDomain(map(getindex, indexes(a), indexes(b)))
end

"""
    alignfirst(a) -> ArrayDomain

Make a subindices a standalone indices.

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
    indices(x::AbstractArray) -> ArrayDomain

The indices of an array is an ArrayDomain.
"""
indices(x::AbstractArray) = ArrayDomain([1:l for l in size(x)])

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
            throw(DimensionMismatch("Blocked indicess being concatenated have different distributions along dimension $i"))
        end
    end
    output = Any[x.cumlength...]
    output[dims] = merge_cumsums(x.cumlength[dims], y.cumlength[dims])
    IndexBlocks(x.start, (output...,))
end

### Lookup parts ###

function project(a::ArrayDomain, b::ArrayDomain)
    map(indexes(a), indexes(b)) do p, q
        q .- (first(p) - 1)
    end |> ArrayDomain
end

_cumsum(x::AbstractArray) = length(x) == 0 ? Int[] : cumsum(x)
function lookup_parts(ps::AbstractArray, subdmns::IndexBlocks{N}, d::ArrayDomain{N}) where N
    groups = map(group_indices, subdmns.cumlength, indexes(d))
    sz = map(length, groups)
    pieces = Array{Union{Chunk,Thunk}}(undef, sz)
    for i = CartesianIndices(sz)
        idx_and_dmn = map(getindex, groups, i.I)
        idx = map(x->x[1], idx_and_dmn)
        dmn = ArrayDomain(map(x->x[2], idx_and_dmn))
        pieces[i] = delayed(getindex)(ps[idx...], project(subdmns[idx...], dmn))
    end
    out_cumlength = map(g->_cumsum(map(x->length(x[2]), g)), groups)
    out_dmn = IndexBlocks(ntuple(x->1,Val(N)), out_cumlength)
    pieces, out_dmn
end

function group_indices(cumlength, idxs,at=1, acc=Any[])
    at > length(idxs) && return acc
    f = idxs[at]
    fidx = searchsortedfirst(cumlength, f)
    current_block = (get(cumlength, fidx-1,0)+1):cumlength[fidx]
    start_at = at
    end_at = at
    for i=(at+1):length(idxs)
        if idxs[i] in current_block
            end_at += 1
            at += 1
        else
            break
        end
    end
    push!(acc, fidx=>idxs[start_at:end_at])
    group_indices(cumlength, idxs, at+1, acc)
end

function group_indices(cumlength, idx::Int)
    group_indices(cumlength, [idx])
end

function group_indices(cumlength, idxs::AbstractRange)
    f = searchsortedfirst(cumlength, first(idxs))
    l = searchsortedfirst(cumlength, last(idxs))
    out = cumlength[f:l]
    isempty(out) && return []
    out[end] = last(idxs)
    map(=>, f:l, map(UnitRange, vcat(first(idxs), out[1:end-1].+1), out))
end

##### Partitioning #####

struct Blocks{N}
    blocksize::NTuple{N, Int}
end
Blocks(xs::Int...) = Blocks(xs)


function _cumlength(len, step)
    nice_pieces = div(len, step)
    extra = rem(len, step)
    ps = [step for i=1:nice_pieces]
    cumsum(extra > 0 ? vcat(ps, extra) : ps)
end

function partition(dom::ArrayDomain, p::Blocks)
    IndexBlocks(map(first, indexes(dom)),
        map(_cumlength, map(length, indexes(dom)), p.blocksize))
end

