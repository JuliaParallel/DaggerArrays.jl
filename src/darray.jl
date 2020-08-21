"""
    DArray{T,N,F}(indices, subindices, chunks, concat)
    DArray(T, indices, subindices, chunks, [concat=cat])

An N-dimensional distributed array of element type T, with a concatenation function of type F.

# Arguments
- `T`: element type
- `indices::ArrayDomain{N}`: the whole ArrayDomain of the array
- `subindices::AbstractArray{ArrayDomain{N}, N}`: a `IndexBlocks` of the same dimensions as the array
- `chunks::AbstractArray{Union{Chunk,Thunk}, N}`: an array of chunks of dimension N
- `concat::F`: a function of type `F`. `concat(x, y; dims=d)` takes two chunks `x` and `y`
  and concatenates them along dimension `d`. `cat` is used by default.
"""
mutable struct DArray{T,N,F} <: AbstractArray{T, N}
    indices::ArrayDomain{N}
    subindices::AbstractArray{ArrayDomain{N}, N}
    chunks::AbstractArray{Union{Chunk,Thunk}, N}
    concat::F
    freed::Threads.Atomic{UInt8}
    function DArray{T,N,F}(indices, subindices, chunks, concat::Function) where {T, N,F}
        new(indices, subindices, chunks, concat, Threads.Atomic{UInt8}(0))
    end
end

function free_chunks(chunks)
    @sync for c in chunks
        if c isa Chunk{<:Any, DRef}
            # increment refcount on the master node
            @async free!(c.handle)
        elseif c isa Thunk
            free_chunks(c.inputs)
        end
    end
end

function free!(x::DArray)
    freed = Bool(Threads.atomic_cas!(x.freed, UInt8(0), UInt8(1)))
    !freed && @async Dagger.free_chunks(x.chunks)
    nothing
end

function DArray(T, indices::ArrayDomain{N},
             subindices::AbstractArray{ArrayDomain{N}, N},
             chunks::AbstractArray{<:Any, N}, concat=cat) where N
    DArray{T, N, typeof(concat)}(indices, subindices, chunks, concat)
end

indices(d::DArray) = d.indices
chunks(d::DArray) = d.chunks
subindices(d::DArray) = d.subindices
size(x::DArray) = size(indices(x))
stage(ctx, c::DArray) = c

function collect(ctx::Context, d::DArray; tree=false, options=nothing)
    a = compute(ctx, d; options=options)

    if isempty(d.chunks)
        return Array{eltype(d)}(undef, size(d)...)
    end

    dimcatfuncs = [(x...) -> d.concat(x..., dims=i) for i in 1:ndims(d)]
    if tree
        collect(treereduce_nd(delayed.(dimcatfuncs), a.chunks))
    else
        treereduce_nd(dimcatfuncs, asyncmap(collect, a.chunks))
    end
end

"""
`view` of a `DArray` chunk returns a `DArray` of thunks
"""
function Base.view(c::DArray, d)
    subchunks, subindices = lookup_parts(chunks(c), subindices(c), d)
    d1 = alignfirst(d)
    DArray(eltype(c), d1, subindices, subchunks)
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


"""
A DArray object may contain a thunk in it, in which case
we first turn it into a Thunk object and then compute it.
"""
function compute(ctx::Context, x::DArray; persist=true, options=nothing)
    thunk = thunkize(ctx, x, persist=persist)
    if isa(thunk, Thunk)
        compute(ctx, thunk; options=options)
    else
        x
    end
end

"""
If a DArray tree has a Thunk in it, make the whole thing a big thunk
"""
function thunkize(ctx::Context, c::DArray; persist=true)
    if any(istask, chunks(c))
        thunks = chunks(c)
        sz = size(thunks)
        dmn = indices(c)
        dmnchunks = subindices(c)
        if persist
            foreach(persist!, thunks)
        end
        Thunk(thunks...; meta=true) do results...
            t = eltype(results[1])
            DArray(t, dmn, dmnchunks,
                                  reshape(Union{Chunk,Thunk}[results...], sz))
        end
    else
        c
    end
end

function distribute(x::AbstractArray, subindices)
    if isa(x, DArray)
        # distributing a dsitributed array
        if subindices == subindices(x)
            return x # already properly distributed
        end

        Nd = ndims(x)
        T = eltype(x)
        concat = x.concat
        cs = map(subindices) do idx
            chunks = cached_stage(ctx, x[idx]).chunks
            shape = size(chunks)
            (delayed() do shape, parts...
                if prod(shape) == 0
                    return Array{T}(undef, shape)
                end
                dimcatfuncs = [(x...) -> concat(x..., dims=i) for i in 1:length(shape)]
                ps = reshape(Any[parts...], shape)
                collect(treereduce_nd(dimcatfuncs, ps))
            end)(shape, chunks...)
        end
    else
        cs = map(c -> delayed(identity)(x[c]), d.subindices)
    end

    DArray(T,
           indices(x),
           subindices,
           cs
    )
end

function distribute(x::AbstractArray{T,N}, n::NTuple{N}) where {T,N}
    p = map((d, dn)->ceil(Int, d / dn), size(x), n)
    distribute(x, Blocks(p))
end

function distribute(x::AbstractVector, n::Int)
    distribute(x, (n,))
end

function distribute(x::AbstractVector, n::Vector{<:Integer})
    distribute(x, IndexBlocks((1,), (cumsum(n),)))
end

function Base.:(==)(x::DArray{T,N}, y::AbstractArray{S,N}) where {T,S,N}
    collect(x) == y
end

function Base.:(==)(x::AbstractArray{T,N}, y::DArray{S,N}) where {T,S,N}
    return collect(x) == y
end
