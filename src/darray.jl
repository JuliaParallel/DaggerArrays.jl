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

function Base.show(io::IO, ::MIME"text/plain", x::DArray)
    dims = ndims(x) == 1 ? "$(length(x))-element" : join(string.(size(x)), "×")
    cdims = join(string.(size(chunks(x))), "×")
    l = length(chunks(x))

    print(io, "$dims DArray{$(eltype(x)), $(ndims(x))} with $cdims chunks")
end

function Base.show(io::IO, x::DArray)
    show(io::IO, MIME("text/plain"), x)
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
        cs = map(c -> delayed(identity)(x[c]), subindices)
    end

    DArray(eltype(x),
           indices(x),
           subindices,
           cs
    )
end

function distribute(x::AbstractArray, blocks::Blocks)
    distribute(x, partition(indices(x), blocks))
end

function distribute(x::AbstractArray{T,N}, n::NTuple{N}) where {T,N}
    p = map((d, dn)->ceil(Int, d / dn), size(x), n)
    distribute(x, partition(indices(x), Blocks(p)))
end

function distribute(x::AbstractVector, n::Int)
    distribute(x, (n,))
end

function distribute(x::AbstractVector, n::Vector{<:Integer})
    distribute(x, IndexBlocks((1,), (cumsum(n),)))
end

Base.:(==)(x::DArray, y::AbstractArray) = collect(x) == y

Base.:(==)(x::AbstractArray, y::DArray) = collect(x) == y

# Getindex

"""
`view` of a `DArray` chunk returns a `DArray` of thunks
"""
function Base.view(c::DArray, d)
    subchunks, subinds = lookup_parts(chunks(c), subindices(c), d)
    d1 = alignfirst(d)
    DArray(eltype(c), d1, subinds, subchunks)
end

function Base.getindex(c::DArray, idx...)
    ranges = indices(c)
    idx′ = [if isa(idx[i], Colon)
        indexes(ranges)[i]
    else
        idx[i]
    end for i in 1:length(idx)]

    # Figure out output dimension
    view(c, ArrayDomain(idx′))
end

function Base.getindex(x::DArray, idx::Integer...)
    d = ArrayDomain(idx...)
    subchunks, subinds = lookup_parts(chunks(x), subindices(x), d)
    d1 = alignfirst(d)
    collect(delayed(x->first(x[indexes(first(subinds))...]))(subchunks[1]))
end

Base.getindex(c::DArray, idx::ArrayDomain) = c[indexes(idx)...]
