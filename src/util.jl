##### Utility functions ######

"""
Utility function to divide the range `range` into `n` chunks
"""
function split_range(range::AbstractRange{T}, n) where T
    len = length(range)

    starts = len >= n ?
        round.(T, Base.range(first(range), stop=last(range)+1, length=n+1)) :
        vcat(collect(first(range):(last(range)+1)), zeros(T, n-len))

    map((x,y)->x:y, starts[1:end-1], starts[2:end] .- 1)
end

function split_range(r::AbstractRange{Char}, n)
    map((x) -> Char(first(x)):Char(last(x)), split_range(Int(first(r)):Int(last(r)), n))
end

"""
    split_range_interval(range, n)
split a range into pieces each of length `n` or lesser
"""
function split_range_interval(range, n)
    f = first(range)
    N = length(range)
    npieces = ceil(Int, N / n)

    ranges = Array{UnitRange}(npieces)
    for i=1:npieces
        ranges[i] = f:min(N, f+n-1)
        f += n
    end
    ranges
end


"""
Tree reduce
"""
function treereduce(f, xs)
    length(xs) == 1 && return xs[1]
    l = length(xs)
    m = div(l, 2)
    f(treereduce(f, xs[1:m]), treereduce(f, xs[m+1:end]))
end

const  ENABLE_DEBUG = true # Will remove code under @dbg

"""
Run a block of code only if DEBUG is true
"""
macro dbg(expr)
    if ENABLE_DEBUG
        esc(expr)
    end
end

if !isdefined(Base, :reduced_indices)
    function reduced_dims(x, dim)
        Base.reduced_dims(size(x), dim)
    end
else
    reduced_dims(x, dim) = Base.reduced_indices(axes(x), dim)
end

function treereducedim(op, xs::Array, dim::Int)
    l = size(xs, dim)
    colons = Any[Colon() for i=1:length(size(xs))]
    if dim > length(size(xs))
        return xs
    end
    ys = treereduce((x,y)->map(op, x,y), Any[(
        colons[dim] = [i];
        view(xs, colons...)
    ) for i=1:l])
    reshape(ys[:], reduced_dims(xs, dim))
end

function treereducedim(op, xs::Array, dim::Tuple)
    reduce((prev, d) -> treereducedim(op, prev, d), dim, init=xs)
end

function allslices(xs, n)
    idx = Any[Colon() for i in 1:ndims(xs)]
    [(idx[n] = j; view(xs, idx...)) for j in 1:size(xs, n)]
end

function treereduce_nd(fs, xs)
    n = ndims(xs)
    if n==1
        treereduce(fs[n], xs)
    else
        treereduce(fs[n], map(ys->treereduce_nd(fs, ys), allslices(xs, n)))
    end
end

function setindex(x::NTuple{N}, idx, v) where N
    map(ifelse, ntuple(x->idx === x, Val(N)), ntuple(x->v, Val(N)), x)
end
