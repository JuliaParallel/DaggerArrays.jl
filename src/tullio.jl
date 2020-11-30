# Accumulate reduced thunks in a vector, and then call
# unwrap_reduce to associatively reduce the thunks
struct ReduceThunks
    f
    ts::Vector
end

unwrap_reduce(r::ReduceThunks) = Dagger.treereduce(delayed(r.f), r.ts)
unwrap_reduce(r) = r

struct Zero end
Base.zero(::Type{Union{Thunk, Chunk, ReduceThunks}}) = ReduceThunks(identity,[])
Base.zero(::Type{Union{Zero, ArrayDomain}}) = Zero()
Base.:(+)(z::Zero, x) = x
Base.:(+)(x, z::Zero) = x
Base.:(+)(z::Zero, x::Zero) = z

function map_calls(f, expr)
    if expr isa Expr
        args = map_calls(f, expr.args)
        if expr.head == :call
            return f(Expr(:call, args...))
        end
        return expr
    else
        return expr
    end
end

allequal(x) = nothing
allequal(x, y) = :($x == $y)
allequal(x, y, z, xs...) = :($x == $y && $(allequal(x, z, xs...)))

equal(x,y) = (@assert(x == y, (x,y)); x)
equal(x::Zero,y) = y
equal(y, z::Zero) = y
equal(x::Zero, z::Zero) = z

function combine_domains(leftind, constrs, rightdomains...)
    #allequal(consts)
    output_dims = first.(getindex.((constraints,), leftind))
end

Base.axes(d::ArrayDomain, i::Int) = i <= ndims(d) ? axes(d)[i] : Base.OneTo(1)

bcast(f, x::Zero, y) = y
bcast(f, x::Zero, y::Zero) = x
bcast(f, x, y::Zero) = x
bcast(f, x, y) = map(f, x, y)

bcast_f(f) = (x...) -> bcast(f, x...)

function __dtullio(dd)
    output_dims = first.(getindex.((dd.constraints,), dd.leftind))

    checks = filter(!isnothing, map(x->allequal(x...), values(dd.constraints)))
    checks = map(x->:(@assert $x), checks)


    ## Subdomains
    dT = Union{Zero, ArrayDomain}
    comp = :(Array{$dT}(undef, map(length, ($(output_dims...),))))
    domain_comprehension = :(fill!($comp, $Zero()))

    subinds = map(d -> :($d = $d.subindices), dd.arrays)

    dright = map_calls(dd.right) do c
        Expr(:call, :make_left_subdomain, c.args[2:end]...)
    end

    # if there's no function call on RHS
    dright = isequal(dright, dd.right) ? :(make_left_subdomain($(dd.right))) : dright

    subdomain_texpr = :($(dd.leftarray)[$(dd.leftind...)] = $(dright))

    ## Chunks
    cT = Union{Thunk, Chunk, ReduceThunks}
    comp = :(Array{$cT}(undef, map(length, ($(output_dims...),))))
    chunk_comprehension = :(fill!($comp, $ReduceThunks(identity, [])))

    chunks = map(d -> :($d = $d.chunks), dd.arrays)

    on_chunk_texpr = :($(dd.leftarray)[$(dd.leftind...)] := $(dd.right))

    cright = map_calls(dd.right) do c
        Expr(:call, :on_each, c.args[2:end]...)
    end

    cright = isequal(cright, dd.right) ? :(on_each($(dd.right))) : cright

    delayed_redfun = :((x, y)->$ReduceThunks($bcast_f($(dd.redfun)), vcat(x.ts, y)))

    chunks_texpr = :($(dd.leftarray)[$(dd.leftind...)] = $(cright))

    quote
        $(dd.leftarray) = let
            $(checks...)

            # within this let block the array symbols refer to the subinds
            subinds = let $(subinds...), $(dd.leftarray) = $domain_comprehension
                make_left_subdomain($(dd.arrays...)) = ArrayDomain($(output_dims...))
                equal = $equal
                DaggerArrays.Tullio.@tullio (equal) $subdomain_texpr threads=false grad=false
                $(dd.leftarray)
            end

            # within this let block the array symbols refer to the array of chunks
            chunks = let $(chunks...), $(dd.leftarray) = $chunk_comprehension
                on_each = delayed(($(dd.arrays...),) -> DaggerArrays.Tullio.@tullio $on_chunk_texpr grad=false)
                delayed_redfun = $delayed_redfun
                DaggerArrays.Tullio.@tullio (delayed_redfun) $chunks_texpr threads=false grad=false
                $unwrap_reduce.($(dd.leftarray))
            end

            # Make the output DArray
            DArray(Any, # todo
                   ArrayDomain($(output_dims...)),
            map(identity, subinds), # tighten type
            chunks,
            cat)
        end
    end
end

function _dtullio(expr...)
    dd = Tullio._tullio(expr..., :(lowered=true))
    __dtullio(dd)
end

macro dtullio(expr...)
    _dtullio(expr...) |> esc
end
